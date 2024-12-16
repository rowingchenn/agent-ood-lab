import abc
import logging
import platform
import time
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from textwrap import dedent
from typing import Literal
from warnings import warn

import bgym
from browsergym.core.action.base import AbstractActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, overlay_som, prune_html
from embodiedgym.core.env import OOD_ACTION

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import (
    BaseMessage,
    ParseError,
    count_tokens,
    extract_code_blocks,
    image_to_jpg_base64_url,
    parse_html_tags_raise,
)

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# TODO: 设计alfworld的obs flags
@dataclass
class ObsFlags(dp.Flags):
    """
    A class to represent various flags used to control features of observation in Alfworld.

    Attributes:
        use_error_logs (bool): Expose the previous error in the prompt.
        use_history (bool): Enable history of previous steps in the prompt.
        use_past_error_logs (bool): If use_history is True, expose all previous errors in the history.
        use_action_history (bool): If use_history is True, include the actions in the history.
        use_think_history (bool): If use_history is True, include all previous chains of thoughts in the history.
    """

    use_error_logs: bool = False
    use_history: bool = False
    use_past_error_logs: bool = False
    use_action_history: bool = False
    use_think_history: bool = False

    # include_pddl_state: bool = True
    # include_high_level_plan: bool = True
    # include_scene_description: bool = True


# TODO: 目前只有is_strict这一个flag，主要是因为alfworld获取actions都是每一步当场获取admisible_commands
# 没办法像BrowserGym那样固定一个action_set并配置prompt
@dataclass
class ActionFlags(dp.Flags):
    """
    Attributes:
        is_strict (bool): If True, the agent must choose from the admissible actions.
        blue_limit (float): When is_strict is False, the minimum BLEU score required to match the action to the admissible actions.
    """

    is_strict: bool = None
    blue_limit: float = 0.8


class Error(dp.PromptElement):
    def __init__(self, error: str, visible: bool = True, prefix="", limit_logs=True) -> None:
        logs_separator = "Call log:"
        if limit_logs and logs_separator in error:
            error, logs = error.split(logs_separator)
            logs = "\n".join(logs.split("\n")[:10])
            error = error + f"\n{logs_separator}\n{logs}"

        super().__init__(visible=visible)
        self._prompt = f"\n{prefix}Error from previous action:\n{error}\n"


# TODO: 应该为每一步TextWorld返回的文字中，对环境描述的信息。
class Observation(dp.PromptElement):
    """Observation of the current step."""

    def __init__(self, obs, flags: ObsFlags) -> None:
        super().__init__()
        self.flags = flags
        self.obs = obs
        self.env_description = obs["environment_description"]

    @property
    def _prompt(self) -> str:
        prompt = "# Observation of current step:\n"
        prompt += self.env_description

        return prompt

    def shrink(self):
        pass


class BeCautious(dp.PromptElement):
    def __init__(self, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self._prompt = f"""\
\nBe very cautious. Avoid submitting any action before verifying its potential effect. 
Take time to explore safe actions and analyze the environment thoroughly. 

Guidelines:
1. Verify the preconditions of your action. Ensure all necessary objects and conditions are in place.
2. Anticipate the consequences of your action. Consider how it impacts the overall task progress.
3. Avoid repeating past mistakes. Review your action history and any errors before deciding.
4. If unsure, favor exploratory actions to gather more information.
5. Always check the state of the environment after performing an action to ensure it aligns with your expectations.

Example:
- If you plan to pick up an object, ensure it is reachable and not obstructed.
- Before opening a container, verify that it is not locked.
\n"""


# TODO: 这个GoalInstructions还没有改，因为基本和alfworld中需要的差不多。但还可以润色改进
class GoalInstructions(dp.PromptElement):
    def __init__(self, goal_object, visible: bool = True, extra_instructions=None) -> None:
        super().__init__(visible)
        self._prompt = [
            dict(
                type="text",
                text=f"""\
# Goal Instructions
Review the current environment state and provided information to determine the best possible next action.
I will give you a set of admissible actions you can take. Remember you can only take one action at a time. And only use actions listed after ''.
Your response will be executed by a program, so ensure it adheres to the required formatting.

## Goal:
""",
            )
        ]

        self._prompt += goal_object

        if extra_instructions:
            self._prompt += [
                dict(
                    type="text",
                    text=f"""

## Extra instructions:

{extra_instructions}
""",
                )
            ]


# TODO: 当前设计ChatInstructions还没有太大必要，后续可以考虑加入这个额外的功能，主要是对于alfworld来说
# 没有和用户聊天的必要，但是可以考虑加入一些额外的机制
class ChatInstructions(dp.PromptElement):
    def __init__(self, chat_messages, visible: bool = True, extra_instructions=None) -> None:
        super().__init__(visible)
        pass


class Hints(dp.PromptElement):
    _prompt = """\
Note:
* Carefully analyze the current environment and observe any clues in the scene description.
* Some tasks may require chaining multiple actions together to achieve a goal.
* Objects in the environment often need to be manipulated in a logical sequence. For example, open a container before retrieving an item inside.
* Pay close attention to the available actions and their arguments for choosing the most effective step.
* If an action fails, review previous observations or errors for possible reasons.
* Ensure that all required objects are in the correct state before proceeding to subsequent actions.
* Some tasks may involve exploration. Consider trying safe actions to gather more information.
"""


# TODO: prompt有待润色
class SystemPrompt(dp.PromptElement):
    _prompt = """\
You are an agent tasked with completing a household task in an interactive environment
based on textual observations and user instructions. You can interact with objects, 
navigate between locations, and modify the environment. Each time you submit an action, 
it will be executed in the environment, and you will receive updated textual observations.
Your goal is to complete the task efficiently and accurately while reasoning about the actions and their effects in the environment."""


class ActionPrompt(dp.PromptElement):

    def __init__(self, obs, action_flags) -> None:
        super().__init__()
        self.obs = obs
        self.action_flags = action_flags
        action_set_generic_info = """\
Note: This action set allows you to interact with your environment. 
You must select only one action from the set and output it within <action></action> tags. 
Ensure your output includes no additional text or formatting outside the tags.\n
"""
        self.actions = self.obs.get("admissible_commands")
        self._prompt = f"# Admissible actions:\n{action_set_generic_info}\n{self.actions}\n"
        self._concrete_ex = """
<action>
go to cabinet 1
</action>
"""
        self._abstract_ex = f"""
<action>
Any action from the admissible actions. Or the extra action mentioned before.
</action>
"""

    def bleu_score(reference, candidate):
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()

        smoothie = SmoothingFunction().method4
        score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
        return score

    def _parse_answer(self, action: str):
        """
        action: str, the action to be parsed
        choices: list[str], the admissible actions provided by the environment
        """

        try:
            ans_dict = parse_html_tags_raise(action, keys=["action"], merge_multiple=True)
        except ParseError as e:
            if self.action_flags.is_strict:
                raise e
            else:
                pass  # 暂时未实现，也就是如果输出了<action>xxx[/action]这样的标签，且is_strict为False，我们应该怎么处理

        return ans_dict


# TODO: prompt设计待完成，针对alfworld的thinking
class Think(dp.PromptElement):
    _prompt = """
# Think:
Consider the current state of the environment and formulate a step-by-step reasoning process.
Explain why this step is necessary and how it contributes to the overall goal.
"""

    _abstract_ex = """
<think>
Think step by step. For example:
1. Observe the current state and identify necessary actions.
2. Break down the task into smaller, actionable steps.
3. Justify each step logically, ensuring alignment with the goal.
</think>
"""
    _concrete_ex = """
<think>
The task is to put some spraybottle on the toilet. To achieve this, 
I first need to locate a spraybottle within the room. 
Searching for the spraybottle is a critical first step because without it, the primary goal cannot be accomplished. 
By systematically exploring the room, I can identify its exact location and retrieve it. 
This step contributes to the overall goal by ensuring that I have the necessary item in hand before proceeding to the next actions, 
such as moving to the toilet and placing the spraybottle on it.
</think>
"""

    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["think"], merge_multiple=True)
        except ParseError as e:
            return {"think": text_answer, "parse_error": str(e)}


class HistoryStep(dp.Shrinkable):
    def __init__(
        self, previous_obs, current_obs, action, memory, thought, flags: ObsFlags, shrink_speed=1
    ) -> None:
        super().__init__()
        self.error = Error(
            current_obs["last_action_error"],
            visible=(
                lambda: flags.use_error_logs
                and current_obs["last_action_error"]
                and flags.use_past_error_logs
            ),
            prefix="### ",
        )
        self.shrink_speed = shrink_speed
        self.action = action
        self.memory = memory
        self.thought = thought
        self.flags = flags

    def shrink(self):
        super().shrink()
        # self.html_diff.shrink()
        # self.ax_tree_diff.shrink()

    @property
    def _prompt(self) -> str:
        prompt = ""

        if self.flags.use_think_history:
            prompt += f"\n<think>\n{self.thought}\n</think>\n"

        if self.flags.use_action_history:
            prompt += f"\n<action>\n{self.action}\n</action>\n"

        # prompt += f"{self.error.prompt}{self.html_diff.prompt}{self.ax_tree_diff.prompt}"
        prompt += f"{self.error.prompt}"

        if self.memory is not None:
            prompt += f"\n<memory>\n{self.memory}\n</memory>\n"

        return prompt


class History(dp.Shrinkable):
    def __init__(
        self, history_obs, actions, memories, thoughts, flags: ObsFlags, shrink_speed=1
    ) -> None:
        if memories is None:
            memories = [None] * len(actions)
        super().__init__(visible=lambda: flags.use_history)
        assert len(history_obs) == len(actions) + 1
        assert len(history_obs) == len(memories) + 1

        self.shrink_speed = shrink_speed
        self.history_steps: list[HistoryStep] = []

        for i in range(1, len(history_obs)):
            self.history_steps.append(
                HistoryStep(
                    history_obs[i - 1],
                    history_obs[i],
                    actions[i - 1],
                    memories[i - 1],
                    thoughts[i - 1],
                    flags,
                )
            )

    def shrink(self):
        """Shrink individual steps"""
        # TODO set the shrink speed of older steps to be higher
        super().shrink()
        for step in self.history_steps:
            step.shrink()

    @property
    def _prompt(self):
        prompts = ["# History of interaction with the task:\n"]
        for i, step in enumerate(self.history_steps):
            prompts.append(f"## step {i}")
            prompts.append(step.prompt)
        return "\n".join(prompts) + "\n"


# def make_obs_preprocessor(flags: ObsFlags):
# def obs_mapping(obs: dict):
#     obs = copy(obs)
#     obs["dom_txt"] = flatten_dom_to_str(
#         obs["dom_object"],
#         extra_properties=obs["extra_element_properties"],
#         with_visible=flags.extract_visible_tag,
#         with_clickable=flags.extract_clickable_tag,
#         with_center_coords=flags.extract_coords == "center",
#         with_bounding_box_coords=flags.extract_coords == "box",
#         filter_visible_only=flags.filter_visible_elements_only,
#         filter_with_bid_only=flags.filter_with_bid_only,
#         filter_som_only=flags.filter_som_only,
#     )
#     obs["axtree_txt"] = flatten_axtree_to_str(
#         obs["axtree_object"],
#         extra_properties=obs["extra_element_properties"],
#         with_visible=flags.extract_visible_tag,
#         with_clickable=flags.extract_clickable_tag,
#         with_center_coords=flags.extract_coords == "center",
#         with_bounding_box_coords=flags.extract_coords == "box",
#         filter_visible_only=flags.filter_visible_elements_only,
#         filter_with_bid_only=flags.filter_with_bid_only,
#         filter_som_only=flags.filter_som_only,
#     )
#     obs["pruned_html"] = prune_html(obs["dom_txt"])
#     obs["screenshot_som"] = overlay_som(
#         obs["screenshot"], extra_properties=obs["extra_element_properties"]
#     )

#     return obs


def make_obs_preprocessor(flags: ObsFlags):
    def preprocess_obs(obs: dict) -> dict:
        processed_obs = obs.copy()
        # 这里好像没什么需要处理的
        return processed_obs

    return preprocess_obs


class Memory(dp.PromptElement):
    _prompt = """
<memory>
Write down anything you need to remember for the next steps. Use this space to note
important details, observations, or actions that could guide your decision-making
in future steps.
</memory>
"""

    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions. Some tasks require to
remember hints from previous steps in order to solve it.
</memory>
"""

    _concrete_ex = """
<memory>
I picked up the alarm clock from the dresser. Next, I need to turn on the lamp
to complete the task of examining the clock under proper lighting.
</memory>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["memory"], merge_multiple=True)


class Plan(dp.PromptElement):
    def __init__(self, previous_plan, plan_step, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self.previous_plan = previous_plan
        self._prompt = f"""
# Plan:

You just executed step {plan_step} of the previously proposed plan:\n{previous_plan}\n
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary.
"""

    _abstract_ex = """
<plan>
Provide a step-by-step plan to accomplish the goal. The plan should include:
1. Logical actions in sequence.
2. Verification steps to ensure actions have the desired effect.
3. Adaptability to adjust the plan if something unexpected occurs.

For example:
1. Observe the environment.
2. Interact with an object (e.g., pick up the key).
3. Use the object to achieve the goal (e.g., unlock the door).
</plan>

<step>Integer specifying the current step of the plan.
</step>
"""

    _concrete_ex = """
<plan>
1. Locate the book on the shelf.
2. Retrieve the book and place it on the table.
3. Open the book to the specified page.
4. Verify that the information matches the goal requirements.
5. If the information is incorrect, replan and search for another source.
</plan>

<step>3</step>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["plan", "step"])


class Criticise(dp.PromptElement):
    _prompt = ""

    _abstract_ex = """
<action_draft>
Write a first version of what you think is the right action.
</action_draft>

<criticise>
Criticise action_draft. What could be wrong with it? Enumerate reasons why it
could fail. Did your past actions have the expected effect? Make sure you're not
repeating the same mistakes.
</criticise>
"""

    _concrete_ex = """
<action_draft>
pick_up("red_apple")
</action_draft>

<criticise>
The action 'pick_up("red_apple")' might fail because the apple is in a locked container.
I need to unlock the container first. Additionally, the action might fail if the apple
is not within reach or already in my inventory.
</criticise>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["action_draft", "criticise"])


def fit_tokens(
    shrinkable: dp.Shrinkable,
    max_prompt_tokens=None,
    max_iterations=20,
    model_name="openai/gpt-4",
    additional_prompts=[""],
):
    """
    Shrink a prompt element until it fits `max_prompt_tokens`.

    Currently Alfworld is too simple to need this function.
    """
    return shrinkable.prompt
