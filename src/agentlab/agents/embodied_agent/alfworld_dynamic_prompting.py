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


# TODO: 应该为TextWorld返回的文字
class Observation(dp.PromptElement):
    """Observation of the current step.

    Contains the html, the accessibility tree and the error logs.
    """

    def __init__(self, obs, flags: ObsFlags) -> None:
        super().__init__()
        self.flags = flags
        self.obs = obs

        self.env_description = obs["env_description"]

    def shrink(self):
        pass

    @property
    def _prompt(self) -> str:
        return f"""
# Observation of current step:
{self.env_description}

"""

    def add_screenshot(self, prompt: BaseMessage) -> BaseMessage:
        pass


class BeCautious(dp.PromptElement):
    def __init__(self, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self._prompt = f"""\
\nBe very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions first. \n"""  # TODO: 提示词有待进一步完善，比如：For example...，给agent举例子；


# TODO: 这个GoalInstructions还没有改，因为基本和alfworld中需要的差不多。但还可以润色改进
class GoalInstructions(dp.PromptElement):
    def __init__(self, goal_object, visible: bool = True, extra_instructions=None) -> None:
        super().__init__(visible)
        self._prompt = [
            dict(
                type="text",
                text=f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

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


# TODO: prompt设计待完成，针对alfworld的hints
class Hints(dp.PromptElement):
    """Not super useful and stale."""

    # NOTE: are these hints still relevant?
    _prompt = """\
Note:
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

    def __init__(self, info, action_flags) -> None:
        super().__init__()
        self.info = info
        self.action_flags = action_flags
        self.actions = self.info.get(info.get("admissible_commands", [[]])[0])
        self.actions = "\n".join(self.actions)
        self._prompt = "Available actions:\n" + self.actions + "\n"
        self._concrete_ex = """"""
        self._abstract_ex = f""""""

    def bleu_score(reference, candidate):
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()

        smoothie = SmoothingFunction().method4
        score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
        return score

    def _parse_answer(self, action, choices=None):
        """ """
        if not choices:
            return action

        try:
            action = parse_html_tags_raise(action, keys=["action"], merge_multiple=True)
        except ParseError as e:
            raise e

        if self.action_flags.is_strict:
            if action in choices:
                return action
            else:
                raise ParseError(
                    f"Action {action} is not in the admissible actions. Make sure your answer is restricted to the allowed actions."
                )
        else:
            bleus = [self.bleu_score(choice, action) for choice in choices]
            max_index = np.argmax(np.array(bleus))
            max_score = bleus[max_index]
            if max_score > self.action_flags.blue_limit:
                return choices[max_index]
            else:
                raise ParseError(
                    f"Action is not close enough to the admissible actions. The limit is set to {self.action_flags.blue_limit}."
                )


# TODO: prompt设计待完成，针对alfworld的thinking
class Think(dp.PromptElement):
    _prompt = ""

    _abstract_ex = """
<think>
Think step by step. 
</think>
"""
    _concrete_ex = """
<think>

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


def make_obs_preprocessor(flags: ObsFlags):
    def obs_mapping(obs: dict):
        obs = copy(obs)
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            extra_properties=obs["extra_element_properties"],
            with_visible=flags.extract_visible_tag,
            with_clickable=flags.extract_clickable_tag,
            with_center_coords=flags.extract_coords == "center",
            with_bounding_box_coords=flags.extract_coords == "box",
            filter_visible_only=flags.filter_visible_elements_only,
            filter_with_bid_only=flags.filter_with_bid_only,
            filter_som_only=flags.filter_som_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=obs["extra_element_properties"],
            with_visible=flags.extract_visible_tag,
            with_clickable=flags.extract_clickable_tag,
            with_center_coords=flags.extract_coords == "center",
            with_bounding_box_coords=flags.extract_coords == "box",
            filter_visible_only=flags.filter_visible_elements_only,
            filter_with_bid_only=flags.filter_with_bid_only,
            filter_som_only=flags.filter_som_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])
        obs["screenshot_som"] = overlay_som(
            obs["screenshot"], extra_properties=obs["extra_element_properties"]
        )

        return obs

    return obs_mapping


class Memory(dp.PromptElement):
    _prompt = ""  # provided in the abstract and concrete examples

    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions. Some tasks require to
remember hints from previous steps in order to solve it.
</memory>
"""

    _concrete_ex = """
<memory>
I clicked on bid "32" to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
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
Provide a multi step plan that will guide you to accomplish the goal. There
should always be steps to verify if the previous action had an effect. The plan
can be revisited at each steps. Specifically, if there was something unexpected.
The plan should be cautious and favor exploring befor submitting.
</plan>

<step>Integer specifying the step of current action
</step>
"""

    _concrete_ex = """
<plan>
1. fill form (failed)
    * type first name
    * type last name
2. Try to activate the form
    * click on tab 2
3. fill form again
    * type first name
    * type last name
4. verify and submit
    * verify form is filled
    * submit if filled, if not, replan
</plan>

<step>2</step>
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
could fail. Did your past actions had the expected effect? Make sure you're not
repeating the same mistakes.
</criticise>
"""

    _concrete_ex = """
<action_draft>
click("32")
</action_draft>

<criticise>
click("32") might not work because the element is not visible yet. I need to
explore the page to find a way to activate the form.
</criticise>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["action_draft", "criticise"])
