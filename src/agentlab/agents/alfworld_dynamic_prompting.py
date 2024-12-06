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

from agentlab.llm.llm_utils import (
    BaseMessage,
    ParseError,
    count_tokens,
    extract_code_blocks,
    image_to_jpg_base64_url,
    parse_html_tags_raise,
)


class Flags:
    """Base class for flags. Mostly for backward compatibility."""

    def copy(self):
        return deepcopy(self)

    def asdict(self):
        """Helper for JSON serializable requirement."""
        return asdict(self)

    @classmethod
    def from_dict(self, flags_dict):
        """Helper for JSON serializable requirement."""
        if isinstance(flags_dict, ObsFlags):
            return flags_dict

        if not isinstance(flags_dict, dict):
            raise ValueError(f"Unregcognized type for flags_dict of type {type(flags_dict)}.")
        return ObsFlags(**flags_dict)


# TODO: 设计alfworld的obs flags
@dataclass
class ObsFlags(Flags):
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


# TODO: 设计alfworld的action flags
@dataclass
class ActionFlags(Flags):
    pass


class PromptElement:
    """Base class for all prompt elements. Prompt elements can be hidden."""

    _prompt = ""
    _abstract_ex = ""
    _concrete_ex = ""

    def __init__(self, visible: bool = True) -> None:
        """Prompt element that can be hidden.

        Args:
            visible : bool, optional
                Whether the prompt element should be visible, by default True. Can
                be a callable that returns a bool. This is useful when a specific
                flag changes during a shrink iteration.
        """
        self._visible = visible

    @property
    def prompt(self) -> str | BaseMessage:
        """Avoid overriding this method. Override _prompt instead."""
        if self.is_visible:
            return self._prompt
        else:
            return ""

    @property
    def abstract_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide an abstract example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _abstract_ex instead

        Returns:
            str: The abstract example
        """
        if self.is_visible:
            return self._abstract_ex
        else:
            return ""

    @property
    def concrete_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide a concrete example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _concrete_ex instead

        Returns:
            str: The concrete example
        """
        if self.is_visible:
            return self._concrete_ex
        else:
            return ""

    @property
    def is_visible(self):
        """Handle the case where visible is a callable."""
        visible = self._visible
        if callable(visible):
            visible = visible()
        return visible

    def _parse_answer(self, text_answer):
        """Override to actually extract elements from the answer."""
        return {}

    def parse_answer(self, text_answer) -> dict:
        if self.is_visible:
            return self._parse_answer(text_answer)
        else:
            return {}


class Shrinkable(PromptElement, abc.ABC):
    @abc.abstractmethod
    def shrink(self) -> None:
        """Implement shrinking of this prompt element.

        You need to recursively call all shrinkable elements that are part of
        this prompt. You can also implement a shriking startegy for this prompt.
        Shrinking is can be called multiple times to progressively shrink the
        prompt until it fits max_tokens. Default max shrink iterations is 20.
        """
        pass


class Trunkater(Shrinkable):
    """Shrinkable element that truncates the prompt element from the bottom
    after a certain number of iterations."""

    def __init__(self, visible, shrink_speed=0.3, start_trunkate_iteration=10):
        super().__init__(visible=visible)
        self.shrink_speed = shrink_speed
        self.start_trunkate_iteration = start_trunkate_iteration
        self.shrink_calls = 0
        self.deleted_lines = 0

    def shrink(self) -> None:
        if self.is_visible and self.shrink_calls >= self.start_trunkate_iteration:
            # remove the fraction of _prompt
            lines = self._prompt.splitlines()
            new_line_count = int(len(lines) * (1 - self.shrink_speed))
            self.deleted_lines += len(lines) - new_line_count
            self._prompt = "\n".join(lines[:new_line_count])
            self._prompt += f"\n... Deleted {self.deleted_lines} lines to reduce prompt size."

        self.shrink_calls += 1


def fit_tokens(
    shrinkable: Shrinkable,
    max_prompt_tokens=None,
    max_iterations=20,
    model_name="openai/gpt-4",
    additional_prompts=[""],
):
    """Shrink a prompt element until it fits `max_prompt_tokens`.

    Args:
        shrinkable (Shrinkable): The prompt element to shrink.
        max_prompt_tokens (int): The maximum number of tokens allowed.
        max_iterations (int, optional): The maximum number of shrink iterations, by default 20.
        model_name (str, optional): The name of the model used when tokenizing.
        additional_prompts (str or List[str], optional): Additional prompts to account for when shrinking, by default [""].

    Returns:
        str: the prompt after shrinking.

    Raises:
        ValueError: Unrecognized type for prompt
    """

    if max_prompt_tokens is None:
        return shrinkable.prompt

    if isinstance(additional_prompts, str):
        additional_prompts = [additional_prompts]

    for prompt in additional_prompts:
        max_prompt_tokens -= count_tokens(prompt, model=model_name) + 1  # +1 because why not ?

    for _ in range(max_iterations):
        prompt = shrinkable.prompt
        if isinstance(prompt, str):
            prompt_str = prompt
        elif isinstance(prompt, list):
            # warn deprecated
            warn(
                "Using list of prompts is deprecated. Use a Discussion object instead.",
                DeprecationWarning,
            )
            prompt_str = "\n".join([p["text"] for p in prompt if p["type"] == "text"])
        elif isinstance(prompt, BaseMessage):
            prompt_str = str(prompt)
        else:
            raise ValueError(f"Unrecognized type for prompt: {type(prompt)}")
        n_token = count_tokens(prompt_str, model=model_name)
        if n_token <= max_prompt_tokens:
            return prompt
        shrinkable.shrink()

    logging.info(
        dedent(
            f"""\
            After {max_iterations} shrink iterations, the prompt is still
            {count_tokens(prompt_str)} tokens (greater than {max_prompt_tokens}). Returning the prompt as is."""
        )
    )
    return prompt


class Error(PromptElement):
    def __init__(self, error: str, visible: bool = True, prefix="", limit_logs=True) -> None:
        logs_separator = "Call log:"
        if limit_logs and logs_separator in error:
            error, logs = error.split(logs_separator)
            logs = "\n".join(logs.split("\n")[:10])
            error = error + f"\n{logs_separator}\n{logs}"

        super().__init__(visible=visible)
        self._prompt = f"\n{prefix}Error from previous action:\n{error}\n"


# TODO: 应该为TextWorld返回的文字
class Observation(Shrinkable):
    """Observation of the current step.

    Contains the html, the accessibility tree and the error logs.
    """

    def __init__(self, obs, flags: ObsFlags) -> None:
        super().__init__()
        self.flags = flags
        self.obs = obs

        self.error = Error(
            obs["last_action_error"],
            visible=lambda: flags.use_error_logs and obs["last_action_error"],
            prefix="## ",
        )

    def shrink(self):
        pass

    @property
    def _prompt(self) -> str:
        return f"""
# Observation of current step:
{self.error.prompt}

"""

    def add_screenshot(self, prompt: BaseMessage) -> BaseMessage:
        if self.flags.use_screenshot:
            if self.flags.use_som:
                screenshot = self.obs["screenshot_som"]
            else:
                screenshot = self.obs["screenshot"]
            img_url = image_to_jpg_base64_url(screenshot)
            prompt.add_image(img_url, detail=self.flags.openai_vision_detail)
        return prompt


class BeCautious(PromptElement):
    def __init__(self, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self._prompt = f"""\
\nBe very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions first. For example
you can fill a few elements of a form, but don't click submit before verifying
that everything was filled correctly.\n"""


class GoalInstructions(PromptElement):
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


class ChatInstructions(PromptElement):
    def __init__(self, chat_messages, visible: bool = True, extra_instructions=None) -> None:
        super().__init__(visible)
        self._prompt = f"""\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, in which the user gives you instructions and in which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Chat messages:

"""
        self._prompt += "\n".join(
            [
                f"""\
 - [{msg['role']}] UTC Time: {time.asctime(time.gmtime(msg['timestamp']))} - Local Time: {time.asctime(time.localtime(msg['timestamp']))} - {msg['message']}"""
                for msg in chat_messages
            ]
        )

        if extra_instructions:
            self._prompt += f"""

## Extra instructions:

{extra_instructions}
"""


class Hints(PromptElement):
    """Not super useful and stale."""

    # NOTE: are these hints still relevant?
    _prompt = """\
Note:
* Some tasks may be game like and may require to interact with the mouse position
in x, y coordinates.
* Some text field might have auto completion. To see it, you have to type a few
characters and wait until next step.
* If you have to cut and paste, don't forget to select the text first.
* Coordinate inside an SVG are relative to it's top left corner.
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky,
sometimes you need to use select_option, while other times you need to use fill
or click and wait for the reaction of the page.
"""


class SystemPrompt(PromptElement):
    _prompt = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You can interact with the page and explore, and send messages to the user. Each time you
submit an action it will be sent to the browser and you will receive a new page."""


class ActionPrompt(PromptElement):

    _concrete_ex = """
<action>
click('a324')
</action>
"""

    def __init__(self, action_set: AbstractActionSet, action_flags: ActionFlags) -> None:
        super().__init__()
        self.action_set = action_set
        self.action_flags = action_flags
        action_set_generic_info = """\
Note: This action set allows you to interact with your environment. Most of them
are python function executing playwright code. The primary way of referring to
elements in the page is through bid which are specified in your observations.

"""
        action_description = action_set.describe(
            with_long_description=action_flags.long_description,
            with_examples=action_flags.individual_examples,
        )
        self._prompt = (
            f"# Action space:\n{action_set_generic_info}{action_description}{MacNote().prompt}\n"
        )
        self._abstract_ex = f"""
<action>
{self.action_set.example_action(abstract=True)}
</action>
"""

    #         self._concrete_ex = f"""
    # <action>
    # {self.action_set.example_action(abstract=False)}
    # </action>
    # """

    def _parse_answer(self, text_answer):
        """
        Parses the provided text answer to extract an "action" from HTML tags or code blocks.

        This method attempts to parse a given text input (`text_answer`) and extract the "action" key,
        which is expected to be in HTML format or within code blocks. The method handles potential parsing
        errors based on the strictness level defined in `self.action_flags`. If `action` is successfully
        extracted, it checks if the action can be mapped to Python code for further use.

        Args:
            text_answer (str): The input string containing the HTML-formatted or code block-defined action.

        Returns:
            dict: A dictionary containing the parsed action. If there was a parsing error and code blocks
                  were found, includes an additional "parse_error" message.

        Raises:
            ParseError: If parsing fails and the strict flag is set, or if the action does not map to
                        allowed actions, a ParseError is raised with an appropriate message.
        """
        try:
            ans_dict = parse_html_tags_raise(text_answer, keys=["action"], merge_multiple=True)
        except ParseError as e:
            if self.action_flags.is_strict:
                raise e
            else:
                # try to extract code blocks
                blocks = extract_code_blocks(text_answer)
                if len(blocks) == 0:
                    raise e
                else:
                    code = "\n".join([block for _, block in blocks])
                    ans_dict = {"action": code, "parse_error": str(e)}

        try:
            if ans_dict["action"] == "None":
                # Used by reproducibility agent for backward compatibility of
                # traces missing LLM's response in chat messages.
                ans_dict["action"] = None
            else:
                # just check if action can be mapped to python code but keep action as is
                # the environment will be responsible for mapping it to python
                self.action_set.to_python_code(ans_dict["action"])
        except Exception as e:
            raise ParseError(
                f"Error while parsing action\n: {e}\n"
                "Make sure your answer is restricted to the allowed actions."
            )

        return ans_dict


# def make_action_set(action_flags: ActionFlags) -> AbstractActionSet:

#     if action_flags.action_set == "python":
#         action_set = PythonActionSet(strict=action_flags.is_strict)
#         if action_flags.demo_mode != "off":
#             warn(
#                 f'Action_set "python" is incompatible with demo_mode={repr(action_flags.demo_mode)}.'
#             )
#         return action_set

#     action_set = HighLevelActionSet(
#         subsets=list(set(["chat"] + ["infeas"] + action_flags.action_set.split("+"))),
#         multiaction=action_flags.multi_actions,
#         strict=action_flags.is_strict,
#         demo_mode=action_flags.demo_mode,
#     )

#     return action_set


class Think(PromptElement):
    _prompt = ""

    _abstract_ex = """
<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>
"""
    _concrete_ex = """
<think>
From previous action I tried to set the value of year to "2022",
using select_option, but it doesn't appear to be in the form. It may be a
dynamic dropdown, I will try using click with the bid "a324" and look at the
response from the page.
</think>
"""

    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["think"], merge_multiple=True)
        except ParseError as e:
            return {"think": text_answer, "parse_error": str(e)}


# def diff(previous, new):
#     """Return a string showing the difference between original and new.

#     If the difference is above diff_threshold, return the diff string."""

#     if previous == new:
#         return "Identical", []

#     if len(previous) == 0 or previous is None:
#         return "previous is empty", []

#     diff_gen = difflib.ndiff(previous.splitlines(), new.splitlines())

#     diff_lines = []
#     plus_count = 0
#     minus_count = 0
#     for line in diff_gen:
#         if line.strip().startswith("+"):
#             diff_lines.append(line)
#             plus_count += 1
#         elif line.strip().startswith("-"):
#             diff_lines.append(line)
#             minus_count += 1
#         else:
#             continue

#     header = f"{plus_count} lines added and {minus_count} lines removed:"

#     return header, diff_lines


# class Diff(Shrinkable):
#     def __init__(
#         self, previous, new, prefix="", max_line_diff=20, shrink_speed=2, visible=True
#     ) -> None:
#         super().__init__(visible=visible)
#         self.previous = previous
#         self.new = new
#         self.max_line_diff = max_line_diff
#         self.shrink_speed = shrink_speed
#         self.prefix = prefix

#     def shrink(self):
#         self.max_line_diff -= self.shrink_speed
#         self.max_line_diff = max(1, self.max_line_diff)

#     @property
#     def _prompt(self) -> str:
#         header, diff_lines = diff(self.previous, self.new)

#         diff_str = "\n".join(diff_lines[: self.max_line_diff])
#         if len(diff_lines) > self.max_line_diff:
#             original_count = len(diff_lines)
#             diff_str = f"{diff_str}\nDiff truncated, {original_count - self.max_line_diff} changes now shown."
#         return f"{self.prefix}{header}\n{diff_str}\n"


class HistoryStep(Shrinkable):
    def __init__(
        self, previous_obs, current_obs, action, memory, thought, flags: ObsFlags, shrink_speed=1
    ) -> None:
        super().__init__()
        # self.html_diff = Diff(
        #     previous_obs[flags.html_type],
        #     current_obs[flags.html_type],
        #     prefix="\n### HTML diff:\n",
        #     shrink_speed=shrink_speed,
        #     visible=lambda: flags.use_html and flags.use_diff,
        # )
        # self.ax_tree_diff = Diff(
        #     previous_obs["axtree_txt"],
        #     current_obs["axtree_txt"],
        #     prefix=f"\n### Accessibility tree diff:\n",
        #     shrink_speed=shrink_speed,
        #     visible=lambda: flags.use_ax_tree and flags.use_diff,
        # )
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


class History(Shrinkable):
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