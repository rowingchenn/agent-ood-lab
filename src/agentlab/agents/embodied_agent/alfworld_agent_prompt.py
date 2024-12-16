import logging
from dataclasses import dataclass

from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.embodied_agent import alfworld_dynamic_prompting as adp

from agentlab.llm.llm_utils import HumanMessage


@dataclass
class AlfworldPromptFlags(dp.Flags):
    """
    A class to represent various flags used to control features in an agent works in Alfworld.

    Attributes:
        use_plan (bool): Ask the agent to provide a plan.
        use_criticise (bool): Ask the agent to first draft and criticise the action before producing it.
        use_thinking (bool): Enable a chain of thoughts.
        use_concrete_example (bool): Use a concrete example of the answer in the prompt for a generic task.
        use_hints (bool): Add some human-engineered hints to the prompt.
        enable_chat (bool): Enable chat mode, where the agent can interact with the user.
        max_prompt_tokens (int): Maximum number of tokens allowed in the prompt.
        be_cautious (bool): Instruct the agent to be cautious about its actions.
        extra_instructions (Optional[str]): Extra instructions to provide to the agent.
        add_missparsed_messages (bool): When retrying, add the missparsed messages to the prompt.
    """

    obs: adp.ObsFlags
    actions: adp.ActionFlags
    use_plan: bool = False
    use_criticise: bool = False
    use_thinking: bool = False
    use_memory: bool = False
    use_concrete_example: bool = True
    use_abstract_example: bool = True
    use_hints: bool = False
    enable_chat: bool = False  # Currently not supported
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    max_trunc_itr: int = 20


# TODO: 设计alfworld的prompt
class AlfworldPrompt(dp.Shrinkable):
    """
    Attributes:
        obs_history (list[dict]): The history of observations.
        actions (list[str]): The list of actions taken by the agent.
        memories (list[str]): The list of memories.
        thoughts (list[str]): The list of thoughts.
        previous_plan (str): The previous plan.
        step (int): The current step.
        flags (AlfworldPromptFlags): The flags used to control the features in the agent.

    obs keys:
        - "chat_messages"
        - "goal_object"
        - "environment_description"
        - "admissible_commands"
        - "last_action"
        - "last_action_error"
        - "elapsed_time"
    """

    def __init__(
        self,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: AlfworldPromptFlags,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = adp.History(obs_history, actions, memories, thoughts, flags.obs)
        if self.flags.enable_chat:
            # self.instructions = dp.ChatInstructions(
            #     obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            # )
            pass
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = adp.GoalInstructions(
                obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
            )

        self.obs = adp.Observation(obs_history[-1], flags.obs)
        self.action_prompt = adp.ActionPrompt(obs_history[-1], action_flags=flags.actions)
        self.be_cautious = adp.BeCautious(visible=lambda: flags.be_cautious)
        self.think = adp.Think(visible=lambda: flags.use_thinking)
        self.hints = adp.Hints(visible=lambda: flags.use_hints)
        self.plan = adp.Plan(previous_plan, step, lambda: flags.use_plan)  # TODO add previous plan
        self.criticise = adp.Criticise(visible=lambda: flags.use_criticise)  # TODO
        self.memory = adp.Memory(visible=lambda: flags.use_memory)  # TODO

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.hints.prompt}\
{self.be_cautious.prompt}\
{self.think.prompt}\
{self.plan.prompt}\
{self.memory.prompt}\
{self.criticise.prompt}\
"""
        )

        if self.flags.use_concrete_example:
            prompt.add_text(
                f"""\
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.think.concrete_ex}\
{self.plan.concrete_ex}\
{self.memory.concrete_ex}\
{self.criticise.concrete_ex}\
{self.action_prompt.concrete_ex}\
"""
            )
        if self.flags.use_abstract_example:
            prompt.add_text(
                f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.think.abstract_ex}\
{self.plan.abstract_ex}\
{self.memory.abstract_ex}\
{self.criticise.abstract_ex}\
{self.action_prompt.abstract_ex}\
"""
            )

        return prompt

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    # currently, we only parse the action and think prompt for alfworld
    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.think.parse_answer(text_answer))
        # ans_dict.update(self.plan.parse_answer(text_answer))
        # ans_dict.update(self.memory.parse_answer(text_answer))
        # ans_dict.update(self.criticise.parse_answer(text_answer))
        ans_dict.update(self.action_prompt.parse_answer(text_answer))
        return ans_dict
