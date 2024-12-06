import logging
from dataclasses import dataclass

from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import alfworld_dynamic_prompting as dp

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

    obs: dp.AlfworldObsFlags
    action: dp.AlfworldActionFlags
    use_plan: bool = False
    use_criticise: bool = False
    use_thinking: bool = False
    use_concrete_example: bool = True
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True


# TODO: 设计alfworld的prompt
class AlfworldPrompt(dp.MainPrompt):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: AlfworldPromptFlags,
    ) -> None:
        super().__init__(
            action_set, obs_history, actions, memories, thoughts, previous_plan, step, flags
        )
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = dp.GoalInstructions(
                obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(obs_history[-1], flags.obs)

        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

        self.be_cautious = dp.BeCautious(visible=lambda: flags.be_cautious)
        self.think = dp.Think(visible=lambda: flags.use_thinking)
        self.hints = dp.Hints(visible=lambda: flags.use_hints)
        self.plan = dp.Plan(previous_plan, step, lambda: flags.use_plan)  # TODO add previous plan
        self.criticise = dp.Criticise(visible=lambda: flags.use_criticise)  # TODO
        self.memory = dp.Memory(visible=lambda: flags.use_memory)  # TODO

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.criticise.prompt}\
{self.be_cautious.prompt}\
{self.plan.prompt}\
{self.memory.prompt}\
{self.thinking.prompt}\
"""
        )

        if self.flags.use_concrete_example:
            prompt.add_text(
                f"""\
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.thinking.concrete_ex}\
{self.plan.concrete_ex}\
{self.memory.concrete_ex}\
{self.criticise.concrete_ex}\
{self.action_prompt.concrete_ex}\
"""
            )
        if self.flags.use_abstract_example:
            prompt.add_text(self.action_prompt.abstract_ex)
        return prompt
