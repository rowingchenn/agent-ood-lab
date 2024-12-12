from dataclasses import dataclass

from .alfworld_agent_prompt import AlfworldPromptFlags

from browsergym.experiments.agent import Agent, AgentInfo
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.embodied_agent import alfworld_dynamic_prompting as dp
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.tracking import cost_tracker_decorator


@dataclass
class AlfworldAgentArgs(AgentArgs):
    """
    A class to represent various flags used to control features in an agent works in Alfworld.

    """

    chat_model_args: BaseModelArgs = None
    flags: AlfworldPromptFlags
    max_retry: int = 4

    def __post_init__(self):
        try:
            self.agent_name = f"AlfworldAgent-{self.chat_model_args.model_name_or_path}".replace(
                "/", "_"
            )
        except AttributeError:
            pass

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return AlfworldAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class AlfworldAgent(Agent):
    def __init__(
        self, chat_model_args: BaseModelArgs, flags: AlfworldPromptFlags, max_retry: int = 4
    ):
        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.flags = flags
        self.max_retry = max_retry

    def obs_preprocessor(self, obs):
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):
        pass

    def reset(self, seed=None):
        pass

    def run(self, info: AgentInfo):
        pass
