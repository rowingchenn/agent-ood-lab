from dataclasses import asdict, dataclass
import logging

from browsergym.experiments.agent import Agent, AgentInfo
from agentlab.agents.agent_args import AgentArgs

# from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.embodied_agent import alfworld_dynamic_prompting as adp
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.tracking import cost_tracker_decorator
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry


# from .alfworld_agent_prompt import AlfworldPrompt, AlfworldPromptFlags
from agentlab.agents.embodied_agent.alfworld_agent_prompt import AlfworldPrompt, AlfworldPromptFlags

logger = logging.getLogger(__name__)


@dataclass
class AlfworldAgentArgs(AgentArgs):
    """
    A class to represent various flags used to control features in an agent works in Alfworld.

    """

    chat_model_args: BaseModelArgs = None
    flags: AlfworldPromptFlags = None
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
        self.reset(seed=None)

    def obs_preprocessor(self, obs):
        # This is used to preprocess the observation for the agent
        # In BrowserGym, here we need to preprocess all the html, axtree, dom, etc.
        # But in EmbodiedGym, we currently do nothing here in Alfworld.
        pass

    @cost_tracker_decorator
    def get_action(self, obs):
        self.obs_history.append(obs)

        main_prompt = AlfworldPrompt(
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(adp.SystemPrompt().prompt)

        human_prompt = adp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name_or_path,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long
            chat_messages = Discussion([system_prompt, human_prompt])
            logger.debug(f"FULL PROMPT:\n {chat_messages}")
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))
        if ans_dict.get("think", None) is not None:
            logger.info(f"Agent thought:\n {ans_dict.get('think', None)}")

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def run(self, info: AgentInfo):
        pass

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr
