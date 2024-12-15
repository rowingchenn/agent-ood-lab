from agentlab.agents.embodied_agent.alfworld_agent_prompt import AlfworldPromptFlags
from agentlab.agents.embodied_agent.alfworld_agent_prompt import AlfworldPrompt
from agentlab.agents.embodied_agent.alfworld_agent import AlfworldAgentArgs
from agentlab.agents.embodied_agent import alfworld_dynamic_prompting as adp
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

# from embodiedgym.experiments.loop import EnvArgs, ExpArgs
import logging

# TODO: import error in loop.py          
from embodiedgym.experiments.loop import ExpArgs, AlfworldEnvArgs

logger = logging.getLogger(__name__)

FLAG_TEST = AlfworldPromptFlags(
    obs=adp.ObsFlags(),
    actions=adp.ActionFlags(),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    # use_memory=False,
    use_concrete_example=True,
    # use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=128000,  # The context of Qwen2.5-7B-Instruct is 128K tokens
    be_cautious=True,
    extra_instructions=None,
)

AGENT_TEST = AlfworldAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
    flags=FLAG_TEST,
    max_retry=2
)


def main():
    exp_dir = "./test_embodied_id_results/"

    env_args = AlfworldEnvArgs(
        # task_name="webarena.692",
        # task_name="workarena.servicenow.infeasible-navigate-and-order-apple-mac-book-pro15-l2",  # L2 is multi-tab
        task_name="workarena.servicenow.workload-balancing-small-l2",
        task_seed=89,
        max_steps=15,
        headless=False,
    )

    if env_args.task_name == "openended":
        AGENT_TEST.flag.enable_chat = True
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": "https://www.google.com"}

    exp_args_list = [
        ExpArgs(
            agent_args=AGENT_TEST,
            env_args=env_args,
            logging_level=logging.DEBUG,
        ),
    ]

    for exp_args in exp_args_list:
        exp_args.agent_args.prepare()
        exp_args.prepare(exp_root=exp_dir)
        logging.info(f"Ready to run {exp_args}.")
        exp_args.run()
        logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
        exp_args.agent_args.close()
        logging.info("Experiment finished.")
        # TODO: add ood result information to ExpResult
        # loading and printing results
        # exp_result = get_exp_result(exp_args.exp_dir)
        # exp_record = exp_result.get_exp_record()
        # for key, val in exp_record.items():
        #     print(f"{key}: {val}")


if __name__ == "__main__":
    main()
