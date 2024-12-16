from agentlab.agents.embodied_agent.alfworld_agent_prompt import AlfworldPromptFlags
from agentlab.agents.embodied_agent.alfworld_agent_prompt import AlfworldPrompt
from agentlab.agents.embodied_agent.alfworld_agent import AlfworldAgentArgs
from agentlab.agents.embodied_agent import alfworld_dynamic_prompting as adp
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from embodiedgym.experiments.loop import AlfworldEnvArgs, ExpArgs

# TODO: import error in loop.py
from embodiedgym.experiments.loop import ExpArgs, AlfworldEnvArgs
from embodiedgym.core.env import OOD_ACTION
import logging
import os


logger = logging.getLogger(__name__)

# 获取当前 Python 文件的目录路径
current_directory = os.path.dirname(os.path.abspath(__file__))

# 将工作目录更改为当前文件目录
os.chdir(current_directory)

OOD_PROMPT = f"""
Besides all the actions listed in "Available actions:", there is always one more action you can take, which is `{OOD_ACTION}`.
Use the action `{OOD_ACTION}` only when you observe something highly abnormal in the environment or detect an unexpected change that suggests the environment may no longer be in its normal state. Do not use `{OOD_ACTION}` for general difficulties or uncertainties.

When using `{OOD_ACTION}`, follow this format:
<action>
{OOD_ACTION}: "<Your Analysis>"
</action>

In your analysis:
1. Clearly describe the abnormality or unexpected change.
2. Briefly explain why it appears abnormal and what risks or uncertainties are involved in proceeding.
3. Ask a clarifying question to confirm whether the environment is normal or if adjustments are needed before continuing the task.
"""
FLAG_TEST = AlfworldPromptFlags(
    obs=adp.ObsFlags(),
    actions=adp.ActionFlags(),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    # use_memory=False,
    use_concrete_example=True,
    use_abstract_example=False,  # Current abstract prompt is not designed well and it's also not that useful.
    use_hints=False,
    enable_chat=False,  # currently not supported in Alfworld
    max_prompt_tokens=128000,  # The context of Qwen2.5-7B-Instruct is 128K tokens
    be_cautious=False,
    extra_instructions=OOD_PROMPT,  # give the agent the OOD prompt that it can have the option to detect OOD and report it.
    add_missparsed_messages=True,
)

AGENT_TEST = AlfworldAgentArgs(
    # chat_model_args=CHAT_MODEL_ARGS_DICT["local/Llama-3-8B-Instruct-sft-alfworld"],
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o"],
    flags=FLAG_TEST,
    max_retry=3,
)


def main():
    exp_dir = "./test_embodied_id_results/"
    exp_dir = "./test_embodied_id_results/"

    env_args = AlfworldEnvArgs(
        task_name="json_2.1.1/valid_unseen/pick_and_place_simple-Pencil-None-Shelf-308/trial_T20190908_121952_610012/game.tw-pddl",
        max_step=15,
        wait_for_user_message=False,
        terminate_on_infeasible=True,
    )

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
