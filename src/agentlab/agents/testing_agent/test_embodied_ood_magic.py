from agentlab.agents.embodied_agent.alfworld_agent_prompt import AlfworldPromptFlags
from agentlab.agents.embodied_agent.alfworld_agent_prompt import AlfworldPrompt
from agentlab.agents.embodied_agent.alfworld_agent import AlfworldAgentArgs
from agentlab.agents.embodied_agent import alfworld_dynamic_prompting as adp
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from embodiedgym.experiments.loop import AlfworldEnvArgs, ExpArgs
from embodiedgym.core.env import OOD_ACTION
import logging
import os
import json

# 导入 tqdm 以显示进度条
from tqdm import tqdm

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
{OOD_ACTION}: '<Your Analysis>'
</action>

In your analysis:
1. Clearly describe the abnormality or unexpected change.
2. Briefly explain why it appears abnormal and what risks or uncertainties are involved in proceeding.
3. Ask a clarifying question to confirm whether the environment is normal or if adjustments are needed before continuing the task.
"""

FLAG_TEST = AlfworldPromptFlags(
    obs=adp.ObsFlags(
        use_error_logs=False,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
    ),
    actions=adp.ActionFlags(
        is_strict=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=True,
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
    # chat_model_args=CHAT_MODEL_ARGS_DICT["local/Meta-Llama-3-8B-Instruct"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
    chat_model_args=CHAT_MODEL_ARGS_DICT["claude-3-5-sonnet-20241022"],
    flags=FLAG_TEST,
    max_retry=3,
)


def main():
    # JSON文件路径
    ood_file_dir = "/home/weichenzhang/LLMAgentOODGym/agent_ood_gym/embodiedgym/core/src/embodiedgym/core/configs/ood_semantic_first_step.json"

    # 打开并读取 JSON 文件
    with open(ood_file_dir, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 获取文件名（不包括路径和扩展名）用于生成结果文件夹名称
    json_filename = os.path.splitext(os.path.basename(ood_file_dir))[0]

    # 创建总体保存的大文件夹名称
    exp_dir = f"./claude/embodied_ood_results_{json_filename}_magic/"
    os.makedirs(exp_dir, exist_ok=True)

    # 使用 tqdm 包装数据以显示进度条
    for sample in tqdm(data, desc="处理进度", unit="任务"):
        task_id = sample["task_id"]
        task_name = sample["task_name"]

        if task_id <38:
            continue

        # 创建每个样本的单独文件夹
        sample_dir = os.path.join(exp_dir, f"task_{task_id}")
        os.makedirs(sample_dir, exist_ok=True)

        logger.info(f"开始处理任务ID: {task_id}, 任务名称: {task_name}")

        env_args = AlfworldEnvArgs(
            task_name=task_name,  # 动态设置task_name
            max_step=35,
            wait_for_user_message=False,
            terminate_on_infeasible=True,
        )

        exp_args = ExpArgs(
            agent_args=AGENT_TEST,
            env_args=env_args,
            logging_level=logging.INFO,
            ood_args=sample,  # 使用当前样本的OOD参数
        )

        # 设置结果保存路径
        exp_args.prepare(exp_root=sample_dir)

        logging.info(f"准备运行任务ID: {task_id}")
        exp_args.agent_args.prepare()

        try:
            exp_args.run()
            logging.info(f"任务ID: {task_id} 运行完成。")
        except Exception as e:
            logging.error(f"任务ID: {task_id} 运行时出错: {e}")
        finally:
            exp_args.agent_args.close()
            logging.info(f"任务ID: {task_id} 实验结束。")
            # TODO: 添加 OOD 结果信息到 ExpResult
            # 例如，可以加载和打印结果，保存到文件等
            # exp_result = get_exp_result(exp_args.exp_dir)
            # exp_record = exp_result.get_exp_record()
            # for key, val in exp_record.items():
            #     print(f"{key}: {val}")


if __name__ == "__main__":
    main()
