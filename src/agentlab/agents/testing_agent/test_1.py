import bgym
from browsergym.experiments import EnvArgs
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

import numpy as np
import logging

logger = logging.getLogger(__name__)


FLAGS_TEST = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=False,
        action_set="bid",
        long_description=False,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=128000, # The context of Qwen2.5-7B-Instruct is 128K tokens
    be_cautious=True,
    extra_instructions=None,
)

AGENT_TEST = GenericAgentArgs(
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4-1106-preview"],
    chat_model_args=CHAT_MODEL_ARGS_DICT["local/Qwen2.5-7B-Instruct"],
    flags=FLAGS_TEST,
    max_retry=3,
)


def main():
    exp_dir = "/home/weichen/AgentLab_OOD/src/agentlab/agents/testing_agent/test_1"

    env_args = bgym.EnvArgs(
        task_name="workarena.servicenow.create-incident",
        task_seed=484,
        max_steps=15,
        headless=False,
    )

    exp_args_list = [
        bgym.ExpArgs(
            agent_args=AGENT_TEST,
            env_args=env_args,
            logging_level=logging.INFO,
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


# 检查是否是主程序
if __name__ == "__main__":
    main()
