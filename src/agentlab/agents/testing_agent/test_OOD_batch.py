import bgym
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
import os
import pandas as pd
import json
import random
from pathlib import Path
'''
reddit 
similars: social forums
advertisement: vpn, streaming, grammarly, vpn, shopping

shopping
similars: shopping sites
advertisement: coupon, vpn, credit cards, payment others

shopping admin
similars: order management, customer support
advertisement: shipping, payment, ads
'''



# 获取当前 Python 文件的目录路径
current_directory = os.path.dirname(os.path.abspath(__file__))

# 将工作目录更改为当前文件目录
os.chdir(current_directory)

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Define flags
FLAGS_TEST = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_action_history=True,
        use_diff=False,
        html_type="pruned_html",
        extract_visible_tag=True,
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=False,
        action_set="bid",
        individual_examples=True,
    ),
    use_plan=False,
    use_thinking=True,
    max_prompt_tokens=128000,
)

def main():
    # Paths
    current_path = Path(__file__).resolve()
    csv_path = current_path.parents[5] / "agent_ood_gym/browsergym/experiments/src/browsergym/experiments/benchmark/metadata/webarena.csv"
    json_path = current_path.parents[5] / "agent_ood_gym/browsergym/oodarena/src/browsergym/oodarena/task_data/test.json"

    # Load CSV and JSON
    webarena_data = pd.read_csv(csv_path)
    webarena_data = webarena_data[webarena_data["sites"].isin(["shopping_admin", "reddit", "shopping"])]
    print(f"len(webarena_data): {len(webarena_data)}")

    shopping_data = webarena_data[webarena_data["sites"] == "shopping"]
    shopping_admin_data = webarena_data[webarena_data["sites"] == "shopping_admin"]
    reddit_data = webarena_data[webarena_data["sites"] == "reddit"]


    with open(json_path, 'r') as file:
        ood_data = json.load(file)

    
    ood_shopping_data = [task for task in ood_data if task["web_arena_type"] == "shopping"]
    ood_shopping_admin_data = [task for task in ood_data if task["web_arena_type"] == "shopping_admin"]
    ood_reddit_data = ood_reddit_data = [task for task in ood_data if task["web_arena_type"] == "reddit"]

    ood_overall = {"shopping": ood_shopping_data, "shopping_admin": ood_shopping_admin_data, "reddit": ood_reddit_data}
    web_overall = {"shopping": shopping_data, "shopping_admin": shopping_admin_data, "reddit": reddit_data}
    # Iterate through all agent types
    # for agent_name, agent_config in CHAT_MODEL_ARGS_DICT.items():


    # agent_name = "openai/gpt-4o"
    # agent_name = "openai/gpt-3.5-turbo-0125"
    # agent_config = CHAT_MODEL_ARGS_DICT[agent_name]
    # agent_config=CHAT_MODEL_ARGS_DICT["openai/gpt-3.5-turbo-0125"],
    # agent_args = GenericAgentArgs(chat_model_args=agent_config, flags=FLAGS_TEST, max_retry=3)

    output_rows = []
    for agent_name in ["openai/gpt-4o"]:
        agent_config = CHAT_MODEL_ARGS_DICT[agent_name]
        agent_args = GenericAgentArgs(chat_model_args=agent_config, flags=FLAGS_TEST, max_retry=3)
        for ood_insert_step in [1]:
            for sites, each_data in web_overall.items():
                filtered_ood_tasks = ood_overall[sites]
                print(f"Sites: {each_data['sites'].iloc[0]} | Length: {len(each_data)}")
                print(f"len(filtered_ood_tasks): {len(filtered_ood_tasks)}")
                # print(filtered_ood_tasks)
            # Iterate through all tasks in webarena.csv
                for index, (_, task_row) in enumerate(each_data.iterrows()):
                    task_name = task_row["task_name"]
                    sites = task_row["sites"]

                    if not filtered_ood_tasks:
                        logging.warning(f"No OOD tasks found for site: {sites}")
                        continue
                    

                    ood_place_index = index % len(filtered_ood_tasks)
                    ood_task = filtered_ood_tasks[ood_place_index]
                    
                    ood_task_type = ood_task["ood_task_type"]
                    ood_task_id = ood_task["ood_task_id"]
                    ood_url = ood_task["start_url"]
                    ood_goal = ood_task["goal"]
                    # print(f"Index: {index}, OOD Task Index: {ood_place_index}, ood_task_id: {ood_task_id}")

                    
                    
                    # Append task details to output
                    output_rows.append({
                        "agent_name": agent_name,
                        "ood_insert_step": ood_insert_step,
                        **task_row.to_dict(),
                        "ood_task_type": ood_task_type,
                        "ood_task_id": ood_task_id,
                        "start_url": ood_url,
                        "goal": ood_goal
                    })

                    # print(f"ood_task_type {ood_task["ood_task_type"]}")
                    # print(f"ood_task_id {ood_task["ood_task_id"]}")
                    ood_args = {
                        "ood_task_type": ood_task_type,
                        "ood_task_id": ood_task_id,
                        "ood_insert_step": ood_insert_step,
                        "ood_max_steps": 5,
                    }

                    # Define environment arguments
                    env_args = EnvArgs(
                        task_name=task_name,
                        task_seed=89,
                        max_steps=15,
                        headless=False,
                        timeout=30000,
                    )

                    # Set up experiment arguments
                    exp_args = ExpArgs(
                        agent_args=agent_args,
                        env_args=env_args,
                        logging_level=logging.DEBUG,
                        ood_args=ood_args,
                    )

                    # Prepare and run the experiment
                    # exp_dir = "./test_OOD/"
                    # benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l2_agent_curriculum_eval"]()
                    # exp_args.agent_args.set_benchmark(benchmark, demo_mode=True) # Override Some flags based on the benchmark.
                    # exp_args.agent_args.prepare()
                    # exp_args.prepare(exp_root=exp_dir)
                    # logging.info(f"Ready to run {exp_args}.")
                    # exp_args.run()
                    # logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
                    # exp_args.agent_args.close()
                    # logging.info("Experiment finished.")
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv("./output_df.csv", index=False)

if __name__ == "__main__":
    main()
