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
    max_prompt_tokens=None,
    be_cautious=True,
    extra_instructions=None,
)

AGENT_TEST = GenericAgentArgs(
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=FLAGS_TEST,
    max_retry=3,
)

def get_benchmark_env_args(
    benchmark_name: str, meta_seed=42, max_steps=None, n_repeat=None
) -> list[EnvArgs]:
    """
    Changed from the original function in task_collection.py.
    Returns a list of EnvArgs for the given benchmark_name.

    Args:
        benchmark_name: A string representing the benchmark name.
        meta_seed: The seed for the random number generator.
        max_steps: None or int. The maximum number of steps for each task.
            if None, it will use the default value for the benchmark.
        n_repeat: None or int. The number of seeds for each task.
            if None, it will use the default value for the benchmark.
        is_agent_curriculum: wether to use the agent curriculum or the human curriculum.

    Returns:
        A list of EnvArgs.

    Raises:
        ValueError: If the benchmark_name is not recognized, or if the benchmark_name is not
            followed by a subcategory for workarena.
    """
    env_args_list = []
    rng = np.random.RandomState(meta_seed)

    filters = benchmark_name.split(".")
    benchmark_id = filters[0]
    if filters[0] == "workarena":
        benchmark_id = "workarena." + filters[1]

    max_steps_default = {
        "workarena.l1": 15,
        "workarena.l2": 40,
        "workarena.l3": 40,
        "webarena": 15,
        "weblinx": None,
    }

    n_repeat_default = {
        "workarena.l1": 10,
        "workarena.l2": 1,
        "workarena.l3": 1,
        "webarena": 1,
        "weblinx": 1,
    }

    if max_steps is None:
        max_steps = max_steps_default.get(benchmark_id, None)
    if n_repeat is None:
        n_repeat = n_repeat_default.get(benchmark_id, 1)
    else:
        if benchmark_id == "webarena" and n_repeat != 1:
            logger.warning(
                f"webarena is expected to have only one seed per task. Ignoring n_seeds_default = {n_repeat}"
            )
            n_repeat = 1

    if benchmark_name.startswith("workarena"):
        from browsergym.workarena import ALL_WORKARENA_TASKS, ATOMIC_TASKS, get_all_tasks_agents

        if len(filters) < 2:
            raise ValueError(f"You must specify the sub set of workarena, e.g.: workarena.l2.")

        if benchmark_name == "workarena.l1.sort":
            task_names = [task.get_task_id() for task in ATOMIC_TASKS]
            task_names = [task for task in task_names if "sort" in task]
            env_args_list = _make_env_args(task_names, max_steps, n_repeat, rng)

        else:
            for task, seed in get_all_tasks_agents(
                filter=".".join(filters[1:]),
                meta_seed=meta_seed,
                n_seed_l1=n_repeat,
            ):
                task_name = task.get_task_id()
                env_args_list.append(
                    EnvArgs(task_name=task_name, task_seed=seed, max_steps=max_steps)
                )

    elif benchmark_name == "webarena":
        from browsergym.webarena import ALL_WEBARENA_TASK_IDS

        env_args_list = _make_env_args(ALL_WEBARENA_TASK_IDS, max_steps, n_repeat, rng)
    elif benchmark_name.startswith("miniwob"):
        miniwob_benchmarks_map = {
            "miniwob": MINIWOB_ALL,
            "miniwob_tiny_test": MINIWOB_TINY_TEST,
        }
        env_args_list = _make_env_args(
            miniwob_benchmarks_map[benchmark_name], max_steps, n_repeat, rng
        )
    elif benchmark_name.startswith("weblinx"):
        from weblinx_browsergym import ALL_WEBLINX_TASK_IDS

        env_args_list = _make_env_args(ALL_WEBLINX_TASK_IDS, max_steps, n_repeat, rng)
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    return env_args_list

def main():
    exp_dir = f"/home/weichen/AgentLab_OOD/src/agentlab/agents/testing_agent/{__name__}"
    
    env_args = bgym.EnvArgs(
        task_name="workarena.servicenow.create-incident",
        task_seed=130,
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