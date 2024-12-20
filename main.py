"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import bgym

# from agent_ood_gym.browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from src.agentlab.agents import dynamic_prompting as dp
from src.agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from src.agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from src.agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
import os
import pandas as pd
import json
import random
from pathlib import Path
import logging

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    AGENT_TEST_LOCAL,
    AGENT_TEST_API,
    AGENT_8B,
)
from src.agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
# agent_args = [AGENT_TEST_LOCAL]
# agent_args = [AGENT_4o]
agent_args = [AGENT_8B]

# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
# benchmark = "workarena_l2_agent_curriculum_eval"
benchmark = "webarena"

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 3  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores
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
        action_set="bid",  # change to benchmark specific action set!
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
    max_prompt_tokens=128000,  # The context of Qwen2.5-7B-Instruct is 128K tokens
    be_cautious=True,
    extra_instructions=None,
)

# agent_args = AGENT_TEST = [GenericAgentArgs(
#     # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
#     # chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/anthropic/claude-3.5-sonnet:beta"],
#     # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-3.5-turbo-0125"],
#     # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4-1106-preview"],
#     # chat_model_args=CHAT_MODEL_ARGS_DICT["local/Qwen2.5-7B-Instruct"],
#     flags=FLAGS_TEST,
#     max_retry=3,
# )]

if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
