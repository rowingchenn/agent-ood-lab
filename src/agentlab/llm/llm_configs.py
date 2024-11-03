from agentlab.llm.chat_api import (
    AzureModelArgs,
    OpenAIModelArgs,
    OpenRouterModelArgs,
    SelfHostedModelArgs,
    LocalHuggingFaceModelArgs,
)

default_oss_llms_args = {
    "n_retry_server": 4,
    "temperature": 0.01,
}

CLOSED_SOURCE_APIS = [
    "openai",
    "reka",
    "test",
]

CHAT_MODEL_ARGS_DICT = {
    "openai/gpt-4o-mini": OpenAIModelArgs(
        model_name_or_path="gpt-4o-mini",
        max_total_tokens=128_000,
        max_input_tokens=80_000,  # input limit of 4o-mini is 128,000 tokens. Prompt shrink based on max_input_tokens.
        max_new_tokens=4000,  # output limit of 4o-mini is 16,384 tokens
    ),
    "openai/gpt-4o-mini-2024-07-18": OpenAIModelArgs(
        model_name_or_path="gpt-4o-mini-2024-07-18",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4000,
    ),
    "openai/gpt-4-1106-preview": OpenAIModelArgs(
        model_name_or_path="gpt-4-1106-preview",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,
    ),
    "openai/gpt-4-vision-preview": OpenAIModelArgs(
        model_name_or_path="gpt-4-vision-preview",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-4o": OpenAIModelArgs(
        model_name_or_path="gpt-4o",
        max_total_tokens=128_000,
        max_input_tokens=60_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-4o-2024-05-13": OpenAIModelArgs(
        model_name_or_path="gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-3.5-turbo-0125": OpenAIModelArgs(
        model_name_or_path="gpt-3.5-turbo-0125",
        max_total_tokens=16_384,
        max_input_tokens=15_000,
        max_new_tokens=1_000,
    ),
    "openai/gpt-3.5-turbo-1106": OpenAIModelArgs(
        model_name_or_path="gpt-3.5-turbo-1106",
        max_total_tokens=16_384,
        max_input_tokens=15_000,
        max_new_tokens=1_000,
    ),
    "azure/gpt-35-turbo/gpt-35-turbo": AzureModelArgs(
        model_name_or_path="gpt-35-turbo",
        deployment_name="gpt-35-turbo",
        max_total_tokens=8_192,
        max_input_tokens=7500,
        max_new_tokens=500,
    ),
    "azure/gpt-4o-2024-05-13": AzureModelArgs(
        model_name_or_path="gpt-4o",
        deployment_name="gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4_000,
    ),
    "azure/gpt-4o-2024-08-06": AzureModelArgs(
        model_name_or_path="gpt-4o",
        deployment_name="gpt-4o-2024-08-06",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4_000,
    ),
    # ---------------- OSS LLMs ----------------#
    "meta-llama/Meta-Llama-3-70B-Instruct": SelfHostedModelArgs(
        model_name_or_path="meta-llama/Meta-Llama-3-70B-Instruct",
        max_total_tokens=8_192,
        max_input_tokens=8_192 - 512,
        max_new_tokens=512,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": SelfHostedModelArgs(
        model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        max_total_tokens=16_384,
        max_input_tokens=16_384 - 512,
        max_new_tokens=512,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": SelfHostedModelArgs(
        model_name_or_path="mistralai/Mixtral-8x22B-Instruct-v0.1",
        max_total_tokens=32_000,
        max_input_tokens=30_000,
        max_new_tokens=2_000,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    # ---------------- OPENROUTER ----------------#
    "openrouter/meta-llama/llama-3.1-405b-instruct": OpenRouterModelArgs(
        model_name_or_path="meta-llama/llama-3.1-405b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-70b-instruct": OpenRouterModelArgs(
        model_name_or_path="meta-llama/llama-3.1-70b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3-70b-instruct": OpenRouterModelArgs(
        model_name_or_path="meta-llama/llama-3-70b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-8b-instruct:free": OpenRouterModelArgs(
        model_name_or_path="meta-llama/llama-3.1-8b-instruct:free",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-8b-instruct": OpenRouterModelArgs(
        model_name_or_path="meta-llama/llama-3.1-8b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=40_000,
        max_new_tokens=4000,
        temperature=1e-1,
    ),
    "openrouter/anthropic/claude-3.5-sonnet:beta": OpenRouterModelArgs(
        model_name_or_path="anthropic/claude-3.5-sonnet:beta",
        max_total_tokens=200_000,
        max_input_tokens=40_000,
        max_new_tokens=4000,
        temperature=1e-1,
    ),
    "openrouter/qwen/qwen-2-72b-instruct": OpenRouterModelArgs(
        model_name_or_path="qwen/qwen-2-72b-instruct",
        max_total_tokens=32_000,
        max_input_tokens=30_000,
        max_new_tokens=2_000,
        temperature=1e-1,
    ),
    # ---------------- LOCAL SOURCE ----------------#
    "local/Qwen2.5-0.5B-Instruct": LocalHuggingFaceModelArgs(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=2048,
        max_retry=4,
        temperature=1e-1,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
    ),
    "local/Qwen2.5-7B-Instruct": LocalHuggingFaceModelArgs(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=2048,
        max_retry=4,
        temperature=1e-1,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
    ),
    "local/Qwen2.5-32B-Instruct": LocalHuggingFaceModelArgs(
        model_name_or_path="Qwen/Qwen2.5-32B-Instruct",
        max_new_tokens=2048,
        max_retry=4,
        temperature=1e-1,
        top_k=15,
        top_p=0.5,
        repetition_penalty=1.2,
        do_sample=True,
    ),
}
