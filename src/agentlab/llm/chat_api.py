import logging
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import openai
from huggingface_hub import InferenceClient
from openai import AzureOpenAI, OpenAI

import agentlab.llm.tracking as tracking
from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.huggingface_utils import HFBaseChatModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)


def make_system_message(content: str) -> dict:
    return dict(role="system", content=content)


def make_user_message(content: str) -> dict:
    return dict(role="user", content=content)


def make_assistant_message(content: str) -> dict:
    return dict(role="assistant", content=content)


class CheatMiniWoBLLM(AbstractChatModel):
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    def __call__(self, messages) -> str:
        prompt = messages[-1]["content"]
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        answer = f"""I'm clicking the button as requested.
<action>
{action}
</action>
"""
        return make_assistant_message(answer)


@dataclass
class CheatMiniWoBLLMArgs:
    model_name = "test/cheat_miniwob_click_test"
    max_total_tokens = 10240
    max_input_tokens = 8000
    max_new_tokens = 128

    def make_model(self):
        return CheatMiniWoBLLM()

    def prepare_server(self):
        pass

    def close_server(self):
        pass


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return OpenRouterChatModel(
            model_name=self.model_name_or_path,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


@dataclass
class OpenAIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return OpenAIChatModel(
            model_name=self.model_name_or_path,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""

    deployment_name: str = None

    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name_or_path,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            deployment_name=self.deployment_name,
        )


@dataclass
class SelfHostedModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a self-hosted model."""

    model_url: str = None
    token: str = None
    backend: str = "huggingface"
    n_retry_server: int = 4

    def make_model(self):
        if self.backend == "huggingface":
            # currently only huggingface tgi servers are supported
            if self.model_url is None:
                self.model_url = os.environ["AGENTLAB_MODEL_URL"]
            if self.token is None:
                self.token = os.environ["AGENTLAB_MODEL_TOKEN"]

            return HuggingFaceURLChatModel(
                model_name=self.model_name,
                model_url=self.model_url,
                token=self.token,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                n_retry_server=self.n_retry_server,
            )
        else:
            raise ValueError(f"Backend {self.backend} is not supported")


@dataclass
class ChatModelArgs(BaseModelArgs):
    """Object added for backward compatibility with the old ChatModelArgs."""

    model_path: str = None
    model_url: str = None
    model_size: str = None
    training_total_tokens: int = None
    hf_hosted: bool = False
    is_model_operational: str = False
    sliding_window: bool = False
    n_retry_server: int = 4
    infer_tokens_length: bool = False
    vision_support: bool = False
    shard_support: bool = True
    extra_tgi_args: dict = None
    tgi_image: str = None
    info: dict = None

    def __post_init__(self):
        import warnings

        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "ChatModelArgs is deprecated and used only for xray. Use one of the specific model args classes instead.",
            DeprecationWarning,
        )
        warnings.simplefilter("default", DeprecationWarning)

    def make_model(self):
        pass


@dataclass
class LocalHuggingFaceModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a local HuggingFace model."""

    max_retry: int = 4
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True

    def make_model(self):
        return LocalHuggingFaceChatModel(
            model_name_or_path=self.model_name_or_path,
            max_new_tokens=self.max_new_tokens,
            max_retry=self.max_retry,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
        )


def _extract_wait_time(error_message, min_retry_wait_time=60):
    """Extract the wait time from an OpenAI RateLimitError message."""
    match = re.search(r"try again in (\d+(\.\d+)?)s", error_message)
    if match:
        return max(min_retry_wait_time, float(match.group(1)))
    return min_retry_wait_time


class RetryError(Exception):
    pass


def handle_error(error, itr, min_retry_wait_time, max_retry):
    if not isinstance(error, openai.OpenAIError):
        raise error
    logging.warning(
        f"Failed to get a response from the API: \n{error}\n" f"Retrying... ({itr+1}/{max_retry})"
    )
    wait_time = _extract_wait_time(
        error.args[0],
        min_retry_wait_time=min_retry_wait_time,
    )
    logging.info(f"Waiting for {wait_time} seconds")
    time.sleep(wait_time)
    error_type = error.args[0]
    return error_type


class ChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        api_key_env_var=None,
        client_class=OpenAI,
        client_args=None,
        pricing_func=None,
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time

        # Get the API key from the environment variable if not provided
        if api_key_env_var:
            api_key = api_key or os.getenv(api_key_env_var)
        self.api_key = api_key

        # Get pricing information
        if pricing_func:
            pricings = pricing_func()
            try:
                self.input_cost = float(pricings[model_name]["prompt"])
                self.output_cost = float(pricings[model_name]["completion"])
            except KeyError:
                logging.warning(
                    f"Model {model_name} not found in the pricing information, prices are set to 0. Maybe try upgrading langchain_community."
                )
                self.input_cost = 0.0
                self.output_cost = 0.0
        else:
            self.input_cost = 0.0
            self.output_cost = 0.0

        client_args = client_args or {}
        self.client = client_class(
            api_key=api_key,
            **client_args,
        )

    def __call__(self, messages: list[dict]) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        e = None
        for itr in range(self.max_retry):
            self.retries += 1
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.success = True
                break
            except openai.OpenAIError as e:
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        return make_assistant_message(completion.choices[0].message.content)

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
            # "busted_retry_llm": int(not self.success), # not logged if it occurs anyways
        }


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        client_args=None,  # mainly for base_url
    ):
        if client_args is None:
            client_args = {}
        if client_args.get("base_url") is None:
            client_args["base_url"] = "https://api.shubiaobiao.cn/v1/"
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENAI_API_KEY",
            client_class=OpenAI,
            pricing_func=tracking.get_pricing_openai,
            client_args=client_args,
        )


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
    ):
        client_args = {
            "base_url": "https://openrouter.ai/api/v1",
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENROUTER_API_KEY",
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=tracking.get_pricing_openrouter,
        )


class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        deployment_name=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"

        client_args = {
            "azure_deployment": deployment_name,
            "azure_endpoint": endpoint,
            "api_version": "2024-02-01",
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=AzureOpenAI,
            client_args=client_args,
            pricing_func=tracking.get_pricing_openai,
        )


class HuggingFaceURLChatModel(HFBaseChatModel):
    def __init__(
        self,
        model_name: str,
        model_url: str,
        token: Optional[str] = None,
        temperature: Optional[int] = 1e-1,
        max_new_tokens: Optional[int] = 512,
        n_retry_server: Optional[int] = 4,
    ):
        super().__init__(model_name, n_retry_server)
        if temperature < 1e-3:
            logging.warning("Models might behave weirdly when temperature is too low.")

        if token is None:
            token = os.environ["TGI_TOKEN"]

        client = InferenceClient(model=model_url, token=token)
        self.llm = partial(
            client.text_generation, temperature=temperature, max_new_tokens=max_new_tokens
        )


class LocalChatModel:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 100,
        max_retry: int = 4,
        temperature: float = 0.5,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ):
        """
        Base class for local chat models.

        Args:
            model_name_or_path (str): Path or model ID to load the model from.
            max_new_tokens (int): Maximum number of tokens to generate.
            max_retry (int): Maximum number of retry attempts for generating a response.
            temperature (float): Sampling temperature for generation.
            top_k (int): Number of highest probability vocabulary tokens to keep for top-k sampling.
            top_p (float): Cumulative probability for top-p (nucleus) sampling.
            repetition_penalty (float): Penalty for repetition during generation.
            do_sample (bool): Whether to sample from the model's output distribution.
        """
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.max_retry = max_retry
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

    def __call__(self, messages: List[Dict[str, Any]]) -> Dict[str, str]:
        raise NotImplementedError("Subclasses should implement this method.")


class LocalHuggingFaceChatModel(LocalChatModel):
    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Subclass for handling local Hugging Face models with additional configuration options.

        Args:
            model_name_or_path (str): Path or model ID to load the tokenizer and model from.
            kwargs: Additional generation parameters passed to the parent class.
        """
        super().__init__(model_name_or_path, **kwargs)

        # Load tokenizer and model
        logger.debug("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        logger.debug("Loading model and moving it to the appropriate device...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",  # Automatically distribute the model across available GPUs
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )  # Use half precision to save memory (optional))

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """Generates a response using the model based on the input messages."""
        self.retries = 0
        self.error_types = []

        for itr in range(self.max_retry):
            self.retries += 1
            try:
                # Use apply_chat_template if available
                if hasattr(self.tokenizer, "apply_chat_template"):
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    logger.debug("apply_chat_template not available, using manual formatting.")
                    prompt = self._format_messages(messages)

                logger.debug("Encoding input prompt...")
                inputs = self.tokenizer(prompt, return_tensors="pt")

                logger.debug("Generating response...")
                output_ids = self.llm.generate(
                    input_ids=inputs.input_ids,
                    # attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=self.do_sample,
                )

                generated_ids = [
                    output_id[len(input_id) :]
                    for input_id, output_id in zip(inputs.input_ids, output_ids)
                ]

                logger.debug("Decoding generated output...")
                generated_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                return make_assistant_message(generated_text)
            except Exception as e:
                logger.debug(f"Error on attempt {itr + 1}: {str(e)}")
                self.error_types.append(self._handle_error(e, itr))
                if itr == self.max_retry - 1:
                    raise e

        logger.debug("An error occurred after multiple retries.")
        return make_assistant_message("An error occurred after multiple retries.")

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Formats a list of input messages into a single string prompt."""
        formatted = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role and content:
                formatted += f"{role}: {content}\n"
            else:
                raise ValueError("Each message must have a 'role' and 'content'.")
        return formatted

    def _extract_response(self, prompt: str, generated_text: str) -> str:
        """Extracts the model's response from the generated text by removing the prompt part."""
        return generated_text[len(prompt) :].strip()

    def _handle_error(self, e: Exception, itr: int) -> str:
        """Handles errors and logs information about the failure."""
        return type(e).__name__

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
            # "busted_retry_llm": int(not self.success), # not logged if it occurs anyways
        }
