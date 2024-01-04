import json
import logging
from typing import Any, Dict, List, Optional

import requests
import tiktoken
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.pydantic_v1 import Field, root_validator
from langchain.llms.ollama import Ollama
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


OLLAMA_MODEL = "mistral"

# { "class_name": (<min>, <max>) }
TEMP_RANGES = {
    ChatOpenAI.__name__: (0.0, 2.0),
    ChatAnthropic.__name__: (0.0, 1.0),
}

MAX_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "claude-2.0": 100000,
    "claude-2.1": 200000,
}

class ChatDynamicParams(BaseChatModel):
    """Chat model abstraction that dynamically selects model parameters at
    runtime.

    Supported model parameters:
    - temperature (temp)
    - presence penalty (pp)
    - max tokens (tkn)
    """
    model: BaseChatModel
    temp_min: float = 0.0
    temp_max: float = 1.0
    pp_min: float = Field(default=-2.0, ge=-2.0, le=2.0)
    pp_max: float = Field(default=2.0, ge=-2.0, le=2.0)
    tkn_min: int = 256
    tkn_max: int = 1024

    _local_model: Ollama = Ollama(model=OLLAMA_MODEL, temperature=0)

    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        model = values.get("model")
        temp_min = values.get("temp_min", 0.0)
        temp_max = values.get("temp_max", 2.0)
        pp_min = values.get("pp_min", -2.0)
        pp_max = values.get("pp_max", 2.0)

        # validate that Ollama server is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            tags = json.loads(response.text)["models"]
        except Exception:
            raise ValueError("Ollama server is not available.")

        # validate that Ollama mistral model is available
        found_model = False
        for tag in tags:
            model_name = tag["name"].split(":")[0]
            if model_name == OLLAMA_MODEL:
                found_model = True

        if not found_model:
            raise ValueError(f"Ollama {OLLAMA_MODEL} model not found.")
        
        # validate temperature
        temp_range = TEMP_RANGES.get(type(model).__name__)
        if (temp_range and
            (temp_min < temp_range[0] or temp_max > temp_range[1])
        ):
            raise ValueError(
                f"temp_min must be greater than or equal to {temp_range[0]} "
                f"and temp_max must be less than or equal to {temp_range[1]}."
            )
        if temp_min > temp_max:
            raise ValueError("temp_min must be less than temp_max.")
        
        # validate presence penalty
        if pp_min > pp_max:
            raise ValueError("pp_min must be less than pp_max.")

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "dynamic-params-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Reset parameters of model based on messages.
        
        Newly set parameters are retrieved from the model's _default_params
        property and passed to the underlying client.
        """
        prompt = self._get_prompt(messages)

        # check if model supports temperature
        if hasattr(self.model, "temperature"):
            new_temp = self._get_temperature(prompt)
            logger.info(
                "Changing model temperature from "
                f"{getattr(self.model, 'temperature')} to {new_temp}"
            )
            setattr(self.model, "temperature", new_temp)

        # check if model supports presence penalty
        if isinstance(self.model, ChatOpenAI):
            new_fp = self._get_presence_penalty(prompt)
            logger.info(
                "Changing model presence_penalty from "
                f"{self.model.model_kwargs.get('presence_penalty', 0.0)} "
                f"to {new_fp}"
            )
            self.model.model_kwargs["presence_penalty"] = new_fp

        # check if model support max_tokens
        if hasattr(self.model, "max_tokens"):
            # OpenAI
            new_tkn = self._get_max_tokens(messages)
            logger.info(
                "Changing model max_tokens from "
                f"{getattr(self.model, 'max_tokens')} to {new_tkn}"
            )
            setattr(self.model, "max_tokens", new_tkn)

        elif hasattr(self.model, "max_tokens_to_sample"):
            # Anthropic
            new_tkn = self._get_max_tokens(messages)
            logger.info(
                "Changing model max_tokens_to_sample from "
                f"{getattr(self.model, 'max_tokens_to_sample')} to {new_tkn}"
            )
            setattr(self.model, "max_tokens_to_sample", new_tkn)

        return self.model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
    
    def _get_prompt(self, messages: List[BaseMessage]) -> str:
        """Get prompt from list of messages.
        
        The current prompt is typically the last human message.
        """
        for message in reversed(messages):
            if message.type == "human":
                return message.content
            
        # return the last message if no human message is found
        return messages[-1].content
    
    def _get_msgs_as_str(self, messages: List[BaseMessage]) -> str:
        """Convert list of messages to string."""
        return "".join([message.content for message in messages])
    
    def _get_temperature(self, prompt: str) -> float:
        """Return optimal temperature based on prompt."""
        local_model_prompt = (
            "Classify the following LLM prompt by determining if it requires "
            "a factually correct response, if it requires a creative response, "
            "or if it requires a mix of both factual accuracy and creativity. "
            "Return only one of the following responses without any formatting: "
            "`accuracy`, `creativity`, or `mix`\n"
            "\n"
            f"Prompt: `{prompt}`"
        )
        response = self._local_model(local_model_prompt)

        # Retrieve the first token from response. This is typically the
        # classification value.
        first_token = response.split()[0].lower()

        # convert classification to temperature
        if "accuracy" in first_token:
            return self.temp_min
        elif "creativity" in first_token:
            return self.temp_max
        elif "mix" in first_token:
            return (self.temp_min + self.temp_max) / 2
        else:
            # default to original model temperature
            return getattr(self.model, "temperature")

    def _get_presence_penalty(self, prompt: str) -> float:
        """Return optimal presence penalty based on prompt."""
        local_model_prompt = (
            "Classify the following LLM prompt by determining if it requires "
            "a focused response, if it requires an open-ended response, "
            "or if it requires a mix of both focus and open-endedness. "
            "Return only one of the following responses without any formatting: "
            "`focus`, `open-ended`, or `mix`\n"
            "\n"
            f"Prompt: `{prompt}`"
        )
        response = self._local_model(local_model_prompt)

        # Retrieve the first token from response. This is typically the
        # classification value.
        first_token = response.split()[0].lower()

        # convert classification to presence penalty
        if "focus" in first_token:
            return self.pp_min
        elif "open-ended" in first_token:
            return self.pp_max
        elif "mix" in first_token:
            return (self.pp_min + self.pp_max) / 2
        else:
            # default to original model presence penalty
            if isinstance(self.model, ChatOpenAI):
                return self.model.model_kwargs.get("presence_penalty", 0.0)
            else:
                return 0.0

    def _get_max_tokens(self, messages: List[BaseMessage]) -> int:
        """Return max_tokens value based on size of prompt and max token limit
        for model.
        """
        msgs_as_str = self._get_msgs_as_str(messages)

        if isinstance(self.model, ChatOpenAI):
            max_token_limit = MAX_TOKEN_LIMITS.get(self.model.model_name)

            # get token count of messages
            encoding = tiktoken.encoding_for_model(self.model.model_name)
            msgs_tkn_cnt = len(encoding.encode(msgs_as_str))

        elif isinstance(self.model, ChatAnthropic):
            max_token_limit = MAX_TOKEN_LIMITS.get(self.model.model)

            # get token count of messages
            msgs_tkn_cnt = self.model.get_num_tokens_from_messages(messages)
        else:
            # unsupported model, return any value
            return 256
        
        # request_max_tokens is the optimal number of tokens to request
        requested_max_tokens = max_token_limit - msgs_tkn_cnt

        # Return request_max_tokens if it's in the specified range (between
        # tkn_min and tkn_max). Return tkn_max, if request_max_tokens is
        # greater than tkn_max. Return tkn_min, if request_max_tokens is
        # less than tkn_min.
        #
        # Warning: If tkn_min is returned, the total token count for the
        # request may still exceed the model's max token limit.
        return min(self.tkn_max, max(self.tkn_min, requested_max_tokens))
    