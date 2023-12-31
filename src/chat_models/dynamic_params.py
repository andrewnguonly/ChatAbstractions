import json
import logging
from typing import Any, Dict, List, Optional

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.pydantic_v1 import Field, root_validator
from langchain.llms.ollama import Ollama
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


OLLAMA_MODEL = "mistral"


class ChatDynamicParams(BaseChatModel):
    """Chat model abstraction that dynamically selects model parameters at
    runtime.

    Supported model parameters:
    - temperature (temp)
    - presence penalty (pp)
    """
    model: BaseChatModel
    temp_min: float = Field(default=0.0, ge=0.0, le=2.0)
    temp_max: float = Field(default=2.0, ge=0.0, le=2.0)
    pp_min: float = Field(default=-2.0, ge=-2.0, le=2.0)
    pp_max: float = Field(default=2.0, ge=-2.0, le=2.0)

    _local_model: Ollama = Ollama(model=OLLAMA_MODEL, temperature=0)

    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        temp_min = values.get("temp_min", 0.0)
        temp_max = values.get("temp_max", 2.0)
        pp_min = values.get("pp_min", -2.0)
        pp_max = values.get("pp_max", 2.0)

        # validate that Ollama server is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = json.loads(response.text)["models"]
        except Exception:
            raise ValueError("Ollama server is not available.")

        # validate that Ollama mistral model is available
        found_model = False
        for model in models:
            name = model["name"].split(":")[0]
            if name == OLLAMA_MODEL:
                found_model = True

        if not found_model:
            raise ValueError(f"Ollama {OLLAMA_MODEL} model not found.")
        
        # validate temperature
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
