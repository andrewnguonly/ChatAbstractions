import logging
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.ollama import Ollama
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


class ChatDynamicParams(BaseChatModel):
    """Chat model abstraction that dynamically selects model parameters at
    runtime.

    Supported model parameters:
    - temperature
    """
    model: BaseChatModel
    temp_min: float = 0.0
    temp_max: float = 2.0

    # TODO: validate that Ollama server is running
    _local_model: Ollama = Ollama(model="mistral", temperature=0)

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
        """Reset parameters of model based on messages."""
        prompt = self._get_prompt(messages)

        if hasattr(self.model, "temperature"):
            new_temp = self._get_temperature(prompt)
            logger.info(f"Changing model temperature from {self.model.temperature} to {new_temp}")
            self.model.temperature = new_temp

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
    
    def _get_temperature(self, prompt: str) -> int:
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
            # return minimum temperature to be conservative
            return self.temp_min
