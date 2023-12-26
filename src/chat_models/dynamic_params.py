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
    """
    model: BaseChatModel

    _local_model: Ollama = Ollama(temperature=0)

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
            self.model.temperature = self._get_temperature(prompt)

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
        return 0
