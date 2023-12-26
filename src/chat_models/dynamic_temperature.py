import logging
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


class ChatDynamicTemperature(BaseChatModel):
    """Chat model abstraction that dynamically selects model temperature at
    runtime.
    """
    model: BaseChatModel

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "dynamic-temperature-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Reset temperature parameter of model based on messages."""

        if hasattr(self.model, "temperature"):
            self.model.temperature = 0.7

        return self.model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
