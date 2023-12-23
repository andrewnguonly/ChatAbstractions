import logging
from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


class ChatCustomRouter(BaseChatModel):
    """Chat model abstraction that dynamically selects model at runtime
    based on custom routing logic.
    """
    models: Dict[str, BaseChatModel]
    default_model: str
    routing_func: Callable[[List[BaseMessage], Any], str]
    
    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        models = values.get("models", {})
        default_model = values.get("default_model", None)

        if not models or len(models) == 0:
            raise ValueError(
                "The 'models' attribute must have a size greater than 0."
            )

        if default_model not in models:
            raise ValueError(
                f"The 'default_model' attribute '{default_model}' must exist "
                "in the 'models' dict."
            )

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "custom-router-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Select chat model based on custom routing logic."""
        current_model_id = self.routing_func(messages=messages, **kwargs)
        logger.info(
            f"Selected model id from routing function: {current_model_id}"
        )
        current_model = self.models.get(current_model_id, self.default_model)

        return current_model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
