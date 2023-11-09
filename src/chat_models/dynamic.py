import logging
import os
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


class ChatDynamic(BaseChatModel):
    """Chat model abstraction that dynamically selects model at runtime."""
    models: Dict[str, BaseChatModel]
    default_model: str
    
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
        
        current_model_id = os.environ.get("DYNAMIC_CHAT_MODEL_ID")
        if current_model_id is None:
            logger.warning(
                f"WARNING! Environment variable DYNAMIC_CHAT_MODEL_ID is not "
                "set. Model ID '{default_model}' will be used."
            )
        elif current_model_id not in models:
            raise ValueError(
                f"DYNAMIC_CHAT_MODEL_ID '{current_model_id}' must exist in "
                "the 'models' dict."
            )

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "dynamic-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Select chat model from environment variable configuration."""
        current_model_id = os.environ.get(
            "DYNAMIC_CHAT_MODEL_ID",
            self.default_model,
        )
        current_model = self.models[current_model_id]

        return current_model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
