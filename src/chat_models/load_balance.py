import logging
import random
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


class ChatLoadBalance(BaseChatModel):
    """Chat model abstraction that load balances model selection at runtime."""
    models: List[BaseChatModel]
    
    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        models = values.get("models", [])

        if not models or len(models) == 0:
            raise ValueError(
                "The 'models' attribute must have a size greater than 0."
            )

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "load-balance-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Load balance chat model selection via random selection."""
        current_model_idx = random.randint(0, len(self.models)-1)
        current_model = self.models[current_model_idx]
        logger.info(f"Selected chat model '{current_model}'")

        return current_model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
