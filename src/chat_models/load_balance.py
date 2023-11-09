import logging
import random
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


LOAD_BALANCER_TYPES = [0, 1, 2]


class ChatLoadBalance(BaseChatModel):
    """Chat model abstraction that load balances model selection at runtime.
    
    Load balancer types:
    0 - random
    1 - round robin
    2 - least rate limited
    """
    models: List[BaseChatModel]
    load_balance_type: int

    # round robin state
    last_used_model: int = 0

    # least rate limited state
    rate_limit_state: Dict[str, Any] = {}
    
    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        models = values.get("models", [])
        load_balance_type = values.get("load_balance_type", 0)

        if not models or len(models) == 0:
            raise ValueError(
                "The 'models' attribute must have a size greater than 0."
            )
        
        if load_balance_type not in LOAD_BALANCER_TYPES:
            raise ValueError(
                "The 'load_balance_type' attribute must be in "
                f"{LOAD_BALANCER_TYPES}"
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
        # default to first model
        current_model_idx = 0

        # select model based on load balancer type
        if self.load_balance_type == 0:
            # random
            current_model_idx = random.randint(0, len(self.models)-1)

        elif self.load_balance_type == 1:
            # round robin
            if len(self.models) > 1:
                current_model_idx = (self.last_used_model + 1) % len(self.models)
                self.last_used_model = current_model_idx

        elif self.load_balance_type == 2:
            # TODO: least rate limited
            # See https://github.com/langchain-ai/langchain/issues/9601
            raise NotImplementedError(
                "Least rate limited load balancer is not implemented."
            )

        current_model = self.models[current_model_idx]
        logger.info(f"Selected chat model '{current_model}'")

        return current_model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
