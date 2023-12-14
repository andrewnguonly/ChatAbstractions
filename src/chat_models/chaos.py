import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from croniter import croniter
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


# chaotic behaviors
B_MALFORMED_JSON = "malformed_json"
B_HALUCINATION = "halucination"
B_LATENCY = "latency"


class ChatChaos(BaseChatModel):
    """Chat model abstraction that substitutes normal LLM behavior with
    chaotic behavior.
    """
    model: BaseChatModel
    enabled: bool

    # configure how often chaotic behavior should occur and for how long
    cron: croniter
    duration_mins: int = 1
    ratio: float = 0.01  # percentage of inferences that should be chaotic

    # configure types of chaotic behavior
    enable_malformed_json: bool = False
    enable_halucination: bool = False
    enable_latency: bool = False

    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        duration_mins = values.get("duration_mins", 1)
        ratio = values.get("ratio", 0.01)

        enable_malformed_json = values.get("enable_malformed_json", False)
        enable_halucination = values.get("enable_halucination", False)
        enable_latency = values.get("enable_latency", False)
        
        if duration_mins < 1:
            raise ValueError(
                "The 'duration_mins' attribute must be greater than 0."
            )

        if not (0.01 <= ratio <= 1.0):
            raise ValueError(
                "The 'ratio' attribute must be a float between 0.01 and 1.0."
            )
        
        chaos_type_enabled = any([
            enable_malformed_json,
            enable_halucination,
            enable_latency,
        ])
        if not chaos_type_enabled:
            raise ValueError(
                "At least one type of chaos must be enabled: "
                "['enable_malformed_json', 'enable_halucination', "
                "'enable_latency']."
            )

        return values
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return f"chaos-{self.model._llm_type}"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chaotic behavior during inference."""
        current_time = datetime.now()
        if self._in_cron_range(current_time) and self._in_ratio_range():
            behavior = self._select_behavior()
            logger.info(
                "Chaotic behavior is enabled for this inference instance. "
                f"Current time: {current_time}. Behavior: {behavior}."
            )
            if behavior == B_MALFORMED_JSON:
                pass
            elif behavior == B_HALUCINATION:
                pass
            elif behavior == B_LATENCY:
                pass

        return self.model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
    
    def _in_cron_range(self, current_time: datetime) -> bool:
        """Return True if current time is within cron start time and end time
        (cron start + duration).
        """
        # Get the previous scheduled time
        start_time = self.cron.get_prev(datetime)

        # Check if the current time is within range
        end_time = start_time + timedelta(minutes=self.duration_mins)
        return start_time <= current_time <= end_time

    def _in_ratio_range(self) -> bool:
        """Return true is randomly selected number is below ratio value."""
        return random.random() <= self.ratio

    def _select_behavior(self) -> str:
        """Randomly select behavior."""
        enabled_behaviors = []
        if self.enable_malformed_json:
            enabled_behaviors.append(B_MALFORMED_JSON)
        if self.enable_halucination:
            enabled_behaviors.append(B_HALUCINATION)
        if self.enable_latency:
            enabled_behaviors.append(B_LATENCY)

        return random.choice(enabled_behaviors)
