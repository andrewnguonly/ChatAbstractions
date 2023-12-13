import logging
from typing import Any, Dict, List, Optional

from croniter import croniter, CroniterBadCronError
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


class ChatChaos(BaseChatModel):
    """Chat model abstraction that substitutes normal LLM behavior with
    chaotic behavior.
    """
    model: BaseChatModel
    enabled: bool

    # configure how often chaotic behavior should occur and for how long
    cron_schedule: str
    duration_mins: int = 1
    ratio: float = 0.01  # percentage of inferences that should be chaotic

    # configure types of chaotic behavior
    enable_malformed_json: bool = False
    enable_halucination: bool = False
    enable_latency: bool = False

    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        cron_schedule = values.get("cron_schedule", "")
        duration_mins = values.get("duration_mins", 1)
        ratio = values.get("ratio", 0.01)

        enable_malformed_json = values.get("enable_malformed_json", False)
        enable_halucination = values.get("enable_halucination", False)
        enable_latency = values.get("enable_latency", False)

        try:
            croniter(cron_schedule)
        except CroniterBadCronError:
            raise ValueError(
                "The 'cron' attribute must be a valid cron schedule."
            )
        
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

        return self.model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
