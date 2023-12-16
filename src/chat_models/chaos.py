import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from croniter import croniter
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import AIMessage, ChatGeneration, ChatResult, HumanMessage
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


# chaotic behaviors
B_MALFORMED_JSON = "malformed_json"
B_HALUCINATION = "halucination"
B_LATENCY = "latency"

# malformed JSON behaviors
J_BACKTICKS = "backticks"
J_TRUNCATED = "truncated"
J_SINGLE_QUOTES = "single_quotes"
J_BEHAVIORS = [J_BACKTICKS, J_TRUNCATED, J_SINGLE_QUOTES]


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

    # behavior configurations
    halucination_prompt: str = "Write a poem about the Python programming language."
    latency_min_sec: int = 30
    latency_max_sec: int = 60

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
        current_time = datetime.utcnow()
        if (self.enabled and
            self._in_cron_range(current_time) and
            self._in_ratio_range()
        ):
            behavior = self._select_behavior()
            logger.info(
                "Chaotic behavior is enabled for this inference instance. "
                f"Current time: {current_time}. Behavior: {behavior}."
            )

            # some behaviors must be configured prior to inference
            if behavior == B_HALUCINATION:
                # manually add halucination prompt to messages
                logger.info(
                    f"Behavior {B_HALUCINATION}: Adding halucination prompt "
                    f"'{self.halucination_prompt}' to messages."
                )
                messages = self._halucination_messages(messages)

            if behavior == B_LATENCY:
                # manually add random delay
                random_delay = random.uniform(
                    self.latency_min_sec,
                    self.latency_max_sec,
                )
                logger.info(
                    f"Behavior {B_LATENCY}: Delaying inference by "
                    f"{random_delay} seconds."
                )
                time.sleep(random_delay)

            # inference / call LLM
            chat_result = self.model._generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

            # some behaviors must be configured after inference
            if behavior == B_MALFORMED_JSON:
                # manually augment response to be malformed JSON
                chat_result = self._malform_generations(chat_result)
                logger.info(
                    f"Behavior {B_MALFORMED_JSON}: Manually augmented "
                    "ChatGenerations in ChatResult. Malformed JSON: "
                    f"{chat_result.generations[0].text}"
                )

        else:
            # normal inference
            chat_result = self.model._generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        return chat_result
    
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
    
    def _halucination_messages(
        self,
        messages: List[BaseMessage],
    ) -> List[BaseMessage]:
        """Manually add halucination prompt to list of messages."""
        last_message = messages[-1]
        new_last_message_text = (
            f"{last_message.content}\n\n{self.halucination_prompt}"
        )
        return messages[:-1] + [HumanMessage(content=new_last_message_text)]

    def _malform_generations(self, chat_result: ChatResult) -> ChatResult:
        """Manually malform ChatGenerations in ChatResult."""
        malformed_generations = [
            self._malform_generation(generation)
            for generation in chat_result.generations
        ]
        chat_result.generations = malformed_generations
        return chat_result

    def _malform_generation(
        self,
        chat_generation: ChatGeneration,
    ) -> ChatGeneration:
        """Manually malform ChatGeneration."""
        try:
            json.loads(chat_generation.text)
        except json.JSONDecodeError:
            # completion is already invalid JSON, so just return it
            return chat_generation

        # manually malform text
        j_behavior = random.choice(J_BEHAVIORS)
        if j_behavior == J_BACKTICKS:
            message=AIMessage(content=f"```{chat_generation.text}```")
        if j_behavior == J_TRUNCATED:
            message=AIMessage(content=f"{chat_generation.text[:-1]}")
        if j_behavior == J_SINGLE_QUOTES:
            message=AIMessage(
                content=f"""{chat_generation.text.replace('"', "'")}""",
            )

        return ChatGeneration(message=message)
