import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


logger = logging.getLogger(__name__)


ND_MODEL_GPT_3_5 = "gpt-3.5"
ND_MODEL_GPT_4 = "gpt-4"
ND_MODEL_CLAUDE_2_1 = "claude-2.1"
ND_MODELS = [
    ND_MODEL_GPT_3_5,
    ND_MODEL_GPT_4,
    ND_MODEL_CLAUDE_2_1,
]


class ChatNotDiamond(BaseChatModel):
    """Chat model abstraction that routes API calls based on Not Diamond's
    model selector.
    
    https://notdiamond.readme.io/reference/modelselector
    """
    models: List[str]
    fallback_model: str

    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        fallback_model = values.get("fallback_model")

        not_diamond_api_key = os.environ.get("NOT_DIAMOND_API_KEY")
        if not_diamond_api_key is None:
            raise ValueError(
                "NOT_DIAMOND_API_KEY environment variable must be set."
            )

        if fallback_model not in ND_MODELS:
            raise ValueError(
                f"The 'fallback_model' attribute must be in {ND_MODELS}."
            )

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "not-diamond-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call Not Diamond API to select model and route request to
        selected model.
        """
        messages = self._get_messages(messages)
        model, estimated_tokens = self._select_model(messages)

    def _get_messages(
        self,
        base_messages: List[BaseMessage],
    ) -> List[Dict[str, str]]:
        """Convert list of BaseMessages to list of dicts to pass to
        Not Diamond API.
        
        messages = [
            {
                "role": "assistant",
                "content": "How can I help you today?"
            },
            {
                "role": "user",
                "content": "Can you write a function that counts from 1 to 10?"
            },
        ]
        """
        return [
            {"role": message.type, "content": message.content}
            for message in base_messages
        ]

    def _select_model(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """Call Not Diamond API to select model.
        
        Return tuple of model name and estimated tokens in prompt.
        """
        url = "https://not-diamond-backend.onrender.com/modelSelector/"
        payload = json.dumps(
            {"messages": messages, "fallback_model": self.fallback_model}
        )
        headers = {
            "Authorization": f"Bearer {os.environ.get('NOT_DIAMOND_API_KEY')}",
            "Content-Type": "application/json",
        }

        response = requests.request(
            "POST", url, headers=headers, data=payload
        ).json()

        return (response["model"], response["estimated_tokens"])
