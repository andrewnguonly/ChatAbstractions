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
    fallback_model: str

    # A client should specify their own model map. The default model map
    # reflects the available routing support in Not Diamond. In practice,
    # a client should specify token limit values smaller than the actual
    # maximum allowed value to give buffer for completion tokens.
    #
    # {
    #     "gpt-3.5": {
    #         4096: ChatOpenAI(model="gpt-3.5-turbo"),
    #         16385: ChatOpenAI(model="gpt-3.5-turbo-16k"),
    #     },
    #     "gpt-4": {
    #         8192: ChatOpenAI(model="gpt-4"),
    #         32768: ChatOpenAI(model="gpt-4-32k"),
    #         128000: ChatOpenAI(model="gpt-4-1106-preview"),
    #     },
    #     "claude-2.1": {
    #         200000: ChatAnthropic(model="claude-2.1"),
    #     },
    # }
    model_map: Dict[str, Dict[int, BaseChatModel]]

    @root_validator(pre=True)
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class attributes."""
        fallback_model = values.get("fallback_model")
        model_map = dict(values.get("model_map"))

        not_diamond_api_key = os.environ.get("NOT_DIAMOND_API_KEY")
        if not_diamond_api_key is None:
            raise ValueError(
                "NOT_DIAMOND_API_KEY environment variable must be set."
            )

        if fallback_model not in ND_MODELS:
            raise ValueError(
                f"The 'fallback_model' attribute must be in {ND_MODELS}."
            )
        
        if not set(model_map.keys()).issubset(ND_MODELS):
            raise ValueError(
                f"The 'model_map' keys must be in {ND_MODELS}."
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
        formatted_msgs = self._get_messages(messages)
        selected_model, estimated_tokens = self._select_model(formatted_msgs)
        logger.info(
            f"Selected model: {selected_model}, "
            f"Estimated tokens: {estimated_tokens}",
        )

        chat_model = self._get_chat_model(selected_model, estimated_tokens)
        return chat_model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

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

        return (response["model"], response["token_estimate"])
    
    def _get_chat_model(
        self,
        selected_model: str,
        estimated_tokens: int,
    ) -> BaseChatModel:
        """Get BaseChatModel based on selected model from Not Diamond."""
        default_token_mapping = self.model_map.get(self.fallback_model)
        token_mapping = self.model_map.get(
            selected_model,
            default_token_mapping,
        )

        sorted_token_counts = sorted(token_mapping.keys())
        for token_limit in sorted_token_counts:
            if estimated_tokens <= token_limit:
                return token_mapping[token_limit]
