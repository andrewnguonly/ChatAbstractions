# ChatAbstractions

This repo is a collection of chat model abstractions that demonstrates how to wrap (subclass) [LangChain's `BaseChatModel`](https://github.com/langchain-ai/langchain/blob/v0.0.350/libs/core/langchain_core/language_models/chat_models.py) in order to add functionality to a chain without breaking existing chat model interfaces. The use cases for wrapping chat models in this manner are mostly focused on dynamic model selection. However, other use cases are possible as well.

Subclassing `BaseChatModel` requires implementing 2 methods: `_llm_type()` and `_generate()`.
```python
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage


class ChatSubclass(BaseChatModel):

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        raise NotImplementedError

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Add custom logic here."""
        raise NotImplementedError
```

## ChatDynamic
The implementation of `ChatDynamic` demonstrates the ability to select a chat model at runtime based on environment variable configuration. In the event of an outage or degraded performance by an LLM provider, this functionality (i.e. failover) may be desirable.

```python
# set environment variable DYNAMIC_CHAT_MODEL_ID=gpt-4

# initialize chat models
gpt_4_model = ChatOpenAI(model="gpt-4")
gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

# specify all models that can be selected in the ChatDynamic instance
chat_dynamic_model = ChatDynamic(
    models={
        "gpt-4": gpt_4_model,
        "gpt-3_5": gpt_3_5_model,
    },
    default_model="gpt-4",
)
```

## ChatLoadBalance
The implementation of `ChatLoadBalance` demonstrates the ability to select a method of load balancing (random, round robin, least rate limited) between LLM models. In the event of rate limiting or peak usage times, this functionality may be desirable.

```python
# initialize chat models
gpt_4_model = ChatOpenAI(model="gpt-4")
gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

# specify all models that can be selected in the ChatLoadBalance instance
chat_load_balance_model = ChatLoadBalance(
    models=[gpt_4_model, gpt_3_5_model],
    load_balance_type=1,  # 0 - random, 1 - round robin, 2 - least rate limited
)
```

## ChatChaos
The implementation of `ChatChaos` demonstrates the ability to substitute normal LLM behavior with chaotic behavior. The purpose of this abstraction is to promote the [Principles of Chaos Engineering](https://principlesofchaos.org/) in the context of LLM applications. This abstraction is inspired by [Netflix's Chaos Monkey](https://github.com/Netflix/chaosmonkey).

```python
# initialize chat model
gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

# configure ChatChaos
chat_chaos_model = ChatChaos(
    model=gpt_3_5_model,
    enabled=True,
    cron=croniter("0 * * * *"),
    duration_mins=60,
    ratio=1.0,
    enable_malformed_json=False,
    enable_hallucination=True,
    enable_latency=False,
    hallucination_prompt="Write a poem about the Python programming language.",
)
```

## ChatNotDiamond
The implementation of `ChatNotDiamond` demonstrates the ability leverage [Not Diamond's](https://www.notdiamond.ai/) optimized LLM routing functionality.

```python
# configure ChatNotDiamond
chat_not_diamond = ChatNotDiamond(
    fallback_model=ND_MODEL_GPT_3_5,
    model_map={
        ND_MODEL_GPT_3_5: {
            4096: ChatOpenAI(model="gpt-3.5-turbo"),
            16385: ChatOpenAI(model="gpt-3.5-turbo-16k"),
        },
        ND_MODEL_GPT_4: {
            8192: ChatOpenAI(model="gpt-4"),
            32768: ChatOpenAI(model="gpt-4-32k"),
            128000: ChatOpenAI(model="gpt-4-1106-preview"),
        },
        ND_MODEL_CLAUDE_2_1: {
            200000: ChatAnthropic(model="claude-2.1"),
        },
    }
)
```

## ChatCustomRouter
The implementation of `ChatCustomRouter` demonstrates the ability to implement custom routing logic. For example, a routing function may count tokens, evaluate the current prompt against historical metrics, or call an external routing service.

```python
# define custom routing function
def random_selection(messages: List[BaseMessage], **kwargs: Any) -> str:
    """Randomly select a model from the available models."""
    return random.choice(["gpt-4", "gpt-3_5"])

# initialize chat models
gpt_4_model = ChatOpenAI(model="gpt-4")
gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

# specify all models that can be selected in the ChatCustomRouter instance
chat_custom_router_model = ChatCustomRouter(
    models={
        "gpt-4": gpt_4_model,
        "gpt-3_5": gpt_3_5_model,
    },
    default_model="gpt-4",
    routing_func=random_selection,
)
```

## ChatDynamicParams

The implementation of `ChatDynamicParams` demonstrates the ability to dynamically set model parameters (e.g. `temperature`, `presence_penalty`, `max_tokens`) based on the prompt. Different prompts require different values for model parameters. Tuning and optimizing model parameters may result in more desirable responses. The implementation leverages a local LLM powered by [Ollama](https://ollama.ai/).

```python
# specify parameter constraints for ChatDynamicParams
chat_dynamic_params = ChatDynamicParams(
    model=ChatOpenAI(model="gpt-3.5-turbo"),
    temp_max=1.0,
    pp_min=0.5,
    tkn_max=1000,
)
```

## Running Examples

Run the following command.

    python3 src/example_chaos.py

## Running Tests

Run the following command.

    PYTHONPATH=src/ python3 -m unittest
