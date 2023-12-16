# ChatDynamic
`ChatDynamic` is a subclass of [LangChain's `BaseChatModel`](https://github.com/langchain-ai/langchain/blob/v0.0.350/libs/core/langchain_core/language_models/chat_models.py). The implementation of `ChatDynamic` demonstrates the ability to select a chat model at runtime based on environment variable configuration. In the event of an outage or degraded performance by an LLM provider, this functionality (i.e. failover) may be desirable.

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

# ChatLoadBalance
`ChatLoadBalance` is a subclass of [LangChain's `BaseChatModel`](https://github.com/langchain-ai/langchain/blob/v0.0.350/libs/core/langchain_core/language_models/chat_models.py). The implementation of `ChatLoadBalance` demonstrates the ability to select a method of load balancing (random, round robin, least rate limited) between LLM models. In the event of rate limting or peak usage times, this functionality may be desirable.

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

# ChatChaos
`ChatChaos` is a subclass of [LangChain's `BaseChatModel`](https://github.com/langchain-ai/langchain/blob/v0.0.350/libs/core/langchain_core/language_models/chat_models.py). The implementation of `ChatChaos` demonstrates the ability to substitute normal LLM behavior with chaotic behavior. The purpose of this abstraction is to promote the [Principles of Chaos Engineering](https://principlesofchaos.org/) in the context of LLM applications. This abstraction is inspired by [Netflix's Chaos Monkey](https://github.com/Netflix/chaosmonkey).

```python
```

# Running Tests

Run the following command.

        PYTHONPATH=src/ python3 -m unittest
