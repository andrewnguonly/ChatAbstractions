# DynamicChatModel
`DynamicChatModel` is a subclass of [LangChain's `BaseChatModel`](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chat_models/base.py). The implementation of `DynamicChatModel` demonstrates the ability to select a chat model at run time based on environment variable configuration. In the event of an outage or degraded performance by an LLM provider, this functionality may be desirable.

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
