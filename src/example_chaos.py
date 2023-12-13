from langchain.chat_models.openai import ChatOpenAI
from langchain.globals import set_debug

from chat_models.chaos import ChatChaos


set_debug(True)


# initialize chat model
gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

# specify all models that can be selected in the ChatDynamic instance
chat_chaos_model = ChatChaos(
    model=gpt_3_5_model,
    enabled=True,
    cron_schedule="0 13 * * 1",
    duration_mins=5,
    ratio=0.01,
    enable_malformed_json=True,
    enable_halucination=True,
    enable_latency=True,
)
