from langchain.chat_models import ChatOpenAI

from chat_models.dynamic_params import ChatDynamicParams


gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

chat_dynamic_temp = ChatDynamicParams(
    model=gpt_3_5_model,
)
