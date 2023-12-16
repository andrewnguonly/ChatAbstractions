from croniter import croniter
from langchain.chat_models.openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from chat_models.chaos import ChatChaos


set_debug(True)


# initialize chat model
gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

# configure ChatChaos
chat_chaos_model = ChatChaos(
    model=gpt_3_5_model,
    enabled=True,
    cron=croniter("0 * * * *"),
    duration_mins=60,
    ratio=1.0,
    enable_malformed_json=True,
    enable_halucination=False,
    enable_latency=False,
)

# create chat prompts
chat_prompt_1 = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python programmer"),
    ("human", "I'm learning how to program in Python"),
    ("ai", "Sure! I can help you write a Python app"),
    ("human", "Write a 'hello world' app in Python"),
])

chat_prompt_2 = ChatPromptTemplate.from_messages([
    ("system", "You are a JSON object generator"),
    ("human", "Return a JSON object with 2 keys: name and age"),
])

# construct chain
chain = chat_prompt_2 | chat_chaos_model | StrOutputParser()

# prompt models
print(chain.invoke({}))
