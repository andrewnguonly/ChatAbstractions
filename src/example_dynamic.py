from langchain.chat_models.openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from chat_models.dynamic import ChatDynamic


set_debug(True)

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

# create chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python programmer"),
    ("human", "I'm learning how to program in Python"),
    ("ai", "Sure! I can help you write a Python app"),
    ("human", "Write a 'hello world' app in Python"),
])

# construct chain
chain = chat_prompt | chat_dynamic_model | StrOutputParser()

# prompt models
print(chain.invoke({}))
