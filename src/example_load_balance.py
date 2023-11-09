from langchain.chat_models.openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from chat_models.load_balance import ChatLoadBalance


set_debug(True)

# initialize chat models
gpt_4_model = ChatOpenAI(model="gpt-4")
gpt_3_5_model = ChatOpenAI(model="gpt-3.5-turbo")

# specify all models that can be selected in the ChatDynamic instance
chat_load_balance_model = ChatLoadBalance(models=[gpt_4_model, gpt_3_5_model])

# create chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python programmer"),
    ("human", "I'm learning how to program in Python"),
    ("ai", "Sure! I can help you write a Python app"),
    ("human", "Write a 'hello world' app in Python. Attempt {index}"),
])

# construct chain
chain = chat_prompt | chat_load_balance_model | StrOutputParser()

# prompt models
for index in range(0, 6):
    print(chain.invoke({"index": index}))
