from langchain.chat_models import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from chat_models.dynamic_params import ChatDynamicParams


set_debug(True)

# specify parameter constraints for ChatDynamicParams
chat_dynamic_params = ChatDynamicParams(
    model=ChatOpenAI(model="gpt-3.5-turbo"),
    temp_max=1.0,
)

# prompts
prompt_accuracy = "What's 1+1?"
prompt_creativity = "Write a poem about the positive consequences of an AI-powered Utopia"
prompt_mix = "Summarize the ideas of capitalism and compare and contrast it with some new emerging ideologies"

# create chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", f"{prompt_accuracy}"),
])

# construct chain
chain = chat_prompt | chat_dynamic_params | StrOutputParser()

# prompt models
print(chain.invoke({}))