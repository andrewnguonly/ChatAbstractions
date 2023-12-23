import random
from typing import Any, List


from langchain.chat_models.openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.messages import BaseMessage
from langchain.schema.output_parser import StrOutputParser

from chat_models.custom_router import ChatCustomRouter


set_debug(True)


# define custom routing functions
def random_selection(messages: List[BaseMessage], **kwargs: Any) -> str:
    """Randomly select a model from the available models."""
    return random.choice(["gpt-4", "gpt-3_5"])

def evaluate_messages(messages: List[BaseMessage], **kwargs: Any) -> str:
    """Select a model based on the messages.
    
    This routing function demostrates the ability to access messages in the
    prompt.
    """
    return "gpt-3_5"

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

# create chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Python programmer"),
    ("human", "I'm learning how to program in Python"),
    ("ai", "Sure! I can help you write a Python app"),
    ("human", "Write a 'hello world' app in Python"),
])

# construct chain
chain = chat_prompt | chat_custom_router_model | StrOutputParser()

# prompt models
print(chain.invoke({}))
