from langchain.chat_models.anthropic import ChatAnthropic
from langchain.chat_models.openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from chat_models.not_diamond import (
    ChatNotDiamond,
    ND_MODEL_CLAUDE_2_1,
    ND_MODEL_GPT_3_5,
    ND_MODEL_GPT_4,
)


set_debug(True)


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

# create chat prompts
# route to GPT-3.5
chat_prompt_1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", 'What does this paragraph mean?: As your perspective of the world increases not only is the pain it inflicts on you less but also its meaning. Understanding the world requires you to take a certain distance from it. Things that are too small to see with the naked eye, such as molecules and atoms, we magnify. Things that are too large, such as cloud formations, river deltas, constellations, we reduce. At length we bring it within the scope of our senses and we stabilize it with fixer. When it has been fixed we call it knowledge. Throughout our childhood and teenage years, we strive to attain the correct distance to objects and phenomena. We read, we learn, we experience, we make adjustments. Then one day we reach the point where all the necessary distances have been set, all the necessary systems have been put in place. That is when time begins to pick up speed. It no longer meets any obstacles, everything is set, time races through our lives, the days pass by in a flash and before we know that is happening we are fort, fifty, sixty... Meaning requires content, content requires time, time requires resistance. Knowledge is distance, knowledge is stasis and the enemy of meaning. My picture of my father on that evening in 1976 is, in other words, twofold: on the one hand I see him as I saw him at that time, through the eyes of an eight-year-old: unpredictable and frightening; on the other hand, I see him as a peer through whose life time is blowing and unremittingly sweeping large chunks of meaning along with it.'),
])

# route to GPT-4
chat_prompt_2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "Create a website that displays a geographical map of all the Michelin star rated restaurants. The website should have filtering functionality to filter restaurants by region/country, cuisine, and year. Return all of the necessary source files including HTML, JavaScript and CSS content. A backend API should be built to serve the data for the restaurants. Build an API in Python using the FastAPI framework. Return the code for the API. Produce all of the Terraform required to deploy both frontend website and backend API to Google Cloud Run. Also include the Dockerfile needed to package both frontend website and backend API."),
])

# construct chain
chain = chat_prompt_1 | chat_not_diamond | StrOutputParser()

# prompt models
print(chain.invoke({}))
