import unittest

from langchain.chat_models.anthropic import ChatAnthropic
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from chat_models.not_diamond import (
    ChatNotDiamond,
    ND_MODEL_CLAUDE_2_1,
    ND_MODEL_GPT_3_5,
    ND_MODEL_GPT_4,
)


class TestChatNotDiamond(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

        self.model = ChatNotDiamond(
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

    def test_get_messages(self):
        # set up test
        base_messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Who won the world series in 2020?"),
        ]

        expected_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "human",
                "content": "Who won the world series in 2020?",
            },
        ]

        # call method / assertions
        self.assertEqual(
            self.model._get_messages(base_messages), expected_messages
        )

    def test_get_chat_model(self):
        # call method / assertions
        chat_model_1 = self.model._get_chat_model("gpt-3.5", 4096)
        chat_model_2 = self.model._get_chat_model("gpt-3.5", 4097)
        chat_model_3 = self.model._get_chat_model("gpt-4", 8192)
        chat_model_4 = self.model._get_chat_model("gpt-4", 8193)
        chat_model_5 = self.model._get_chat_model("claude-2.1", 200000)
        chat_model_6 = self.model._get_chat_model("not_supported", 4096)

        self.assertIsInstance(chat_model_2, ChatOpenAI)
        self.assertIsInstance(chat_model_3, ChatOpenAI)
        self.assertIsInstance(chat_model_4, ChatOpenAI)
        self.assertIsInstance(chat_model_5, ChatAnthropic)
        self.assertIsInstance(chat_model_6, ChatOpenAI)

        if isinstance(chat_model_1, ChatOpenAI):
            self.assertEqual(chat_model_1.model_name, "gpt-3.5-turbo")
        else:
            raise AssertionError("chat_model_1 is not an instance of ChatOpenAI")
        
        if isinstance(chat_model_2, ChatOpenAI):
            self.assertEqual(chat_model_2.model_name, "gpt-3.5-turbo-16k")
        else:
            raise AssertionError("chat_model_2 is not an instance of ChatOpenAI")
        
        if isinstance(chat_model_3, ChatOpenAI):
            self.assertEqual(chat_model_3.model_name, "gpt-4")
        else:
            raise AssertionError("chat_model_3 is not an instance of ChatOpenAI")
        
        if isinstance(chat_model_4, ChatOpenAI):
            self.assertEqual(chat_model_4.model_name, "gpt-4-32k")
        else:
            raise AssertionError("chat_model_4 is not an instance of ChatOpenAI")
        
        if isinstance(chat_model_5, ChatAnthropic):
            self.assertEqual(chat_model_5.model, "claude-2.1")
        else:
            raise AssertionError("chat_model_5 is not an instance of ChatAnthropic")
        
        if isinstance(chat_model_6, ChatOpenAI):
            self.assertEqual(chat_model_6.model_name, "gpt-3.5-turbo")
        else:
            raise AssertionError("chat_model_6 is not an instance of ChatOpenAI")
