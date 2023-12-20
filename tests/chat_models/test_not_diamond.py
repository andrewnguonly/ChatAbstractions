import unittest

from langchain.schema.messages import HumanMessage, SystemMessage

from chat_models.not_diamond import ChatNotDiamond


class TestChatNotDiamond(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

        self.model = ChatNotDiamond(
            models=["gpt-3.5"],
            fallback_model="gpt-3.5",
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
