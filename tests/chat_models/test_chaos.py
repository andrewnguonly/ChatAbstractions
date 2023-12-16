import datetime
import json
import unittest

from croniter import croniter
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import AIMessage, ChatGeneration

from chat_models.chaos import ChatChaos


class TestChatChaos(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

        self.model = ChatChaos(
            model=ChatOpenAI(model="gpt-3.5-turbo"),
            enabled=True,
            cron=croniter("0 13 * * *"),
            duration_mins=60,
            enable_malformed_json=True,
        )

    def test_in_cron_range(self):
        # set up test
        day_of_month = datetime.date.today().day
        # in range
        current_time_1 = datetime.datetime(2023, 12, day_of_month, 13, 30, 0)
        # before
        current_time_2 = datetime.datetime(2023, 12, day_of_month, 12, 59, 59)
        # after
        current_time_3 = datetime.datetime(2023, 12, day_of_month, 14, 0, 0)

        # call method / assertions
        self.assertTrue(self.model._in_cron_range(current_time_1))
        self.assertFalse(self.model._in_cron_range(current_time_2))
        self.assertFalse(self.model._in_cron_range(current_time_3))

    def test_malform_generation(self):
        # set up test
        chat_gen_1 = ChatGeneration(
            message=AIMessage(
                content='{"invalid": "json"',
            ),
        )
        new_chat_gen_1 = ChatGeneration(
            message=AIMessage(
                content='{"invalid": "json"',
            )
        )

        chat_gen_2 = ChatGeneration(
            message=AIMessage(
                content='{"valid": "json"}',
            ),
        )

        # call method / assertions
        self.assertEqual(self.model._malform_generation(chat_gen_1), new_chat_gen_1)
        
        new_chat_gen_2 = self.model._malform_generation(chat_gen_2)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(new_chat_gen_2.text)
