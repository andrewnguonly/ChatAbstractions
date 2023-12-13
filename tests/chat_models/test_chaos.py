import unittest
from datetime import datetime
from unittest.mock import patch

from croniter import croniter
from langchain.chat_models.openai import ChatOpenAI

from chat_models.chaos import ChatChaos


class TestChatChaos(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_in_cron_range(self):
        # set up test
        model = ChatChaos(
            model=ChatOpenAI(model="gpt-3.5-turbo"),
            enabled=True,
            cron=croniter("0 13 * * *"),
            duration_mins=60,
            enable_malformed_json=True,
        )
        # in range
        current_time_1 = datetime(2023, 12, 13, 13, 30, 0)
        # before
        current_time_2 = datetime(2023, 12, 13, 12, 59, 59)
        # after
        current_time_3 = datetime(2023, 12, 13, 14, 0, 0)

        # call method / assertions
        self.assertTrue(model._in_cron_range(current_time_1))
        self.assertFalse(model._in_cron_range(current_time_2))
        self.assertFalse(model._in_cron_range(current_time_3))
