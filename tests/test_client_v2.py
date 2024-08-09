import json
import os
import unittest

import cohere

co = cohere.ClientV2(timeout=10000)

package_dir = os.path.dirname(os.path.abspath(__file__))
embed_job = os.path.join(package_dir, 'embed_job.jsonl')


class TestClientV2(unittest.TestCase):

    def test_chat(self) -> None:
        response = co.chat(
            model="command-r-plus",
            messages=[
                cohere.v2.ChatMessage2_User(
                    content="hello world!"
                )
            ]
        )

        print(response.message)

    def test_chat_stream(self) -> None:
        stream = co.chat_stream(
            model="command-r-plus",
            messages=[
                cohere.v2.ChatMessage2_User(
                    content="hello world!"
                )
            ]
        )

        events = set()

        for chat_event in stream:
            if chat_event is not None:
                events.add(chat_event.type)
                if chat_event.type == "content-delta":
                    print(chat_event.delta.message)

        self.assertTrue("message-start" in events)
        self.assertTrue("content-start" in events)
        self.assertTrue("content-delta" in events)
        self.assertTrue("content-end" in events)
        self.assertTrue("message-end" in events)
