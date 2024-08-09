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

    def test_chat_documents(self) -> None:
        documents = [
        {'title': 'widget sales 2019', 'text': '1 million'},
        {'title': 'widget sales 2020', 'text': '2 million'},
        {'title': 'widget sales 2021', 'text': '4 million'}
        ]
        content = [cohere.v2.TextContent("how many widges were sold in 2020?")]
        for doc in documents:
            content.append(cohere.v2.DocumentContent(document=doc))
        response = co.chat(messages=cohere.v2.UserMessage(content=content))

        print(response.message)

    def test_chat_tools(self) -> None:
        get_weather_tool = {
            "name": "get_weather",
            "desctiption" : "gets the weather of a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type" : "str",
                        "description": "the location to get weather, example: San Fransisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
        tools = [cohere.v2.Tool(type='function', function=get_weather_tool)]
        messages = [cohere.v2.UserMessage(content='what is the weather in Toronto?')]
        res = co.chat(tools=tools, messages=messages)

        # call the get_weather tool
        tool_result = {"temperature": "30C"}
        tool_content = [cohere.v2.ToolContent(output=tool_result)]
        messages.append(res.message)
        messages.append(cohere.v2.ToolMessage(tool_call_id=res.message.tool_calls[0].id, tool_content=tool_content))

        res = co.chat(tools=tools, messages=messages)
        print(res.message)