import os
import typing
import unittest

import cohere
from cohere import ToolMessage, UserMessage, AssistantMessage

co = cohere.ClientV2(timeout=10000)

package_dir = os.path.dirname(os.path.abspath(__file__))
embed_job = os.path.join(package_dir, "embed_job.jsonl")


class TestClientV2(unittest.TestCase):

    def test_chat(self) -> None:
        response = co.chat(
            model="command-r-plus", messages=[cohere.UserChatMessageV2(content="hello world!")])

        print(response.message)

    def test_chat_stream(self) -> None:
        stream = co.chat_stream(
            model="command-r-plus", messages=[cohere.UserChatMessageV2(content="hello world!")])

        events = set()

        for chat_event in stream:
            if chat_event is not None:
                events.add(chat_event.type)
                if chat_event.type == "content-delta":
                    print(chat_event.delta)

        self.assertTrue("message-start" in events)
        self.assertTrue("content-start" in events)
        self.assertTrue("content-delta" in events)
        self.assertTrue("content-end" in events)
        self.assertTrue("message-end" in events)

    @unittest.skip("Skip v2 test for now")
    def test_chat_documents(self) -> None:
        from cohere import Document
        documents = [
            Document(data={"title": "widget sales 2019", "text": "1 million"}),
            Document(data={"title": "widget sales 2020", "text": "2 million"}),
            Document(data={"title": "widget sales 2021", "text": "4 million"}),
        ]
        response = co.chat(
            messages=[cohere.UserChatMessageV2(
                content=[cohere.TextContent(text="how many widges were sold in 2020?")],
            )],
            model="command-r-plus",
            documents=documents,
        )

        print(response.message)

    @unittest.skip("Skip v2 test for now")
    def test_chat_tools(self) -> None:
        from typing import Sequence
        get_weather_tool = cohere.ToolV2Function(
            name="get_weather",
            description="gets the weather of a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "str",
                        "description": "the location to get weather, example: San Fransisco, CA",
                    }
                },
                "required": ["location"],
            },
        )
        tools = [cohere.ToolV2(type="function", function=get_weather_tool)]
        messages: cohere.ChatMessages = [
            cohere.UserChatMessageV2(content="what is the weather in Toronto?")
        ]
        res = co.chat(model="command-r-plus", tools=tools, messages=messages)

        # call the get_weather tool
        tool_result = {"temperature": "30C"}
        tool_content: Sequence[cohere.TextToolContent] = [cohere.TextToolContent(text="The weather in Toronto is 30C")]
        # Use the first text content from the response if available, else fallback to str
        assistant_content = res.message.content[0].text if (hasattr(res.message, 'content') and isinstance(res.message.content, list) and len(res.message.content) > 0 and hasattr(res.message.content[0], 'text')) else str(res.message)
        messages.append(cohere.AssistantChatMessageV2(content=[cohere.TextAssistantMessageV2ContentItem(text=assistant_content)]))
        if res.message.tool_calls is not None and res.message.tool_calls[0].id is not None:
            messages.append(cohere.ToolChatMessageV2(
                tool_call_id=res.message.tool_calls[0].id, content=list(tool_content)))

        res = co.chat(tools=tools, messages=messages, model="command-r-plus")
        print(res.message)
