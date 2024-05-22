import os
import unittest

import cohere
from cohere import ChatMessage, ChatConnector, ClassifyExample, CreateConnectorServiceAuth, Tool, \
    ToolParameterDefinitionsValue, ToolResult, Message_Chatbot, Message_User

co = cohere.Client(timeout=10000)

package_dir = os.path.dirname(os.path.abspath(__file__))
embed_job = os.path.join(package_dir, 'embed_job.jsonl')


class TestClient(unittest.TestCase):

    def test_token_falls_back_on_env_variable(self) -> None:
        cohere.Client(api_key=None)
        cohere.Client(None)

    def test_context_manager(self) -> None:
        with cohere.Client(api_key="xxx") as client:
            self.assertIsNotNone(client)

    def test_chat(self) -> None:
        chat = co.chat(
            chat_history=[
                Message_User(
                    message="Who discovered gravity?"),
                Message_Chatbot(message="The man who is widely credited with discovering "
                                "gravity is Sir Isaac Newton")
            ],
            message="What year was he born?",
            connectors=[ChatConnector(id="web-search")]
        )

        print(chat)

    def test_chat_stream(self) -> None:
        stream = co.chat_stream(
            chat_history=[
                Message_User(
                    message="Who discovered gravity?"),
                Message_Chatbot(message="The man who is widely credited with discovering "
                                "gravity is Sir Isaac Newton")
            ],
            message="What year was he born?",
            connectors=[ChatConnector(id="web-search")]
        )

        for chat_event in stream:
            if chat_event.event_type == "text-generation":
                print(chat_event.text)

    def test_stream_equals_true(self) -> None:
        with self.assertRaises(ValueError):
            co.chat(
                stream=True,  # type: ignore
                message="What year was he born?",
            )

    def test_deprecated_fn(self) -> None:
        with self.assertRaises(ValueError):
            co.check_api_key("dummy", dummy="dummy")  # type: ignore

    def test_moved_fn(self) -> None:
        with self.assertRaises(ValueError):
            co.list_connectors("dummy", dummy="dummy")  # type: ignore

    @unittest.skipIf(os.getenv("CO_API_URL") is not None, "Doesn't work in staging.")
    def test_generate(self) -> None:
        response = co.generate(
            prompt='Please explain to me how LLMs work',
        )
        print(response)

    def test_embed(self) -> None:
        response = co.embed(
            texts=['hello', 'goodbye'],
            model='embed-english-v3.0',
            input_type="classification",
            embedding_types=["float", "int8", "uint8", "binary", "ubinary"]
        )

        if response.response_type == "embeddings_by_type":
            self.assertIsNotNone(response.embeddings.float)  # type: ignore
            self.assertIsNotNone(response.embeddings.float_)

            if response.embeddings.float_ is not None:
                self.assertEqual(type(response.embeddings.float_[0][0]), float)
            if response.embeddings.int8 is not None:
                self.assertEqual(type(response.embeddings.int8[0][0]), int)
            if response.embeddings.uint8 is not None:
                self.assertEqual(type(response.embeddings.uint8[0][0]), int)
            if response.embeddings.binary is not None:
                self.assertEqual(type(response.embeddings.binary[0][0]), int)
            if response.embeddings.ubinary is not None:
                self.assertEqual(type(response.embeddings.ubinary[0][0]), int)

        print(response)

    def test_embed_batch_types(self) -> None:
        # batch more than 96 texts
        response = co.embed(
            texts=['hello']*100,
            model='embed-english-v3.0',
            input_type="classification",
            embedding_types=["float", "int8", "uint8", "binary", "ubinary"]
        )

        if response.response_type == "embeddings_by_type":
            self.assertEqual(len(response.texts or []), 100)
            self.assertEqual(len(response.embeddings.float_ or []), 100)
            self.assertEqual(len(response.embeddings.int8 or []), 100)
            self.assertEqual(len(response.embeddings.uint8 or []), 100)
            self.assertEqual(len(response.embeddings.binary or []), 100)
            self.assertEqual(len(response.embeddings.ubinary or []), 100)
        else:
            self.fail("Expected embeddings_by_type response type")

        print(response)

    def test_embed_batch_v1(self) -> None:
        # batch more than 96 texts
        response = co.embed(
            texts=['hello']*100,
            model='embed-english-v3.0',
            input_type="classification",
        )

        if response.response_type == "embeddings_floats":
            self.assertEqual(len(response.embeddings), 100)
        else:
            self.fail("Expected embeddings_floats response type")

        print(response)

    @unittest.skipIf(os.getenv("CO_API_URL") is not None, "Doesn't work in staging.")
    def test_embed_job_crud(self) -> None:
        dataset = co.datasets.create(
            name="test",
            type="embed-input",
            data=open(embed_job, 'rb'),
        )

        result = co.wait(dataset)

        self.assertEqual(result.dataset.validation_status, "validated")

        # start an embed job
        job = co.embed_jobs.create(
            dataset_id=dataset.id or "",
            input_type="search_document",
            model='embed-english-v3.0')

        print(job)

        # list embed jobs
        my_embed_jobs = co.embed_jobs.list()

        print(my_embed_jobs)

        emb_result = co.wait(job)

        self.assertEqual(emb_result.status, "complete")

        co.embed_jobs.cancel(job.job_id)

        co.datasets.delete(dataset.id or "")

    def test_rerank(self) -> None:
        docs = [
            'Carson City is the capital city of the American state of Nevada.',
            'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
            'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
            'Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.']

        response = co.rerank(
            model='rerank-english-v2.0',
            query='What is the capital of the United States?',
            documents=docs,
            top_n=3,
        )

        print(response)

    @unittest.skipIf(os.getenv("CO_API_URL") is not None, "Doesn't work in staging.")
    def test_classify(self) -> None:
        examples = [
            ClassifyExample(
                text="Dermatologists don't like her!", label="Spam"),
            ClassifyExample(text="'Hello, open to this?'", label="Spam"),
            ClassifyExample(
                text="I need help please wire me $1000 right now", label="Spam"),
            ClassifyExample(text="Nice to know you ;)", label="Spam"),
            ClassifyExample(text="Please help me?", label="Spam"),
            ClassifyExample(
                text="Your parcel will be delivered today", label="Not spam"),
            ClassifyExample(
                text="Review changes to our Terms and Conditions", label="Not spam"),
            ClassifyExample(text="Weekly sync notes", label="Not spam"),
            ClassifyExample(
                text="'Re: Follow up from today's meeting'", label="Not spam"),
            ClassifyExample(text="Pre-read for tomorrow", label="Not spam"),
        ]
        inputs = [
            "Confirm your email address",
            "hey i need u to send some $",
        ]
        response = co.classify(
            inputs=inputs,
            examples=examples,
        )
        print(response)

    @unittest.skipIf(os.getenv("CO_API_URL") is not None, "Doesn't work in staging.")
    def test_datasets_crud(self) -> None:
        my_dataset = co.datasets.create(
            name="test",
            type="embed-input",
            data=open(embed_job, 'rb'),
        )

        print(my_dataset)

        my_datasets = co.datasets.list()

        print(my_datasets)

        dataset = co.datasets.get(my_dataset.id or "")

        print(dataset)

        co.datasets.delete(my_dataset.id or "")

    @unittest.skipIf(os.getenv("CO_API_URL") is not None, "Doesn't work in staging.")
    def test_save_load(self) -> None:
        my_dataset = co.datasets.create(
            name="test",
            type="embed-input",
            data=open(embed_job, 'rb'),
        )

        result = co.wait(my_dataset)

        co.utils.save_dataset(result.dataset, "dataset.jsonl")

        # assert files equal
        self.assertTrue(os.path.exists("dataset.jsonl"))
        self.assertEqual(open(embed_job, 'rb').read(),
                         open("dataset.jsonl", 'rb').read())

        print(result)

        co.datasets.delete(my_dataset.id or "")

    def test_summarize(self) -> None:
        text = (
            "Ice cream is a sweetened frozen food typically eaten as a snack or dessert. "
            "It may be made from milk or cream and is flavoured with a sweetener, "
            "either sugar or an alternative, and a spice, such as cocoa or vanilla, "
            "or with fruit such as strawberries or peaches. "
            "It can also be made by whisking a flavored cream base and liquid nitrogen together. "
            "Food coloring is sometimes added, in addition to stabilizers. "
            "The mixture is cooled below the freezing point of water and stirred to incorporate air spaces "
            "and to prevent detectable ice crystals from forming. The result is a smooth, "
            "semi-solid foam that is solid at very low temperatures (below 2 °C or 35 °F). "
            "It becomes more malleable as its temperature increases.\n\n"
            "The meaning of the name \"ice cream\" varies from one country to another. "
            "In some countries, such as the United States, \"ice cream\" applies only to a specific variety, "
            "and most governments regulate the commercial use of the various terms according to the "
            "relative quantities of the main ingredients, notably the amount of cream. "
            "Products that do not meet the criteria to be called ice cream are sometimes labelled "
            "\"frozen dairy dessert\" instead. In other countries, such as Italy and Argentina, "
            "one word is used fo\r all variants. Analogues made from dairy alternatives, "
            "such as goat's or sheep's milk, or milk substitutes "
            "(e.g., soy, cashew, coconut, almond milk or tofu), are available for those who are "
            "lactose intolerant, allergic to dairy protein or vegan."
        )

        response = co.summarize(
            text=text,
        )

        print(response)

    def test_tokenize(self) -> None:
        response = co.tokenize(
            text='tokenize me! :D',
            model='command',
            offline=False,
        )
        print(response)

    def test_detokenize(self) -> None:
        response = co.detokenize(
            tokens=[10104, 12221, 1315, 34, 1420, 69],
            model="command",
            offline=False,
        )
        print(response)

    @unittest.skipIf(os.getenv("CO_API_URL") is not None, "Doesn't work in staging.")
    def test_connectors_crud(self) -> None:
        created_connector = co.connectors.create(
            name="Example connector",
            url="https://dummy-connector-o5btz7ucgq-uc.a.run.app/search",
            service_auth=CreateConnectorServiceAuth(
                token="dummy-connector-token",
                type="bearer",
            )
        )
        print(created_connector)

        connector = co.connectors.get(created_connector.connector.id)

        print(connector)

        updated_connector = co.connectors.update(
            id=connector.connector.id, name="new name")

        print(updated_connector)

        co.connectors.delete(created_connector.connector.id)

    # @unittest.skipIf(os.getenv("CO_API_URL") is not None, "Doesn't work in staging.")
    def test_tool_use(self) -> None:
        tools = [
            Tool(
                name="sales_database",
                description="Connects to a database about sales volumes",
                parameter_definitions={
                    "day": ToolParameterDefinitionsValue(
                        description="Retrieves sales data from this day, formatted as YYYY-MM-DD.",
                        type="str",
                        required=True
                    )}
            )
        ]

        tool_parameters_response = co.chat(
            message="How good were the sales on September 29?",
            tools=tools,
            model="command-nightly",
            preamble="""
                ## Task Description
                You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

                ## Style Guide
                Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
            """
        )

        if tool_parameters_response.tool_calls is not None:
            self.assertEqual(
                tool_parameters_response.tool_calls[0].name, "sales_database")
            self.assertEqual(tool_parameters_response.tool_calls[0].parameters, {
                             "day": "2023-09-29"})
        else:
            raise ValueError("Expected tool calls to be present")

        local_tools = {
            "sales_database": lambda day: {
                "number_of_sales": 120,
                "total_revenue": 48500,
                "average_sale_value": 404.17,
                "date": "2023-09-29"
            }
        }

        tool_results = []
        for tool_call in tool_parameters_response.tool_calls:
            output = local_tools[tool_call.name](**tool_call.parameters)
            outputs = [output]

            tool_results.append(ToolResult(
                call=tool_call,
                outputs=outputs
            ))

        cited_response = co.chat(
            message="",
            tools=tools,
            tool_results=tool_results,
            model="command-nightly",
        )

        self.assertEqual(cited_response.documents, [
            {
                "tool_name": "sales_database",
                "average_sale_value": "404.17",
                "date": "2023-09-29",
                "id": "sales_database:0:1:0",
                "number_of_sales": "120",
                "total_revenue": "48500",
            }
        ])

    def test_local_tokenize(self) -> None:
        response = co.tokenize(
            model="command",
            text="tokenize me! :D"
        )
        print(response)

    def test_local_detokenize(self) -> None:
        response = co.detokenize(
            model="command",
            tokens=[10104, 12221, 1315, 34, 1420, 69]
        )
        print(response)
