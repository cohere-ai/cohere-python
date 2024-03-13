import cohere
from cohere import CreateConnectorServiceAuth, CreateConnectorOAuth, ChatMessage, ChatConnector, ClassifyExample

co = cohere.Client(
    api_key="xxx",
    # num_workers=64, not supported
    # request_dict={}, not supported
    # check_api_key=True, not supported
    client_name="langchain",
    # max_retries=3, not supported via constructor
    timeout=120,
    base_url="https://api.cohere.com",
)

prediction = co.generate(
    prompt="count with me!",
    # prompt_vars={"count": 1}, not supported
    model="command",
    preset="id",
    num_generations=1,
    max_tokens=100,
    temperature=1,
    k=1,
    p=1,
    frequency_penalty=1,
    presence_penalty=1,
    end_sequences=["\n"],
    stop_sequences=["\n"],
    return_likelihoods="ALL",
    truncate="END",
    # logit_bias={1: 1}, not supported
)

chat = co.chat(
    message="2",
    model="command",
    # return_chat_history=True, not supported
    # return_prompt=True, not supported
    # return_preamble=True, not supported
    chat_history=[
        ChatMessage(role="USER", message="Count with me!"),
        ChatMessage(role="USER", message="1")
    ],
    preamble=None,
    # user_name=None, not supported
    temperature=0.8,
    # max_tokens=None, not supported
    # p=None, not supported
    # k=None, not supported
    # logit_bias=None, not supported
    search_queries_only=True,
    documents=[
        {
            "id": "1",
            "text": "The quick brown fox jumped over the lazy dog.",
        }
    ],
    # citation_quality="ACCURATE", not supported
    prompt_truncation="AUTO",
    connectors=[
        ChatConnector(
            id="web-search",
            user_access_token="xxx",
            continue_on_failure=False,
            options={"site": "cohere.com"}
        )
    ],
)

print('chat: {}'.format(chat.text))

classifies = co.classify(
    examples=[
        ClassifyExample(text="orange", label="fruit"),
        ClassifyExample(text="pear", label="fruit"),
        ClassifyExample(text="lettuce", label="vegetable"),
        ClassifyExample(text="cauliflower", label="vegetable")
    ],
    inputs=[
        "Abiu",
    ],
    model="embed-multilingual-v2.0",
    preset="id",
    truncate="END",
)

print('classifies: {}'.format(classifies.classifications[0].prediction))

tokenise = co.tokenize(
    text="token mctoken face",
    model="base"
)

print('tokenise: {}'.format(tokenise.tokens))

detokenise = co.detokenize(
    tokens=tokenise.tokens,
    model="base"
)

print('detokenise: {}'.format(detokenise.text))

summarise = co.summarize(
    text="the quick brown fox jumped over the lazy dog and then the dog jumped over the fox the quick brown fox jumped over the lazy dog the quick brown fox jumped over the lazy dog the quick brown fox jumped over the lazy dog the quick brown fox jumped over the lazy dog",
    model="command",
    length="short",
    format="paragraph",
    temperature=1,
    additional_command=None,
    extractiveness="low",
)

print('summarise: {}'.format(summarise))

docs = ['Carson City is the capital city of the American state of Nevada.',
        'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
        'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
        'Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.']

rerank = co.rerank(
    model='rerank-english-v2.0',
    query='What is the capital of the United States?',
    documents=docs,
    top_n=3,
    max_chunks_per_doc=1,
)

print('rerank: {}'.format(rerank.results[0].index))

embed = co.embed(
    texts=['hello', 'goodbye'],
    model='embed-english-v3.0',
    truncate="NONE",
    input_type="search_document",
    embedding_types=['uint8'],
)

print(embed)

my_dataset = co.datasets.create(
    name="prompt-completion-dataset",
    data=open("./dataset.jsonl", "rb"),
    type="embed-input",
    # eval_data=open("./prompt-completion.jsonl", "rb"),
    keep_fields="all",
    optional_fields="all",
    # parse_info=ParseInfo(separator="\n", delimiter=","), not supported
)

print(my_dataset)

my_datasets = co.datasets.list(
    dataset_type="embed",
    limit="10",
    offset="0",
)

print(my_datasets)

dataset_usage = co.datasets.get_usage()

print(dataset_usage)

ds = co.datasets.get(
    my_dataset.id or ""
)

print(ds)

co.datasets.delete(
    ds.dataset.id
)

# start an embed job
job = co.embed_jobs.create(
    dataset_id=ds.dataset.id,
    input_type="search_document",
    model='embed-english-v3.0',
    truncate="END",
    name='my embed job',
)

print(job)

my_embed_jobs = co.embed_jobs.list()

print(my_embed_jobs)

my_embed_job = co.embed_jobs.get(
    job.job_id
)

print(my_embed_job)

# cancel an embed job
co.embed_jobs.cancel(
    my_embed_job.job_id
)

created_connector = co.connectors.create(
    name="Example connector",
    url="http://connector-example.com/search",
    active=True,
    continue_on_failure=False,
    excludes=["excluded"],
    oauth=CreateConnectorOAuth(client_id="client_id", client_secret="client_secret"),
    service_auth=CreateConnectorServiceAuth(type="bearer", token="password"),
)

print(created_connector)

updated_connector = co.connectors.update(
    id=created_connector.connector.id,
    name="Example connector",
    url="http://connector-example.com/search",
    active=True,
    continue_on_failure=False,
    excludes=["excluded"],
    oauth=CreateConnectorOAuth(client_id="client_id", client_secret="client_secret"),
    service_auth=CreateConnectorServiceAuth(type="bearer", token="password"),
)

print(updated_connector)

connector = co.connectors.get(
    id=created_connector.connector.id
)

print(connector)

connectors = co.connectors.list(
    limit=10,
    offset=0,
)

print(connectors)

co.connectors.delete(
    id=created_connector.connector.id
)

redirect_url = co.connectors.o_auth_authorize(
    id=created_connector.connector.id,
    after_token_redirect="https://test.com"
)

print(redirect_url)