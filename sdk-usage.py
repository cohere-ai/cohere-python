import cohere
from cohere.client import ClassifyExample, ParseInfo

co = cohere.Client(
    api_key="xxx",
    num_workers=64,
    request_dict={},
    check_api_key=True,
    client_name="langchain",
    max_retries=3,
    timeout=120,
    api_url="https://api.cohere.com",
)

prediction = co.generate(
    stream=False, # please see the streaming section
    prompt="count with me!",
    prompt_vars={"count": 1},
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
    logit_bias={1: 1},
)

chat = co.chat(
    stream=False,
    message="2",
    model="command",
    return_chat_history=True,
    return_prompt=True,
    return_preamble=True,
    chat_history=[
        {"role": "User", "message": "Count with me!"},
        {"role": "User", "message": "1"}
    ],
    preamble_override=None,
    user_name=None,
    temperature=0.8,
    max_tokens=None,
    p=None,
    k=None,
    logit_bias=None,
    search_queries_only=True,
    documents=[
        {
            "id": "1",
            "text": "The quick brown fox jumped over the lazy dog.",
        }
    ],
    citation_quality="ACCURATE",
    prompt_truncation="AUTO",
    connectors=[
        {
            "id": "web-search",
            "id": "web-search",
            "user_access_token": "xxx",
            "continue_on_failure": False,
            "options": {"site": "cohere.com"}
        }
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
    truncate='NONE',
    input_type='search_document',
    embedding_types=['uint8'],
)

print(embed)

my_dataset = co.create_dataset(
    name="prompt-completion-dataset",
    data=open("./dataset.jsonl", "rb"),
    dataset_type="embed-input",
    # eval_data=open("./prompt-completion.jsonl", "rb"),
    keep_fields="all",
    optional_fields="all",
    parse_info=ParseInfo(separator="\n", delimiter=","),
)

print(my_dataset)

my_datasets = co.list_datasets(
    dataset_type="embed",
    limit=10,
    offset=0,
)

print(my_datasets)

dataset_usage = co.get_dataset_usage()

print(dataset_usage)

my_dataset = co.get_dataset(
    my_dataset.id
)

print(my_dataset)

co.delete_dataset(
    my_dataset.id
)

# start an embed job
job = co.create_embed_job(
    dataset_id=my_dataset.id,
    input_type='search_document',
    model='embed-english-v3.0',
    truncate='END',
    name='my embed job',
)

print(job)

my_embed_jobs = co.list_embed_jobs()

print(my_embed_jobs)

my_embed_job = co.get_embed_job(
    job.id
)

print(my_embed_job)

# cancel an embed job
co.cancel_embed_job(
    my_embed_job.id
)

created_connector = co.create_connector(
    name="Example connector",
    url="http://connector-example.com/search",
    active=True,
    continue_on_failure=False,
    excludes=["excluded"],
    oauth={"client_id": "client_id", "client_secret": "client_secret"},
    service_auth={"username": "username", "password": "password"},
)

print(created_connector)

updated_connector = co.update_connector(
    id=created_connector.id,
    name="Example connector",
    url="http://connector-example.com/search",
    active=True,
    continue_on_failure=False,
    excludes=["excluded"],
    oauth={"client_id": "client_id", "client_secret": "client_secret"},
    service_auth={"username": "username", "password": "password"}
)

print(updated_connector)

connector = co.get_connector(
    id=created_connector.id
)

print(connector)

connectors = co.list_connectors(
    limit=10,
    offset=0,
)

print(connectors)

co.delete_connector(
    id=created_connector.id
)

redirect_url = co.oauth_authorize_connector(
    id=created_connector.id,
    after_token_redirect="https://test.com"
)

print(redirect_url)