import json
import os
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from .classification import Classification, Classifications
from .embeddings import Embeddings
from .error import CohereError
from .generation import Generations, StreamingGenerations
from .chat import Chat, StreamingChat
from .rerank import Reranking
from .summary import Summary
from .mode import Mode
import typing
from ..lazy_aws_deps import lazy_boto3, lazy_botocore, lazy_sagemaker

class Client:
    def __init__(
           self,
            aws_region: typing.Optional[str] = None,
        ):
        """
        By default we assume region configured in AWS CLI (`aws configure get region`). You can change the region with
        `aws configure set region us-west-2` or override it with `region_name` parameter.
        """
        self._client = lazy_boto3().client("sagemaker-runtime", region_name=aws_region)
        self._service_client = lazy_boto3().client("sagemaker", region_name=aws_region)
        if os.environ.get('AWS_DEFAULT_REGION') is None:
            os.environ['AWS_DEFAULT_REGION'] = aws_region
        self._sess = lazy_sagemaker().Session(sagemaker_client=self._service_client)
        self.mode = Mode.SAGEMAKER



    def _does_endpoint_exist(self, endpoint_name: str) -> bool:
        try:
            self._service_client.describe_endpoint(EndpointName=endpoint_name)
        except lazy_botocore().ClientError:
            return False
        return True

    def connect_to_endpoint(self, endpoint_name: str) -> None:
        """Connects to an existing SageMaker endpoint.

        Args:
            endpoint_name (str): The name of the endpoint.

        Raises:
            CohereError: Connection to the endpoint failed.
        """
        if not self._does_endpoint_exist(endpoint_name):
            raise CohereError(f"Endpoint {endpoint_name} does not exist.")
        self._endpoint_name = endpoint_name

    def _s3_models_dir_to_tarfile(self, s3_models_dir: str) -> str:
        """
        Compress an S3 folder which contains one or several fine-tuned models to a tar file.
        If the S3 folder contains only one fine-tuned model, it simply returns the path to that model.
        If the S3 folder contains several fine-tuned models, it download all models, aggregates them into a single
        tar.gz file.

        Args:
            s3_models_dir (str): S3 URI pointing to a folder

        Returns:
            str: S3 URI pointing to the `models.tar.gz` file
        """

        s3_models_dir = s3_models_dir.rstrip("/") + "/"

        # Links of all fine-tuned models in s3_models_dir. Their format should be .tar.gz
        s3_tar_models = [
            s3_path
            for s3_path in lazy_sagemaker().s3.S3Downloader.list(s3_models_dir, sagemaker_session=self._sess)
            if (
                s3_path.endswith(".tar.gz")  # only .tar.gz files
                and (s3_path.split("/")[-1] != "models.tar.gz")  # exclude the .tar.gz file we are creating
                and (s3_path.rsplit("/", 1)[0] == s3_models_dir[:-1])  # only files at the root of s3_models_dir
            )
        ]

        if len(s3_tar_models) == 0:
            raise CohereError(f"No fine-tuned models found in {s3_models_dir}")
        elif len(s3_tar_models) == 1:
            print(f"Found one fine-tuned model: {s3_tar_models[0]}")
            return s3_tar_models[0]

        # More than one fine-tuned model found, need to aggregate them into a single .tar.gz file
        with tempfile.TemporaryDirectory() as tmpdir:
            local_tar_models_dir = os.path.join(tmpdir, "tar")
            local_models_dir = os.path.join(tmpdir, "models")

            # Download and extract all fine-tuned models
            for s3_tar_model in s3_tar_models:
                print(f"Adding fine-tuned model: {s3_tar_model}")
                lazy_sagemaker().s3.S3Downloader.download(s3_tar_model, local_tar_models_dir, sagemaker_session=self._sess)
                with tarfile.open(os.path.join(local_tar_models_dir, s3_tar_model.split("/")[-1])) as tar:
                    tar.extractall(local_models_dir)

            # Compress local_models_dir to a tar.gz file
            model_tar = os.path.join(tmpdir, "models.tar.gz")
            with tarfile.open(model_tar, "w:gz") as tar:
                tar.add(local_models_dir, arcname=".")

            # Upload the new tarfile containing all models to s3
            # Very important to remove the trailing slash from s3_models_dir otherwise it just doesn't upload
            model_tar_s3 = lazy_sagemaker().s3.S3Uploader.upload(model_tar, s3_models_dir[:-1], sagemaker_session=self._sess)

            # sanity check
            assert s3_models_dir + "models.tar.gz" in lazy_sagemaker().s3.S3Downloader.list(s3_models_dir, sagemaker_session=self._sess)

        return model_tar_s3

    def create_endpoint(
        self,
        arn: str,
        endpoint_name: str,
        s3_models_dir: Optional[str] = None,
        instance_type: str = "ml.g4dn.xlarge",
        n_instances: int = 1,
        recreate: bool = False,
        role: Optional[str] = None,
    ) -> None:
        """Creates and deploys a SageMaker endpoint.

        Args:
            arn (str): The product ARN. Refers to a ready-to-use model (model package) or a fine-tuned model
                (algorithm).
            endpoint_name (str): The name of the endpoint.
            s3_models_dir (str, optional): S3 URI pointing to the folder containing fine-tuned models. Defaults to None.
            instance_type (str, optional): The EC2 instance type to deploy the endpoint to. Defaults to "ml.g4dn.xlarge".
            n_instances (int, optional): Number of endpoint instances. Defaults to 1.
            recreate (bool, optional): Force re-creation of endpoint if it already exists. Defaults to False.
            role (str, optional): The IAM role to use for the endpoint. If not provided, sagemaker.get_execution_role()
                will be used to get the role. This should work when one uses the client inside SageMaker. If this errors
                out, the default role "ServiceRoleSagemaker" will be used, which generally works outside of SageMaker.
        """
        # First, check if endpoint already exists
        if self._does_endpoint_exist(endpoint_name):
            if recreate:
                self.connect_to_endpoint(endpoint_name)
                self.delete_endpoint()
            else:
                raise CohereError(f"Endpoint {endpoint_name} already exists and recreate={recreate}.")

        kwargs = {}
        model_data = None
        validation_params = dict()
        useBoto = False
        if s3_models_dir is not None:
            # If s3_models_dir is given, we assume to have custom fine-tuned models -> Algorithm
            kwargs["algorithm_arn"] = arn
            model_data = self._s3_models_dir_to_tarfile(s3_models_dir)
        else:
            # If no s3_models_dir is given, we assume to use a pre-trained model -> ModelPackage
            kwargs["model_package_arn"] = arn

            # For now only non-finetuned models can use these timeouts
            validation_params = dict(
                model_data_download_timeout=2400,
                container_startup_health_check_timeout=2400
            )
            useBoto = True

        # Out of precaution, check if there is an endpoint config and delete it if that's the case
        # Otherwise it might block deployment
        try:
            self._service_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except lazy_botocore().ClientError:
            pass

        try:
            self._service_client.delete_model(ModelName=endpoint_name)
        except lazy_botocore().ClientError:
            pass

        if role is None:
            if useBoto:
                accountID = lazy_sagemaker().account_id()
                role = f"arn:aws:iam::{accountID}:role/ServiceRoleSagemaker"
            else:
                try:
                    role = lazy_sagemaker().get_execution_role()
                except ValueError:
                    print("Using default role: 'ServiceRoleSagemaker'.")
                    role = "ServiceRoleSagemaker"

        # deploy fine-tuned model using sagemaker SDK
        if s3_models_dir is not None:
            model = lazy_sagemaker().ModelPackage(
                role=role,
                model_data=model_data,
                sagemaker_session=self._sess,  # makes sure the right region is used
                **kwargs
            )

            try:
                model.deploy(
                    n_instances,
                    instance_type,
                    endpoint_name=endpoint_name,
                    **validation_params
                )
            except lazy_botocore().ParamValidationError:
                # For at least some versions of python 3.6, SageMaker SDK does not support the validation_params
                model.deploy(n_instances, instance_type, endpoint_name=endpoint_name)
        else:
            # deploy pre-trained model using boto to add InferenceAmiVersion
            self._service_client.create_model(
                ModelName=endpoint_name,
                ExecutionRoleArn=role,
                EnableNetworkIsolation=True,
                PrimaryContainer={
                    'ModelPackageName': arn,
                },
            )
            self._service_client.create_endpoint_config(
                EndpointConfigName=endpoint_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': endpoint_name,
                        'InstanceType': instance_type,
                        'InitialInstanceCount': n_instances,
                        'InferenceAmiVersion': 'al2-ami-sagemaker-inference-gpu-2'
                    },
                ],
            )
            self._service_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_name,
            )

            waiter = self._service_client.get_waiter('endpoint_in_service')
            try:
                print(f"Waiting for endpoint {endpoint_name} to be in service...")
                waiter.wait(
                    EndpointName=endpoint_name,
                    WaiterConfig={
                        'Delay': 30,
                        'MaxAttempts': 80
                    }
                )
            except Exception as e:
                raise CohereError(f"Failed to create endpoint: {e}")
        self.connect_to_endpoint(endpoint_name)

    def chat(
        self,
        message: str,
        stream: Optional[bool] = False,
        preamble: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        # should only be passed for stacked finetune deployment
        model: Optional[str] = None,
        # should only be passed for Bedrock mode; ignored otherwise
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        p: Optional[float] = None,
        k: Optional[float] = None,
        max_tokens: Optional[int] = None,
        search_queries_only: Optional[bool] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        prompt_truncation: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        raw_prompting: Optional[bool] = False,
        return_prompt: Optional[bool] = False,
        variant: Optional[str] = None,
    ) -> Union[Chat, StreamingChat]:
        """Returns a Chat object with the query reply.

        Args:
            message (str): The message to send to the chatbot.

            stream (bool): Return streaming tokens.

            preamble (str): (Optional) A string to override the preamble.
            chat_history (List[Dict[str, str]]): (Optional) A list of entries used to construct the conversation. If provided, these messages will be used to build the prompt and the conversation_id will be ignored so no data will be stored to maintain state.

            model (str): (Optional) The model to use for generating the response. Should only be passed for stacked finetune deployment.
            model_id (str): (Optional) The model to use for generating the response. Should only be passed for Bedrock mode; ignored otherwise.
            temperature (float): (Optional) The temperature to use for the response. The higher the temperature, the more random the response.
            p (float): (Optional) The nucleus sampling probability.
            k (float): (Optional) The top-k sampling probability.
            max_tokens (int): (Optional) The max tokens generated for the next reply.

            search_queries_only (bool): (Optional) When true, the response will only contain a list of generated `search_queries`, no reply from the model to the user's message will be generated.
            documents (List[Dict[str, str]]): (Optional) Documents to use to generate grounded response with citations. Example:
                documents=[
                    {
                        "id": "national_geographic_everest",
                        "title": "Height of Mount Everest",
                        "snippet": "The height of Mount Everest is 29,035 feet",
                        "url": "https://education.nationalgeographic.org/resource/mount-everest/",
                    },
                    {
                        "id": "national_geographic_mariana",
                        "title": "Depth of the Mariana Trench",
                        "snippet": "The depth of the Mariana Trench is 36,070 feet",
                        "url": "https://www.nationalgeographic.org/activity/mariana-trench-deepest-place-earth",
                    },
                ],
            prompt_truncation (str) (Optional): Defaults to `OFF`. Dictates how the prompt will be constructed. With `prompt_truncation` set to "AUTO_PRESERVE_ORDER", some elements from `chat_history` and `documents` will be dropped in an attempt to construct a prompt that fits within the model's context length limit. During this process the order of the documents and chat history will be preserved as they are inputted into the API. With `prompt_truncation` set to "OFF", no elements will be dropped. If the sum of the inputs exceeds the model's context length limit, a `TooManyTokens` error will be raised.
        Returns:
            a Chat object if stream=False, or a StreamingChat object if stream=True

        Examples:
            A simple chat message:
                >>> res = co.chat(message="Hey! How are you doing today?")
                >>> print(res.text)
            Streaming chat:
                >>> res = co.chat(
                >>>     message="Hey! How are you doing today?",
                >>>     stream=True)
                >>> for token in res:
                >>>     print(token)
            Stateless chat with chat history:
                >>> res = co.chat(
                >>>     chat_history=[
                >>>         {'role': 'User', message': 'Hey! How are you doing today?'},
                >>>         {'role': 'Chatbot', message': 'I am doing great! How can I help you?'},
                >>>     message="Tell me a joke!",
                >>>     ])
                >>> print(res.text)
            Chat message with documents to use to generate the response:
                >>> res = co.chat(
                >>>     "How deep in the Mariana Trench",
                >>>     documents=[
                >>>         {
                >>>            "id": "national_geographic_everest",
                >>>            "title": "Height of Mount Everest",
                >>>            "snippet": "The height of Mount Everest is 29,035 feet",
                >>>            "url": "https://education.nationalgeographic.org/resource/mount-everest/",
                >>>         },
                >>>         {
                >>>             "id": "national_geographic_mariana",
                >>>             "title": "Depth of the Mariana Trench",
                >>>             "snippet": "The depth of the Mariana Trench is 36,070 feet",
                >>>             "url": "https://www.nationalgeographic.org/activity/mariana-trench-deepest-place-earth",
                >>>         },
                >>>       ])
                >>> print(res.text)
                >>> print(res.citations)
                >>> print(res.documents)
            Generate search queries for fetching documents to use in chat:
                >>> res = co.chat(
                >>>     "What is the height of Mount Everest?",
                >>>      search_queries_only=True)
                >>> if res.is_search_required:
                >>>      print(res.search_queries)
        """
         
        if self.mode == Mode.SAGEMAKER and self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")
        json_params = {
            "model": model,
            "message": message,
            "chat_history": chat_history,
            "preamble": preamble,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "p": p,
            "k": k,
            "tools": tools,
            "tool_results": tool_results,
            "search_queries_only": search_queries_only,
            "documents": documents,
            "raw_prompting": raw_prompting,
            "return_prompt": return_prompt,
            "prompt_truncation": prompt_truncation
        }
    
        for key, value in list(json_params.items()):
            if value is None:
                del json_params[key]

        if self.mode == Mode.SAGEMAKER:
            return self._sagemaker_chat(json_params, variant)
        elif self.mode == Mode.BEDROCK:
            return self._bedrock_chat(json_params, model_id)
        else:
            raise CohereError("Unsupported mode")

    def _sagemaker_chat(self, json_params: Dict[str, Any], variant: str) :
        json_body = json.dumps(json_params)
        params = {
            'EndpointName': self._endpoint_name,
            'ContentType': 'application/json',
            'Body': json_body,
        }
        if variant:
            params['TargetVariant'] = variant

        try:
            if json_params['stream']:
                result = self._client.invoke_endpoint_with_response_stream(
                    **params)
                return StreamingChat(result['Body'], self.mode)
            else:
                result = self._client.invoke_endpoint(**params)
                return Chat.from_dict(json.loads(result['Body'].read().decode()))
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

    def _bedrock_chat(self, json_params: Dict[str, Any], model_id: str) :
        if not model_id:
            raise CohereError("must supply model_id arg when calling bedrock")
        if json_params['stream']:
            stream = json_params['stream']
        else:
            stream = False
        # Bedrock does not expect the stream key to be present in the body, use invoke_model_with_response_stream to indicate stream mode
        del json_params['stream']

        json_body = json.dumps(json_params)
        params = {
            'body': json_body,
            'modelId': model_id,
        }

        try:
            if stream:
                result = self._client.invoke_model_with_response_stream(
                    **params)
                return StreamingChat(result['body'], self.mode)
            else:
                result = self._client.invoke_model(**params)
                return Chat.from_dict(
                    json.loads(result['body'].read().decode()))
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

    def generate(
        self,
        prompt: str,
        # should only be passed for stacked finetune deployment
        model: Optional[str] = None,
        # should only be passed for Bedrock mode; ignored otherwise
        model_id: Optional[str] = None,
        # requires DB with presets
        # preset: str = None,
        num_generations: int = 1,
        max_tokens: int = 400,
        temperature: float = 1.0,
        k: int = 0,
        p: float = 0.75,
        stop_sequences: Optional[List[str]] = None,
        return_likelihoods: Optional[str] = None,
        truncate: Optional[str] = None,
        variant: Optional[str] = None,
        stream: Optional[bool] = True,
    ) -> Union[Generations, StreamingGenerations]:
        if self.mode == Mode.SAGEMAKER and self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

        json_params = {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': k,
            'p': p,
            'stop_sequences': stop_sequences,
            'return_likelihoods': return_likelihoods,
            'truncate': truncate,
            'stream': stream,
        }
        for key, value in list(json_params.items()):
            if value is None:
                del json_params[key]

        if self.mode == Mode.SAGEMAKER:
            # TODO: Bedrock should support this param too
            json_params['num_generations'] = num_generations
            return self._sagemaker_generations(json_params, variant)
        elif self.mode == Mode.BEDROCK:
            return self._bedrock_generations(json_params, model_id)
        else:
            raise CohereError("Unsupported mode")

    def _sagemaker_generations(self, json_params: Dict[str, Any], variant: str) :
        json_body = json.dumps(json_params)
        params = {
            'EndpointName': self._endpoint_name,
            'ContentType': 'application/json',
            'Body': json_body,
        }
        if variant:
            params['TargetVariant'] = variant

        try:
            if json_params['stream']:
                result = self._client.invoke_endpoint_with_response_stream(
                    **params)
                return StreamingGenerations(result['Body'], self.mode)
            else:
                result = self._client.invoke_endpoint(**params)
                return Generations(
                    json.loads(result['Body'].read().decode())['generations'])
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

    def _bedrock_generations(self, json_params: Dict[str, Any], model_id: str) :
        if not model_id:
            raise CohereError("must supply model_id arg when calling bedrock")
        json_body = json.dumps(json_params)
        params = {
            'body': json_body,
            'modelId': model_id,
        }

        try:
            if json_params['stream']:
                result = self._client.invoke_model_with_response_stream(
                    **params)
                return StreamingGenerations(result['body'], self.mode)
            else:
                result = self._client.invoke_model(**params)
                return Generations(
                    json.loads(result['body'].read().decode())['generations'])
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

    def embed(
        self,
        texts: List[str],
        truncate: Optional[str] = None,
        variant: Optional[str] = None,
        input_type: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Embeddings:
        json_params = {
            'texts': texts,
            'truncate': truncate,
            "input_type": input_type
        }
        for key, value in list(json_params.items()):
            if value is None:
                del json_params[key]
        
        if self.mode == Mode.SAGEMAKER:
            return self._sagemaker_embed(json_params, variant)
        elif self.mode == Mode.BEDROCK:
            return self._bedrock_embed(json_params, model_id)
        else:
            raise CohereError("Unsupported mode")

    def _sagemaker_embed(self, json_params: Dict[str, Any], variant: str):
        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")
        
        json_body = json.dumps(json_params)
        params = {
            'EndpointName': self._endpoint_name,
            'ContentType': 'application/json',
            'Body': json_body,
        }
        if variant:
            params['TargetVariant'] = variant

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result['Body'].read().decode())
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return Embeddings(response['embeddings'])

    def _bedrock_embed(self, json_params: Dict[str, Any], model_id: str):
        if not model_id:
            raise CohereError("must supply model_id arg when calling bedrock")
        json_body = json.dumps(json_params)
        params = {
            'body': json_body,
            'modelId': model_id,
        }

        try:
            result = self._client.invoke_model(**params)
            response = json.loads(result['body'].read().decode())
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return Embeddings(response['embeddings'])


    def rerank(self,
               query: str,
               documents: Union[List[str], List[Dict[str, Any]]],
               top_n: Optional[int] = None,
               variant: Optional[str] = None,
               max_chunks_per_doc: Optional[int] = None,
               rank_fields: Optional[List[str]] = None) -> Reranking:
        """Returns an ordered list of documents oridered by their relevance to the provided query
        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            top_n (int): (optional) The number of results to return, defaults to return all results
            max_chunks_per_doc (int): (optional) The maximum number of chunks derived from a document
            rank_fields (list[str]): (optional) The fields used for reranking. This parameter is only supported for rerank v3 models
        """

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

        parsed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                parsed_docs.append({'text': doc})
            elif isinstance(doc, dict):
                parsed_docs.append(doc)
            else:
                raise CohereError(
                    message='invalid format for documents, must be a list of strings or dicts')

        json_params = {
            "query": query,
            "documents": parsed_docs,
            "top_n": top_n,
            "return_documents": False,
            "max_chunks_per_doc" : max_chunks_per_doc,
            "rank_fields": rank_fields
        }
        json_body = json.dumps(json_params)

        params = {
            'EndpointName': self._endpoint_name,
            'ContentType': 'application/json',
            'Body': json_body,
        }
        if variant is not None:
            params['TargetVariant'] = variant

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result['Body'].read().decode())
            reranking = Reranking(response)
            for rank in reranking.results:
                rank.document = parsed_docs[rank.index]
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return reranking

    def classify(self, input: List[str], name: str) -> Classifications:

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

        json_params = {"texts": input, "model_id": name}
        json_body = json.dumps(json_params)

        params = {
            "EndpointName": self._endpoint_name,
            "ContentType": "application/json",
            "Body": json_body,
        }

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result["Body"].read().decode())
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return Classifications([Classification(classification) for classification in response])

    def create_finetune(
        self,
        name: str,
        train_data: str,
        s3_models_dir: str,
        arn: Optional[str] = None,
        eval_data: Optional[str] = None,
        instance_type: str = "ml.g4dn.xlarge",
        training_parameters: Dict[str, Any] = {},  # Optional, training algorithm specific hyper-parameters
        role: Optional[str] = None,
        base_model_id: Optional[str] = None,
    ) -> Optional[str]:
        """Creates a fine-tuning job and returns an optional fintune job ID.

        Args:
            name (str): The name to give to the fine-tuned model.
            train_data (str): An S3 path pointing to the training data.
            s3_models_dir (str): An S3 path pointing to the directory where the fine-tuned model will be saved.
            arn (str, optional): The product ARN of the fine-tuning package. Required in Sagemaker mode and ignored otherwise
            eval_data (str, optional): An S3 path pointing to the eval data. Defaults to None.
            instance_type (str, optional): The EC2 instance type to use for training. Defaults to "ml.g4dn.xlarge".
            training_parameters (Dict[str, Any], optional): Additional training parameters. Defaults to {}.
            role (str, optional): The IAM role to use for the endpoint. 
                In Bedrock this mode is required and is used to access s3 input and output data.
                If not provided in sagemaker, sagemaker.get_execution_role()will be used to get the role.
                This should work when one uses the client inside SageMaker. If this errors
                out, the default role "ServiceRoleSagemaker" will be used, which generally works outside of SageMaker.
            base_model_id (str, optional): The ID of the Bedrock base model to finetune with. Required in Bedrock mode and ignored otherwise.
        """
        assert name != "model", "name cannot be 'model'"

        if self.mode == Mode.BEDROCK:
            return self._bedrock_create_finetune(name=name, train_data=train_data, s3_models_dir=s3_models_dir, base_model=base_model_id, eval_data=eval_data, training_parameters=training_parameters, role=role)

        s3_models_dir = s3_models_dir.rstrip("/") + "/"

        if role is None:
            try:
                role = lazy_sagemaker().get_execution_role()
            except ValueError:
                print("Using default role: 'ServiceRoleSagemaker'.")
                role = "ServiceRoleSagemaker"

        training_parameters.update({"name": name})
        estimator = lazy_sagemaker().algorithm.AlgorithmEstimator(
            algorithm_arn=arn,
            role=role,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=self._sess,
            output_path=s3_models_dir,
            hyperparameters=training_parameters,
        )

        inputs = {}
        if not train_data.startswith("s3:"):
            raise ValueError("train_data must point to an S3 location.")
        inputs["training"] = train_data
        if eval_data is not None:
            if not eval_data.startswith("s3:"):
                raise ValueError("eval_data must point to an S3 location.")
            inputs["evaluation"] = eval_data
        estimator.fit(inputs=inputs)
        job_name = estimator.latest_training_job.name

        current_filepath = f"{s3_models_dir}{job_name}/output/model.tar.gz"

        s3_resource = lazy_boto3().resource("s3")

        # Copy new model to root of output_model_dir
        bucket, old_key = lazy_sagemaker().s3.parse_s3_url(current_filepath)
        _, new_key = lazy_sagemaker().s3.parse_s3_url(f"{s3_models_dir}{name}.tar.gz")
        s3_resource.Object(bucket, new_key).copy(CopySource={"Bucket": bucket, "Key": old_key})

        # Delete old dir
        bucket, old_short_key = lazy_sagemaker().s3.parse_s3_url(s3_models_dir + job_name)
        s3_resource.Bucket(bucket).objects.filter(Prefix=old_short_key).delete()

    def export_finetune(
        self,
        name: str,
        s3_checkpoint_dir: str,
        s3_output_dir: str,
        arn: str,
        instance_type: str = "ml.p4de.24xlarge",
        role: Optional[str] = None,
    ) -> None:
        """Export the merged weights to the TensorRT-LLM inference engine.

        Args:
        name (str): The name used while writing the exported model to the output directory.
        s3_checkpoint_dir (str): An S3 path pointing to the directory of the model checkpoint (merged weights).
        s3_output_dir (str): An S3 path pointing to the directory where the TensorRT-LLM engine will be saved.
        arn (str): The product ARN of the bring your own finetuning algorithm.
        instance_type (str, optional): The EC2 instance type to use for export. Defaults to "ml.p4de.24xlarge".
        role (str, optional): The IAM role to use for export.
            If not provided, sagemaker.get_execution_role() will be used to get the role.
            This should work when one uses the client inside SageMaker. If this errors out,
            the default role "ServiceRoleSagemaker" will be used, which generally works outside SageMaker.
        """
        if name == "model":
            raise ValueError("name cannot be 'model'")

        s3_output_dir = s3_output_dir.rstrip("/") + "/"

        if role is None:
            try:
                role = lazy_sagemaker().get_execution_role()
            except ValueError:
                print("Using default role: 'ServiceRoleSagemaker'.")
                role = "ServiceRoleSagemaker"

        export_parameters = {"name": name}

        estimator = lazy_sagemaker().algorithm.AlgorithmEstimator(
            algorithm_arn=arn,
            role=role,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=self._sess,
            output_path=s3_output_dir,
            hyperparameters=export_parameters,
        )

        if not s3_checkpoint_dir.startswith("s3:"):
            raise ValueError("s3_checkpoint_dir must point to an S3 location.")
        inputs = {"checkpoint": s3_checkpoint_dir}

        estimator.fit(inputs=inputs)

        job_name = estimator.latest_training_job.name
        current_filepath = f"{s3_output_dir}{job_name}/output/model.tar.gz"

        s3_resource = lazy_boto3().resource("s3")

        # Copy the exported TensorRT-LLM engine to the root of s3_output_dir
        bucket, old_key = lazy_sagemaker().s3.parse_s3_url(current_filepath)
        _, new_key = lazy_sagemaker().s3.parse_s3_url(f"{s3_output_dir}{name}.tar.gz")
        s3_resource.Object(bucket, new_key).copy(CopySource={"Bucket": bucket, "Key": old_key})

        # Delete the old S3 directory
        bucket, old_short_key = lazy_sagemaker().s3.parse_s3_url(f"{s3_output_dir}{job_name}")
        s3_resource.Bucket(bucket).objects.filter(Prefix=old_short_key).delete()

    def wait_for_finetune_job(self, job_id: str, timeout: int = 2*60*60) -> str:
        """Waits for a finetune job to complete and returns a model arn if complete. Throws an exception if timeout occurs or if job does not complete successfully
        Args:
            job_id (str): The arn of the model customization job
            timeout(int, optional): Timeout in seconds
        """
        end = time.time() + timeout
        while True:
            customization_job = self._service_client.get_model_customization_job(jobIdentifier=job_id)
            job_status = customization_job["status"]
            if job_status in ["Completed", "Failed", "Stopped"]:
                break
            if time.time() > end:
                raise CohereError("could not complete finetune within timeout")
            time.sleep(10)
        
        if job_status != "Completed":
            raise CohereError(f"finetune did not finish successfuly, ended with {job_status} status")
        return customization_job["outputModelArn"]

    def provision_throughput(
        self,
        model_id: str,
        name: str,
        model_units: int,
        commitment_duration: Optional[str] = None
    ) -> str:
        """Returns the provisined model arn
        Args:
            model_id (str): The ID or ARN of the model to provision
            name (str): Name of the provisioned throughput model
            model_units (int): Number of units to provision
            commitment_duration (str, optional): Commitment duration, one of ("OneMonth", "SixMonths"), defaults to no commitment if unspecified
        """
        if self.mode != Mode.BEDROCK:
            raise ValueError("can only provision throughput in bedrock")
        kwargs = {}
        if commitment_duration:
            kwargs["commitmentDuration"] = commitment_duration

        response = self._service_client.create_provisioned_model_throughput(
            provisionedModelName=name,
            modelId=model_id,
            modelUnits=model_units,
            **kwargs
        )
        return response["provisionedModelArn"]

    def _bedrock_create_finetune(
        self,
        name: str,
        train_data: str,
        s3_models_dir: str,
        base_model: str,
        eval_data: Optional[str] = None,
        training_parameters: Dict[str, Any] = {},  # Optional, training algorithm specific hyper-parameters
        role: Optional[str] = None,
    ) -> None:
        if not name:
            raise ValueError("name must not be empty")
        if not role:
            raise ValueError("must provide a role ARN for bedrock finetuning (https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-iam-role.html)")
        if not train_data.startswith("s3:"):
            raise ValueError("train_data must point to an S3 location.")
        if eval_data:
            if not eval_data.startswith("s3:"):
                raise ValueError("eval_data must point to an S3 location.")
            validationDataConfig = {
                "validators": [{
                    "s3Uri": eval_data
                }]
            }

        job_name = f"{name}-job"
        customization_job = self._service_client.create_model_customization_job(
            jobName=job_name, 
            customModelName=name, 
            roleArn=role,
            baseModelIdentifier=base_model,
            trainingDataConfig={"s3Uri": train_data},
            validationDataConfig=validationDataConfig,
            outputDataConfig={"s3Uri": s3_models_dir}, 
            hyperParameters=training_parameters
        )
        return customization_job["jobArn"]


    def summarize(
        self,
        text: str,
        length: Optional[str] = "auto",
        format_: Optional[str] = "auto",
        # Only summarize-xlarge is supported on Sagemaker
        # model: Optional[str] = "summarize-xlarge",
        extractiveness: Optional[str] = "auto",
        temperature: Optional[float] = 0.3,
        additional_command: Optional[str] = "",
        variant: Optional[str] = None
    ) -> Summary:

        if self._endpoint_name is None:
            raise CohereError("No endpoint connected. "
                              "Run connect_to_endpoint() first.")

        json_params = {
            'text': text,
            'length': length,
            'format': format_,
            'extractiveness': extractiveness,
            'temperature': temperature,
            'additional_command': additional_command,
        }
        for key, value in list(json_params.items()):
            if value is None:
                del json_params[key]
        json_body = json.dumps(json_params)

        params = {
            'EndpointName': self._endpoint_name,
            'ContentType': 'application/json',
            'Body': json_body,
        }
        if variant is not None:
            params['TargetVariant'] = variant

        try:
            result = self._client.invoke_endpoint(**params)
            response = json.loads(result['Body'].read().decode())
            summary = Summary(response)
        except lazy_botocore().EndpointConnectionError as e:
            raise CohereError(str(e))
        except Exception as e:
            # TODO should be client error - distinct type from CohereError?
            # ValidationError, e.g. when variant is bad
            raise CohereError(str(e))

        return summary


    def delete_endpoint(self) -> None:
        if self._endpoint_name is None:
            raise CohereError("No endpoint connected.")
        try:
            self._service_client.delete_endpoint(EndpointName=self._endpoint_name)
        except:
            print("Endpoint not found, skipping deletion.")

        try:
            self._service_client.delete_endpoint_config(EndpointConfigName=self._endpoint_name)
        except:
            print("Endpoint config not found, skipping deletion.")

    def close(self) -> None:
        try:
            self._client.close()
            self._service_client.close()
        except AttributeError:
            print("SageMaker client could not be closed. This might be because you are using an old version of SageMaker.")
            raise
