# This file was auto-generated by Fern from our API Definition.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from .types import (
    ApiMeta,
    ApiMetaApiVersion,
    ApiMetaBilledUnits,
    ApiMetaTokens,
    AuthTokenType,
    ChatCitation,
    ChatCitationGenerationEvent,
    ChatConnector,
    ChatDataMetrics,
    ChatDocument,
    ChatMessage,
    ChatRequestCitationQuality,
    ChatRequestConnectorsSearchOptions,
    ChatRequestPromptTruncation,
    ChatRequestSafetyMode,
    ChatSearchQueriesGenerationEvent,
    ChatSearchQuery,
    ChatSearchResult,
    ChatSearchResultConnector,
    ChatSearchResultsEvent,
    ChatStreamEndEvent,
    ChatStreamEndEventFinishReason,
    ChatStreamEvent,
    ChatStreamRequestCitationQuality,
    ChatStreamRequestConnectorsSearchOptions,
    ChatStreamRequestPromptTruncation,
    ChatStreamRequestSafetyMode,
    ChatStreamStartEvent,
    ChatTextGenerationEvent,
    ChatToolCallsChunkEvent,
    ChatToolCallsGenerationEvent,
    CheckApiKeyResponse,
    CitationEndEvent,
    CitationStartEvent,
    CitationStartEventDelta,
    CitationStartEventDeltaMessage,
    ClassifyDataMetrics,
    ClassifyExample,
    ClassifyRequestTruncate,
    ClassifyResponse,
    ClassifyResponseClassificationsItem,
    ClassifyResponseClassificationsItemClassificationType,
    ClassifyResponseClassificationsItemLabelsValue,
    ClientClosedRequestErrorBody,
    CompatibleEndpoint,
    Connector,
    ConnectorAuthStatus,
    ConnectorOAuth,
    CreateConnectorOAuth,
    CreateConnectorResponse,
    CreateConnectorServiceAuth,
    CreateEmbedJobResponse,
    Dataset,
    DatasetPart,
    DatasetType,
    DatasetValidationStatus,
    DeleteConnectorResponse,
    DetokenizeResponse,
    EmbedByTypeResponse,
    EmbedByTypeResponseEmbeddings,
    EmbedFloatsResponse,
    EmbedInputType,
    EmbedJob,
    EmbedJobStatus,
    EmbedJobTruncate,
    EmbedRequestTruncate,
    EmbedResponse,
    EmbedResponse_EmbeddingsByType,
    EmbedResponse_EmbeddingsFloats,
    EmbeddingType,
    FinetuneDatasetMetrics,
    FinishReason,
    GatewayTimeoutErrorBody,
    GenerateRequestReturnLikelihoods,
    GenerateRequestTruncate,
    GenerateStreamEnd,
    GenerateStreamEndResponse,
    GenerateStreamError,
    GenerateStreamEvent,
    GenerateStreamRequestReturnLikelihoods,
    GenerateStreamRequestTruncate,
    GenerateStreamText,
    GenerateStreamedResponse,
    GenerateStreamedResponse_StreamEnd,
    GenerateStreamedResponse_StreamError,
    GenerateStreamedResponse_TextGeneration,
    Generation,
    GetConnectorResponse,
    GetModelResponse,
    JsonResponseFormat,
    JsonResponseFormat2,
    LabelMetric,
    ListConnectorsResponse,
    ListEmbedJobResponse,
    ListModelsResponse,
    Message,
    Message_Chatbot,
    Message_System,
    Message_Tool,
    Message_User,
    Metrics,
    MetricsEmbedData,
    MetricsEmbedDataFieldsItem,
    NonStreamedChatResponse,
    NotImplementedErrorBody,
    OAuthAuthorizeResponse,
    ParseInfo,
    RerankDocument,
    RerankRequestDocumentsItem,
    RerankResponse,
    RerankResponseResultsItem,
    RerankResponseResultsItemDocument,
    RerankerDataMetrics,
    ResponseFormat,
    ResponseFormat2,
    ResponseFormat2_JsonObject,
    ResponseFormat2_Text,
    ResponseFormat_JsonObject,
    ResponseFormat_Text,
    SingleGeneration,
    SingleGenerationInStream,
    SingleGenerationTokenLikelihoodsItem,
    StreamedChatResponse,
    StreamedChatResponse_CitationGeneration,
    StreamedChatResponse_SearchQueriesGeneration,
    StreamedChatResponse_SearchResults,
    StreamedChatResponse_StreamEnd,
    StreamedChatResponse_StreamStart,
    StreamedChatResponse_TextGeneration,
    StreamedChatResponse_ToolCallsChunk,
    StreamedChatResponse_ToolCallsGeneration,
    SummarizeRequestExtractiveness,
    SummarizeRequestFormat,
    SummarizeRequestLength,
    SummarizeResponse,
    TextResponseFormat,
    TokenizeResponse,
    TooManyRequestsErrorBody,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolMessage,
    ToolParameterDefinitionsValue,
    ToolResult,
    UnprocessableEntityErrorBody,
    UpdateConnectorResponse,
)
from .errors import (
    BadRequestError,
    ClientClosedRequestError,
    ForbiddenError,
    GatewayTimeoutError,
    InternalServerError,
    NotFoundError,
    NotImplementedError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
    UnprocessableEntityError,
)
from . import connectors, datasets, embed_jobs, finetuning, models, v2
from .aws_client import AwsClient
from .client_v2 import AsyncClientV2, ClientV2
from .bedrock_client import BedrockClient
from .client import AsyncClient, Client
from .datasets import (
    DatasetsCreateResponse,
    DatasetsCreateResponseDatasetPartsItem,
    DatasetsGetResponse,
    DatasetsGetUsageResponse,
    DatasetsListResponse,
)
from .embed_jobs import CreateEmbedJobRequestTruncate
from .environment import ClientEnvironment
from .sagemaker_client import SagemakerClient
from .v2 import (
    AssistantMessage,
    AssistantMessageContent,
    AssistantMessageContentItem,
    AssistantMessageContentItem_Text,
    AssistantMessageResponse,
    AssistantMessageResponseContentItem,
    AssistantMessageResponseContentItem_Text,
    ChatContentDeltaEvent,
    ChatContentDeltaEventDelta,
    ChatContentDeltaEventDeltaMessage,
    ChatContentDeltaEventDeltaMessageContent,
    ChatContentEndEvent,
    ChatContentStartEvent,
    ChatContentStartEventDelta,
    ChatContentStartEventDeltaMessage,
    ChatContentStartEventDeltaMessageContent,
    ChatFinishReason,
    ChatMessage2,
    ChatMessage2_Assistant,
    ChatMessage2_System,
    ChatMessage2_Tool,
    ChatMessage2_User,
    ChatMessageEndEvent,
    ChatMessageEndEventDelta,
    ChatMessageStartEvent,
    ChatMessageStartEventDelta,
    ChatMessageStartEventDeltaMessage,
    ChatMessages,
    ChatStreamEventType,
    ChatToolCallDeltaEvent,
    ChatToolCallDeltaEventDelta,
    ChatToolCallDeltaEventDeltaToolCall,
    ChatToolCallDeltaEventDeltaToolCallFunction,
    ChatToolCallEndEvent,
    ChatToolCallStartEvent,
    ChatToolCallStartEventDelta,
    ChatToolCallStartEventDeltaToolCall,
    ChatToolCallStartEventDeltaToolCallFunction,
    ChatToolPlanDeltaEvent,
    ChatToolPlanDeltaEventDelta,
    Citation,
    Content,
    Content_Text,
    DocumentSource,
    NonStreamedChatResponse2,
    Source,
    Source_Document,
    Source_Tool,
    StreamedChatResponse2,
    StreamedChatResponse2_CitationEnd,
    StreamedChatResponse2_CitationStart,
    StreamedChatResponse2_ContentDelta,
    StreamedChatResponse2_ContentEnd,
    StreamedChatResponse2_ContentStart,
    StreamedChatResponse2_MessageEnd,
    StreamedChatResponse2_MessageStart,
    StreamedChatResponse2_ToolCallDelta,
    StreamedChatResponse2_ToolCallEnd,
    StreamedChatResponse2_ToolCallStart,
    StreamedChatResponse2_ToolPlanDelta,
    SystemMessage,
    SystemMessageContent,
    SystemMessageContentItem,
    SystemMessageContentItem_Text,
    TextContent,
    Tool2,
    Tool2Function,
    ToolCall2,
    ToolCall2Function,
    ToolContent,
    ToolMessage2,
    ToolMessage2ToolContentItem,
    ToolMessage2ToolContentItem_ToolResultObject,
    ToolSource,
    Usage,
    UsageBilledUnits,
    UsageTokens,
    UserMessage,
    UserMessageContent,
    V2ChatRequestCitationMode,
    V2ChatStreamRequestCitationMode,
)
from .version import __version__

__all__ = [
    "ApiMeta",
    "ApiMetaApiVersion",
    "ApiMetaBilledUnits",
    "ApiMetaTokens",
    "AssistantMessage",
    "AssistantMessageContent",
    "AssistantMessageContentItem",
    "AssistantMessageContentItem_Text",
    "AssistantMessageResponse",
    "AssistantMessageResponseContentItem",
    "AssistantMessageResponseContentItem_Text",
    "AsyncClient",
    "AuthTokenType",
    "AwsClient",
    "BadRequestError",
    "BedrockClient",
    "ChatCitation",
    "ChatCitationGenerationEvent",
    "ChatConnector",
    "ChatContentDeltaEvent",
    "ChatContentDeltaEventDelta",
    "ChatContentDeltaEventDeltaMessage",
    "ChatContentDeltaEventDeltaMessageContent",
    "ChatContentEndEvent",
    "ChatContentStartEvent",
    "ChatContentStartEventDelta",
    "ChatContentStartEventDeltaMessage",
    "ChatContentStartEventDeltaMessageContent",
    "ChatDataMetrics",
    "ChatDocument",
    "ChatFinishReason",
    "ChatMessage",
    "ChatMessage2",
    "ChatMessage2_Assistant",
    "ChatMessage2_System",
    "ChatMessage2_Tool",
    "ChatMessage2_User",
    "ChatMessageEndEvent",
    "ChatMessageEndEventDelta",
    "ChatMessageStartEvent",
    "ChatMessageStartEventDelta",
    "ChatMessageStartEventDeltaMessage",
    "ChatMessages",
    "ChatRequestCitationQuality",
    "ChatRequestConnectorsSearchOptions",
    "ChatRequestPromptTruncation",
    "ChatRequestSafetyMode",
    "ChatSearchQueriesGenerationEvent",
    "ChatSearchQuery",
    "ChatSearchResult",
    "ChatSearchResultConnector",
    "ChatSearchResultsEvent",
    "ChatStreamEndEvent",
    "ChatStreamEndEventFinishReason",
    "ChatStreamEvent",
    "ChatStreamEventType",
    "ChatStreamRequestCitationQuality",
    "ChatStreamRequestConnectorsSearchOptions",
    "ChatStreamRequestPromptTruncation",
    "ChatStreamRequestSafetyMode",
    "ChatStreamStartEvent",
    "ChatTextGenerationEvent",
    "ChatToolCallDeltaEvent",
    "ChatToolCallDeltaEventDelta",
    "ChatToolCallDeltaEventDeltaToolCall",
    "ChatToolCallDeltaEventDeltaToolCallFunction",
    "ChatToolCallEndEvent",
    "ChatToolCallStartEvent",
    "ChatToolCallStartEventDelta",
    "ChatToolCallStartEventDeltaToolCall",
    "ChatToolCallStartEventDeltaToolCallFunction",
    "ChatToolCallsChunkEvent",
    "ChatToolCallsGenerationEvent",
    "ChatToolPlanDeltaEvent",
    "ChatToolPlanDeltaEventDelta",
    "CheckApiKeyResponse",
    "Citation",
    "CitationEndEvent",
    "CitationStartEvent",
    "CitationStartEventDelta",
    "CitationStartEventDeltaMessage",
    "ClassifyDataMetrics",
    "ClassifyExample",
    "ClassifyRequestTruncate",
    "ClassifyResponse",
    "ClassifyResponseClassificationsItem",
    "ClassifyResponseClassificationsItemClassificationType",
    "ClassifyResponseClassificationsItemLabelsValue",
    "Client",
    "ClientClosedRequestError",
    "ClientClosedRequestErrorBody",
    "ClientEnvironment",
    "CompatibleEndpoint",
    "Connector",
    "ConnectorAuthStatus",
    "ConnectorOAuth",
    "Content",
    "Content_Text",
    "CreateConnectorOAuth",
    "CreateConnectorResponse",
    "CreateConnectorServiceAuth",
    "CreateEmbedJobRequestTruncate",
    "CreateEmbedJobResponse",
    "Dataset",
    "DatasetPart",
    "DatasetType",
    "DatasetValidationStatus",
    "DatasetsCreateResponse",
    "DatasetsCreateResponseDatasetPartsItem",
    "DatasetsGetResponse",
    "DatasetsGetUsageResponse",
    "DatasetsListResponse",
    "DeleteConnectorResponse",
    "DetokenizeResponse",
    "DocumentSource",
    "EmbedByTypeResponse",
    "EmbedByTypeResponseEmbeddings",
    "EmbedFloatsResponse",
    "EmbedInputType",
    "EmbedJob",
    "EmbedJobStatus",
    "EmbedJobTruncate",
    "EmbedRequestTruncate",
    "EmbedResponse",
    "EmbedResponse_EmbeddingsByType",
    "EmbedResponse_EmbeddingsFloats",
    "EmbeddingType",
    "FinetuneDatasetMetrics",
    "FinishReason",
    "ForbiddenError",
    "GatewayTimeoutError",
    "GatewayTimeoutErrorBody",
    "GenerateRequestReturnLikelihoods",
    "GenerateRequestTruncate",
    "GenerateStreamEnd",
    "GenerateStreamEndResponse",
    "GenerateStreamError",
    "GenerateStreamEvent",
    "GenerateStreamRequestReturnLikelihoods",
    "GenerateStreamRequestTruncate",
    "GenerateStreamText",
    "GenerateStreamedResponse",
    "GenerateStreamedResponse_StreamEnd",
    "GenerateStreamedResponse_StreamError",
    "GenerateStreamedResponse_TextGeneration",
    "Generation",
    "GetConnectorResponse",
    "GetModelResponse",
    "InternalServerError",
    "JsonResponseFormat",
    "JsonResponseFormat2",
    "LabelMetric",
    "ListConnectorsResponse",
    "ListEmbedJobResponse",
    "ListModelsResponse",
    "Message",
    "Message_Chatbot",
    "Message_System",
    "Message_Tool",
    "Message_User",
    "Metrics",
    "MetricsEmbedData",
    "MetricsEmbedDataFieldsItem",
    "NonStreamedChatResponse",
    "NonStreamedChatResponse2",
    "NotFoundError",
    "NotImplementedError",
    "NotImplementedErrorBody",
    "OAuthAuthorizeResponse",
    "ParseInfo",
    "RerankDocument",
    "RerankRequestDocumentsItem",
    "RerankResponse",
    "RerankResponseResultsItem",
    "RerankResponseResultsItemDocument",
    "RerankerDataMetrics",
    "ResponseFormat",
    "ResponseFormat2",
    "ResponseFormat2_JsonObject",
    "ResponseFormat2_Text",
    "ResponseFormat_JsonObject",
    "ResponseFormat_Text",
    "SagemakerClient",
    "ServiceUnavailableError",
    "SingleGeneration",
    "SingleGenerationInStream",
    "SingleGenerationTokenLikelihoodsItem",
    "Source",
    "Source_Document",
    "Source_Tool",
    "StreamedChatResponse",
    "StreamedChatResponse2",
    "StreamedChatResponse2_CitationEnd",
    "StreamedChatResponse2_CitationStart",
    "StreamedChatResponse2_ContentDelta",
    "StreamedChatResponse2_ContentEnd",
    "StreamedChatResponse2_ContentStart",
    "StreamedChatResponse2_MessageEnd",
    "StreamedChatResponse2_MessageStart",
    "StreamedChatResponse2_ToolCallDelta",
    "StreamedChatResponse2_ToolCallEnd",
    "StreamedChatResponse2_ToolCallStart",
    "StreamedChatResponse2_ToolPlanDelta",
    "StreamedChatResponse_CitationGeneration",
    "StreamedChatResponse_SearchQueriesGeneration",
    "StreamedChatResponse_SearchResults",
    "StreamedChatResponse_StreamEnd",
    "StreamedChatResponse_StreamStart",
    "StreamedChatResponse_TextGeneration",
    "StreamedChatResponse_ToolCallsChunk",
    "StreamedChatResponse_ToolCallsGeneration",
    "SummarizeRequestExtractiveness",
    "SummarizeRequestFormat",
    "SummarizeRequestLength",
    "SummarizeResponse",
    "SystemMessage",
    "SystemMessageContent",
    "SystemMessageContentItem",
    "SystemMessageContentItem_Text",
    "TextContent",
    "TextResponseFormat",
    "TokenizeResponse",
    "TooManyRequestsError",
    "TooManyRequestsErrorBody",
    "Tool",
    "Tool2",
    "Tool2Function",
    "ToolCall",
    "ToolCall2",
    "ToolCall2Function",
    "ToolCallDelta",
    "ToolContent",
    "ToolMessage",
    "ToolMessage2",
    "ToolMessage2ToolContentItem",
    "ToolMessage2ToolContentItem_ToolResultObject",
    "ToolParameterDefinitionsValue",
    "ToolResult",
    "ToolSource",
    "UnauthorizedError",
    "UnprocessableEntityError",
    "UnprocessableEntityErrorBody",
    "UpdateConnectorResponse",
    "Usage",
    "UsageBilledUnits",
    "UsageTokens",
    "UserMessage",
    "UserMessageContent",
    "V2ChatRequestCitationMode",
    "V2ChatStreamRequestCitationMode",
    "__version__",
    "connectors",
    "datasets",
    "embed_jobs",
    "finetuning",
    "models",
    "v2",
]

warnings.warn("Test 1")
warnings.resetwarnings()
warnings.warn("Test 2")
