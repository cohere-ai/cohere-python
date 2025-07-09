from .v2 import (
    ContentDeltaV2ChatStreamResponse,
    ContentEndV2ChatStreamResponse,
    ContentStartV2ChatStreamResponse,
    MessageEndV2ChatStreamResponse,
    MessageStartV2ChatStreamResponse,
    ToolCallDeltaV2ChatStreamResponse,
    ToolCallEndV2ChatStreamResponse,
    ToolCallStartV2ChatStreamResponse,
    V2ChatStreamResponse,
    V2ChatResponse
)

# alias classes
StreamedChatResponseV2 = V2ChatStreamResponse
MessageStartStreamedChatResponseV2 = MessageStartV2ChatStreamResponse
MessageEndStreamedChatResponseV2 = MessageEndV2ChatStreamResponse
ContentStartStreamedChatResponseV2 = ContentStartV2ChatStreamResponse
ContentDeltaStreamedChatResponseV2 = ContentDeltaV2ChatStreamResponse
ContentEndStreamedChatResponseV2 = ContentEndV2ChatStreamResponse
ToolCallStartStreamedChatResponseV2 = ToolCallStartV2ChatStreamResponse
ToolCallDeltaStreamedChatResponseV2 = ToolCallDeltaV2ChatStreamResponse
ToolCallEndStreamedChatResponseV2 = ToolCallEndV2ChatStreamResponse
ChatResponse = V2ChatResponse
