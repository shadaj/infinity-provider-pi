import type { Tool } from "@earendil-works/pi-ai";

export interface InfinityModelEntry {
  model_id: string;
  display_name: string;
  context_window: number;
  max_output_tokens: number | null;
}

export type InfinityProviderRequest =
  | "ListModels"
  | { InvokeModel: { model_id: string; request: InfinityCompletionRequest } };

export type InfinityProviderResponse =
  | { Models: InfinityModelEntry[] }
  | "InvokeStarted"
  | { Chunk: InfinityWireStreamItem }
  | "StreamEnd"
  | { Error: string };

// Mirrors infinity-agent-core::model_provider::remote::WireCompletionRequest,
// whose message/document/tool fields are rig-core serde shapes.
export interface InfinityCompletionRequest {
  model: string | null;
  preamble: string | null;
  chat_history: OneOrMany<InfinityMessage>;
  documents: InfinityDocument[];
  tools: InfinityToolDefinition[];
  temperature: number | null;
  max_tokens: number | null;
  tool_choice: unknown | null;
  additional_params: Record<string, unknown> | null;
  output_schema: unknown | null;
}

export type OneOrMany<T> = T[] | { One: T } | { Many: T[] } | { one: T } | { many: T[] } | T;

// rig::message::Message is internally tagged: #[serde(tag = "role", rename_all = "lowercase")].
export type InfinityMessage = InfinityUserMessage | InfinityAssistantMessage;

export interface InfinityUserMessage {
  role: "user";
  content: OneOrMany<InfinityUserContent>;
}

export interface InfinityAssistantMessage {
  role: "assistant";
  id: string | null;
  content: OneOrMany<InfinityAssistantContent>;
}

// rig::message::UserContent is internally tagged: #[serde(tag = "type", rename_all = "lowercase")].
export type InfinityUserContent =
  | ({ type: "text" } & InfinityTextContent)
  | ({ type: "toolresult" } & InfinityToolResult)
  | ({ type: "image" } & InfinityMediaContent)
  | ({ type: "audio" } & InfinityMediaContent)
  | ({ type: "video" } & InfinityMediaContent)
  | ({ type: "document" } & InfinityMediaContent);

// rig::message::AssistantContent is untagged. Text/ToolCall/Reasoning/Image are
// serialized as their inner struct fields, with no wrapper/discriminant.
export type InfinityAssistantContent =
  | InfinityTextContent
  | InfinityRawToolCall
  | InfinityReasoningBlock
  | InfinityMediaContent;

export interface InfinityTextContent {
  text: string;
}

export interface InfinityToolResult {
  id: string;
  call_id: string | null;
  content: OneOrMany<InfinityToolResultContent>;
}

// rig::message::ToolResultContent is internally tagged: #[serde(tag = "type", rename_all = "lowercase")].
export type InfinityToolResultContent =
  | ({ type: "text" } & InfinityTextContent)
  | ({ type: "image" } & InfinityMediaContent);

// rig::message::ToolCall inside untagged AssistantContent.
export interface InfinityRawToolCall {
  id: string;
  call_id: string | null;
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
  signature: string | null;
  additional_params: unknown | null;
}

// rig::message::Reasoning inside untagged AssistantContent.
export interface InfinityReasoningBlock {
  id: string | null;
  content: InfinityReasoningContent[];
}

// rig::message::ReasoningContent uses #[serde(tag = "type", content = "content", rename_all = "snake_case")].
export type InfinityReasoningContent =
  | { type: "text"; content: { text: string; signature?: string | null } }
  | { type: "encrypted"; content: string }
  | { type: "redacted"; content: { data: string } }
  | { type: "summary"; content: string };

export interface InfinityMediaContent {
  data: InfinityDocumentSourceKind;
  media_type?: string | null;
  detail?: unknown;
  additional_params?: unknown;
  [key: string]: unknown;
}

// rig::message::DocumentSourceKind uses #[serde(tag = "type", content = "value", rename_all = "camelCase")].
export type InfinityDocumentSourceKind =
  | { type: "url"; value: string }
  | { type: "base64"; value: string }
  | { type: "raw"; value: number[] }
  | { type: "string"; value: string }
  | { type: "unknown" };

// rig::completion::Document, not rig::message::Document.
export interface InfinityDocument {
  id: string;
  text: string;
  [additionalProp: string]: string;
}

export interface InfinityToolDefinition {
  name: string;
  description: string;
  parameters: Tool["parameters"];
}

export type InfinityWireStreamItem =
  | { Message: string }
  | { ToolCall: InfinityWireToolCall }
  | { ToolCallDelta: InfinityWireToolCallDelta }
  | { Reasoning: { id: string | null; content: InfinityReasoningContent } }
  | { ReasoningDelta: { id: string | null; reasoning: string } }
  | { Final: { usage: InfinityRigUsage | null } }
  | { Error: string };

export interface InfinityWireToolCall {
  id: string;
  internal_call_id: string;
  call_id: string | null;
  name: string;
  arguments: Record<string, unknown>;
  signature: string | null;
  additional_params: unknown | null;
}

export interface InfinityWireToolCallDelta {
  id: string;
  internal_call_id: string;
  content: InfinityToolCallDeltaContent;
}

// rig::streaming::ToolCallDeltaContent uses serde's default externally-tagged enum shape.
export type InfinityToolCallDeltaContent =
  | { Name: string }
  | { Delta: string };

export interface InfinityRigUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cached_input_tokens: number;
}
