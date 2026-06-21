#!/usr/bin/env node
import net from "node:net";
import os from "node:os";
import path from "node:path";
import fs from "node:fs/promises";
import process from "node:process";
import { getApiProvider } from "@earendil-works/pi-ai";
import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  AssistantMessageEventStream,
  Context,
  SimpleStreamOptions,
  ImageContent,
  Message as PiMessage,
  Model,
  TextContent,
  ThinkingContent,
  ThinkingLevel,
  Tool,
  ToolCall as PiToolCall,
  Usage,
} from "@earendil-works/pi-ai";
import { createAgentSessionServices } from "@earendil-works/pi-coding-agent";
import type { AgentSessionServices } from "@earendil-works/pi-coding-agent";
import type {
  InfinityAssistantContent,
  InfinityCompletionRequest,
  InfinityMediaContent,
  InfinityMessage,
  InfinityModelEntry,
  InfinityProviderRequest,
  InfinityProviderResponse,
  InfinityRawToolCall,
  InfinityReasoningBlock,
  InfinityRigUsage,
  InfinityToolDefinition,
  InfinityToolResult,
  InfinityToolResultContent,
  InfinityUserContent,
  InfinityWireStreamItem,
  OneOrMany,
} from "./infinity-types.js";

type WriteLine = (value: InfinityProviderResponse) => void;

const socketPath = path.join(
  os.tmpdir(),
  `inf-provider-pi-${process.pid}-${Math.random().toString(16).slice(2)}.sock`,
);

let servicesPromise: Promise<AgentSessionServices> | undefined;
function getServices(): Promise<AgentSessionServices> {
  servicesPromise ??= createAgentSessionServices({ cwd: process.cwd() }).then(
    (services) => {
      for (const diagnostic of services.diagnostics ?? []) {
        const level = diagnostic.type === "error" ? "error" : diagnostic.type;
        console.error(`[pi ${level}] ${diagnostic.message}`);
      }
      return services;
    },
  );
  return servicesPromise;
}

const providerResponse = {
  models: (models: InfinityModelEntry[]): InfinityProviderResponse => ({
    Models: models,
  }),
  invokeStarted: (): InfinityProviderResponse => "InvokeStarted",
  chunk: (item: InfinityWireStreamItem): InfinityProviderResponse => ({
    Chunk: item,
  }),
  streamEnd: (): InfinityProviderResponse => "StreamEnd",
  error: (message: unknown): InfinityProviderResponse => ({
    Error: String(message),
  }),
};

const wireItem = {
  message: (text: string): InfinityWireStreamItem => ({ Message: text }),
  reasoningDelta: (
    reasoning: string,
    id: string | null = null,
  ): InfinityWireStreamItem => ({ ReasoningDelta: { id, reasoning } }),
  toolCall: ({
    id,
    name,
    arguments: args,
    thoughtSignature,
  }: PiToolCall): InfinityWireStreamItem => ({
    ToolCall: {
      id,
      internal_call_id: id,
      call_id: null,
      name,
      arguments: args,
      signature: thoughtSignature ?? null,
      additional_params: null,
    },
  }),
  toolCallNameDelta: ({ id, name }: PiToolCall): InfinityWireStreamItem => ({
    ToolCallDelta: {
      id,
      internal_call_id: id,
      content: { Name: name },
    },
  }),
  toolCallArgumentDelta: (
    { id }: PiToolCall,
    delta: string,
  ): InfinityWireStreamItem => ({
    ToolCallDelta: {
      id,
      internal_call_id: id,
      content: { Delta: delta },
    },
  }),
  final: (usage?: Usage | null): InfinityWireStreamItem => ({
    Final: { usage: usage ? toRigUsage(usage) : null },
  }),
  error: (message: unknown): InfinityWireStreamItem => ({
    Error: String(message),
  }),
};

function toRigUsage(usage: Usage): InfinityRigUsage {
  // pi-ai keeps billable uncached input separate from cache reads/writes,
  // while rig's `input_tokens` represents the full prompt/context size.
  // Infinity's live context meter is based on `input_tokens`, so report the
  // full prompt tokens here instead of only the uncached portion.
  const cachedInputTokens = usage.cacheRead;
  const promptCacheTokens = usage.cacheRead + usage.cacheWrite;
  const componentTotalTokens = usage.input + usage.output + promptCacheTokens;
  const totalTokens = usage.totalTokens || componentTotalTokens;
  const inputTokens = usage.input + promptCacheTokens;

  return {
    input_tokens: inputTokens,
    output_tokens: usage.output,
    total_tokens: totalTokens,
    cached_input_tokens: cachedInputTokens,
  };
}

function modelKey(model: Model<Api>): string {
  return `${model.provider}/${model.id}`;
}

function toModelEntry(model: Model<Api>): InfinityModelEntry {
  return {
    model_id: modelKey(model),
    display_name: `${model.name} (${model.provider})`,
    context_window: model.contextWindow,
    max_output_tokens: model.maxTokens,
  };
}

function findModel(
  models: Model<Api>[],
  modelId: string,
): Model<Api> | undefined {
  let model = models.find((m) => modelKey(m) === modelId);
  if (model) return model;
  // Convenience fallback for providers whose model ids are unique.
  const byBareId = models.filter((m) => m.id === modelId);
  return byBareId.length === 1 ? byBareId[0] : undefined;
}

async function listModels(): Promise<InfinityModelEntry[]> {
  const { modelRegistry } = await getServices();
  modelRegistry.refresh();
  const includeUnavailable =
    process.env.PI_INFINITY_INCLUDE_UNAVAILABLE === "1";
  const models = includeUnavailable
    ? modelRegistry.getAll()
    : modelRegistry.getAvailable();
  return models.map(toModelEntry);
}

async function invokeModel(
  modelId: string,
  request: InfinityCompletionRequest,
  writeLine: WriteLine,
): Promise<void> {
  const { modelRegistry } = await getServices();
  modelRegistry.refresh();
  const model = findModel(modelRegistry.getAll(), modelId);
  if (!model) {
    writeLine(providerResponse.error(`pi model not found: ${modelId}`));
    return;
  }

  const apiProvider = getApiProvider(model.api);
  if (!apiProvider?.streamSimple) {
    writeLine(
      providerResponse.error(
        `pi API provider not available for api '${model.api}'`,
      ),
    );
    return;
  }

  const auth = await modelRegistry.getApiKeyAndHeaders(model);
  if (!auth.ok) {
    writeLine(providerResponse.error(auth.error));
    return;
  }

  const context = wireCompletionRequestToPiContext(request);
  const options: SimpleStreamOptions = {};
  if (request.temperature !== null) options.temperature = request.temperature;
  if (request.max_tokens !== null) options.maxTokens = request.max_tokens;
  if (auth.apiKey !== undefined) options.apiKey = auth.apiKey;
  if (auth.headers !== undefined) options.headers = auth.headers;
  const reasoning = getReasoningLevel(model, request);
  if (reasoning !== undefined) options.reasoning = reasoning;

  let stream: AssistantMessageEventStream;
  try {
    stream = apiProvider.streamSimple(model, context, options);
  } catch (error) {
    writeLine(providerResponse.error(errorMessage(error)));
    return;
  }

  writeLine(providerResponse.invokeStarted());
  const eventTranslationState = createEventTranslationState();
  try {
    for await (const event of stream) {
      for (const item of eventToWireItems(event, eventTranslationState)) {
        writeLine(providerResponse.chunk(item));
      }
    }
  } catch (error) {
    writeLine(providerResponse.chunk(wireItem.error(errorMessage(error))));
  }
  writeLine(providerResponse.streamEnd());
}

interface EventTranslationState {
  toolCallNamesSentByContentIndex: Map<number, string>;
}

function createEventTranslationState(): EventTranslationState {
  return { toolCallNamesSentByContentIndex: new Map() };
}

function eventToWireItems(
  event: AssistantMessageEvent,
  state: EventTranslationState,
): InfinityWireStreamItem[] {
  switch (event.type) {
    case "text_delta":
      return [wireItem.message(event.delta)];
    case "thinking_delta":
      return [wireItem.reasoningDelta(event.delta)];
    case "thinking_end":
      if (event.partial.content[event.contentIndex]!.type !== "thinking") {
        throw new Error(
          `pi assistant message event ${JSON.stringify(event)} content did not end with expected thinking content`,
        );
      }

      return [
        {
          Reasoning: {
            id: null,
            content: {
              type: "text",
              content: {
                text: event.content,
                signature:
                  (event.partial.content[event.contentIndex] as ThinkingContent)
                    .thinkingSignature ?? null,
              },
            },
          },
        },
      ];
    case "toolcall_start":
      return maybeToolCallNameDelta(event, state);
    case "toolcall_delta": {
      const toolCall = partialToolCall(event);
      const items = maybeToolCallNameDelta(event, state);
      if (event.delta.length > 0) {
        items.push(wireItem.toolCallArgumentDelta(toolCall, event.delta));
      }
      return items;
    }
    case "toolcall_end":
      return [wireItem.toolCall(event.toolCall)];
    case "done":
      return [wireItem.final(event.message.usage)];
    case "error":
      if (!event.error.errorMessage)
        throw new Error("pi error event is missing errorMessage");
      return [wireItem.error(event.error.errorMessage)];
    default:
      return [];
  }
}

function partialToolCall(event: {
  type: string;
  contentIndex: number;
  partial: AssistantMessage;
}): PiToolCall {
  const block = event.partial.content[event.contentIndex];
  if (!block || block.type !== "toolCall") {
    throw new Error(
      `pi assistant message event ${JSON.stringify(event)} content did not reference expected tool call content`,
    );
  }
  return block;
}

function maybeToolCallNameDelta(
  event: { type: string; contentIndex: number; partial: AssistantMessage },
  state: EventTranslationState,
): InfinityWireStreamItem[] {
  const toolCall = partialToolCall(event);
  if (toolCall.name.length === 0) return [];

  const previousName = state.toolCallNamesSentByContentIndex.get(
    event.contentIndex,
  );
  if (previousName === toolCall.name) return [];

  state.toolCallNamesSentByContentIndex.set(event.contentIndex, toolCall.name);
  return [wireItem.toolCallNameDelta(toolCall)];
}

function getReasoningLevel(
  model: Model<Api>,
  request: InfinityCompletionRequest,
): ThinkingLevel | undefined {
  if (!model.reasoning) return undefined;

  const requested =
    request.additional_params?.reasoning ??
    request.additional_params?.thinkingLevel ??
    request.additional_params?.thinking_level ??
    process.env.PI_INFINITY_THINKING_LEVEL ??
    "high";

  if (requested === false || requested === "off" || requested === "none")
    return undefined;
  if (isThinkingLevel(requested)) return requested;

  console.error(
    `[pi warning] ignoring unsupported thinking level '${String(requested)}' for ${modelKey(model)}`,
  );
  return undefined;
}

function isThinkingLevel(value: unknown): value is ThinkingLevel {
  return (
    typeof value === "string" &&
    ["minimal", "low", "medium", "high", "xhigh"].includes(value)
  );
}

function wireCompletionRequestToPiContext(
  request: InfinityCompletionRequest,
): Context {
  const messages: PiMessage[] = [];
  const toolCallNames = new Map<string, string>();
  for (const message of asMany(request.chat_history)) {
    const converted = wireMessageToPiMessage(message, toolCallNames);
    if (converted) messages.push(converted);
  }
  if (request.documents.length > 0) {
    throw new Error(
      "Infinity request documents are not yet supported by this provider",
    );
  }
  const context: Context = {
    messages,
    tools: request.tools.map(wireToolToPiTool),
  };
  if (request.preamble !== null) context.systemPrompt = request.preamble;
  return context;
}

function wireMessageToPiMessage(
  message: InfinityMessage,
  toolCallNames: Map<string, string>,
): PiMessage {
  if (message.role === "user") {
    const parts = asMany(message.content);
    if (parts.length === 0)
      throw new Error("Infinity user message content must not be empty");

    const toolResults = parts.filter((part) => part.type === "toolresult");
    if (toolResults.length > 0) {
      if (parts.length !== 1) {
        throw new Error(
          "Infinity tool result content cannot be mixed with other user content in one pi message",
        );
      }
      const toolResult = toolResults[0];
      if (!toolResult)
        throw new Error(
          "internal error: expected one Infinity tool result content part",
        );
      return toolResultToPiMessage(toolResult, toolCallNames);
    }

    return {
      role: "user",
      content: parts.map(userNonToolContentToPi),
      timestamp: Date.now(),
    };
  }

  const content = assistantContentToPi(message.content);
  for (const block of content) {
    if (block.type === "toolCall") toolCallNames.set(block.id, block.name);
  }
  return {
    role: "assistant",
    content,
    api: "unknown",
    provider: "infinity",
    model: "history",
    usage: emptyUsage(),
    stopReason: content.some((block) => block.type === "toolCall")
      ? "toolUse"
      : "stop",
    timestamp: Date.now(),
  };
}

function assistantContentToPi(
  content: OneOrMany<InfinityAssistantContent>,
): AssistantMessage["content"] {
  const blocks = asMany(content).map((part) => {
    if (isRawToolCall(part)) return toolCallToPi(part);
    if (isRawReasoning(part)) return reasoningToPi(part);
    if (isMediaContent(part))
      throw new Error(
        "Infinity assistant image/media content is not supported by pi assistant history",
      );
    return { type: "text" as const, text: part.text };
  });
  if (blocks.length === 0)
    throw new Error("Infinity assistant message content must not be empty");
  return blocks;
}

function userNonToolContentToPi(
  part: InfinityUserContent,
): TextContent | ImageContent {
  switch (part.type) {
    case "text":
      return { type: "text", text: part.text };
    case "image":
      return mediaContentToPiImage(part);
    case "audio":
    case "video":
      throw new Error(
        `Infinity user ${part.type} content is not supported by pi`,
      );
    case "document":
      return { type: "text", text: documentContentToPiText(part) };
    case "toolresult":
      throw new Error(
        "internal error: tool result reached non-tool user content conversion",
      );
  }
}

function toolCallToPi(value: InfinityRawToolCall): PiToolCall {
  const toolCall: PiToolCall = {
    type: "toolCall",
    id: value.id,
    name: value.function.name,
    arguments: value.function.arguments,
  };
  if (value.signature !== null) toolCall.thoughtSignature = value.signature;
  return toolCall;
}

function reasoningToPi(value: InfinityReasoningBlock): ThinkingContent {
  if (value.content.length !== 1) {
    throw new Error(
      `Infinity reasoning block has ${value.content.length} content items; pi history supports exactly one thinking text item`,
    );
  }
  const [block] = value.content;
  if (!block || block.type !== "text") {
    throw new Error(
      `Infinity reasoning content '${block?.type ?? "missing"}' cannot be represented losslessly in pi history`,
    );
  }
  const thinking: ThinkingContent = {
    type: "thinking",
    thinking: block.content.text,
  };
  if (
    block.content.signature !== undefined &&
    block.content.signature !== null
  ) {
    thinking.thinkingSignature = block.content.signature;
  }
  return thinking;
}

function toolResultToPiMessage(
  result: InfinityToolResult,
  toolCallNames: Map<string, string>,
): PiMessage {
  const toolName = toolCallNames.get(result.id);
  if (!toolName)
    throw new Error(
      `Infinity tool result '${result.id}' does not match a prior assistant tool call`,
    );
  return {
    role: "toolResult",
    toolCallId: result.id,
    toolName,
    content: toolResultContentToPi(result.content),
    isError: false,
    timestamp: Date.now(),
  };
}

function toolResultContentToPi(content: OneOrMany<InfinityToolResultContent>) {
  const parts = asMany(content).map((part) => {
    switch (part.type) {
      case "text":
        return { type: "text" as const, text: part.text };
      case "image":
        return mediaContentToPiImage(part);
    }
  });
  if (parts.length === 0)
    throw new Error("Infinity tool result content must not be empty");
  return parts;
}

function mediaContentToPiImage(value: InfinityMediaContent) {
  if (
    value.data.type !== "base64" &&
    value.data.type !== "url" &&
    value.data.type !== "string"
  ) {
    throw new Error(
      `Infinity image content source '${value.data.type}' is not supported by pi`,
    );
  }
  if (!value.media_type)
    throw new Error("Infinity image content is missing media_type");
  return {
    type: "image" as const,
    data: value.data.value,
    mimeType: `image/${value.media_type}`,
  };
}

function documentContentToPiText(value: InfinityMediaContent): string {
  if (value.data.type !== "string") {
    throw new Error(
      `Infinity document source '${value.data.type}' cannot be represented losslessly as pi user text`,
    );
  }
  return value.data.value;
}

function isRawToolCall(
  value: InfinityAssistantContent,
): value is InfinityRawToolCall {
  return (
    "function" in value &&
    typeof value.function === "object" &&
    value.function !== null &&
    "name" in value.function &&
    typeof value.function.name === "string"
  );
}

function isRawReasoning(
  value: InfinityAssistantContent,
): value is InfinityReasoningBlock {
  return "content" in value && Array.isArray(value.content);
}

function isMediaContent(
  value: InfinityAssistantContent,
): value is InfinityMediaContent {
  return "data" in value;
}

function wireToolToPiTool(tool: InfinityToolDefinition): Tool {
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
  };
}

function asMany<T>(value: OneOrMany<T> | null | undefined): T[] {
  if (value == null) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === "object") {
    if ("Many" in value && Array.isArray(value.Many)) return value.Many;
    if ("many" in value && Array.isArray(value.many)) return value.many;
    if ("One" in value) return [value.One];
    if ("one" in value) return [value.one];
  }
  return [value as T];
}

function emptyUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function parseRequest(line: string): InfinityProviderRequest {
  const parsed = JSON.parse(line) as InfinityProviderRequest;
  if (parsed === "ListModels") return parsed;
  if (typeof parsed === "object" && parsed !== null && "InvokeModel" in parsed)
    return parsed;
  throw new Error(`unknown provider request: ${line}`);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function handleConnection(socket: net.Socket): void {
  let buffer = "";
  let handled = false;
  const writeLine: WriteLine = (value) => {
    socket.write(`${JSON.stringify(value)}\n`);
  };

  socket.setEncoding("utf8");
  socket.on("data", async (chunk: string) => {
    if (handled) return;
    buffer += chunk;
    const newline = buffer.indexOf("\n");
    if (newline === -1) return;
    handled = true;

    try {
      const request = parseRequest(buffer.slice(0, newline));
      if (request === "ListModels") {
        writeLine(providerResponse.models(await listModels()));
      } else {
        await invokeModel(
          request.InvokeModel.model_id,
          request.InvokeModel.request,
          writeLine,
        );
      }
    } catch (error) {
      writeLine(providerResponse.error(errorMessage(error)));
    } finally {
      socket.end();
    }
  });
  socket.on("error", (error) =>
    console.error(`[socket error] ${errorMessage(error)}`),
  );
}

await fs.rm(socketPath, { force: true });
const server = net.createServer(handleConnection);
server.on("error", (error: Error) => {
  console.error(`[server error] ${errorMessage(error)}`);
  process.exitCode = 1;
});

await new Promise<void>((resolve, reject) =>
  server.listen(socketPath, () => resolve()).once("error", reject),
);

// Infinity provider stdout contract: first line is the Unix socket path.
console.log(socketPath);

async function cleanup() {
  server.close();
  await fs.rm(socketPath, { force: true }).catch(() => {});
}
process.on("SIGINT", async () => {
  await cleanup();
  process.exit(130);
});
process.on("SIGTERM", async () => {
  await cleanup();
  process.exit(143);
});
process.on("exit", () => {
  try {
    server.close();
  } catch {}
});
