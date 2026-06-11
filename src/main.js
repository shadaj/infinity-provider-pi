#!/usr/bin/env node
import net from "node:net";
import os from "node:os";
import path from "node:path";
import fs from "node:fs/promises";
import process from "node:process";
import { randomUUID } from "node:crypto";

import { getApiProvider } from "@earendil-works/pi-ai";
import { createAgentSessionServices } from "@earendil-works/pi-coding-agent";

const socketPath = path.join(
  os.tmpdir(),
  `inf-provider-pi-${process.pid}-${Math.random().toString(16).slice(2)}.sock`,
);

let servicesPromise;
function getServices() {
  servicesPromise ??= createAgentSessionServices({ cwd: process.cwd() }).then((services) => {
    for (const diagnostic of services.diagnostics ?? []) {
      const level = diagnostic.type === "error" ? "error" : diagnostic.type;
      console.error(`[pi ${level}] ${diagnostic.message}`);
    }
    return services;
  });
  return servicesPromise;
}

const providerResponse = {
  models: (models) => ({ Models: models }),
  invokeStarted: () => "InvokeStarted",
  chunk: (item) => ({ Chunk: item }),
  streamEnd: () => "StreamEnd",
  error: (message) => ({ Error: String(message) }),
};

const wireItem = {
  message: (text) => ({ Message: text }),
  reasoningDelta: (reasoning, id = null) => ({ ReasoningDelta: { id, reasoning } }),
  reasoningSummary: (summary, id = null) => ({ Reasoning: { id, content: { type: "summary", content: summary } } }),
  toolCall: ({ id, name, arguments: args, thoughtSignature }) => ({
    ToolCall: {
      id,
      internal_call_id: id,
      call_id: null,
      name,
      arguments: args ?? {},
      signature: thoughtSignature ?? null,
      additional_params: null,
    },
  }),
  final: (usage) => ({ Final: { usage: usage ? toRigUsage(usage) : null } }),
  error: (message) => ({ Error: String(message) }),
};

function toRigUsage(usage) {
  return {
    input_tokens: usage.input ?? 0,
    output_tokens: usage.output ?? 0,
    total_tokens: usage.totalTokens ?? (usage.input ?? 0) + (usage.output ?? 0) + (usage.cacheRead ?? 0) + (usage.cacheWrite ?? 0),
    cached_input_tokens: usage.cacheRead ?? 0,
  };
}

function modelKey(model) {
  return `${model.provider}/${model.id}`;
}

function toModelEntry(model) {
  return {
    model_id: modelKey(model),
    display_name: model.name ?? model.id,
    context_window: model.contextWindow ?? 128000,
    max_output_tokens: model.maxTokens ?? null,
  };
}

function findModel(models, modelId) {
  let model = models.find((m) => modelKey(m) === modelId);
  if (model) return model;
  // Convenience fallback for providers whose model ids are unique.
  const byBareId = models.filter((m) => m.id === modelId);
  return byBareId.length === 1 ? byBareId[0] : undefined;
}

async function listModels() {
  const { modelRegistry } = await getServices();
  modelRegistry.refresh();
  const includeUnavailable = process.env.PI_INFINITY_INCLUDE_UNAVAILABLE === "1";
  const models = includeUnavailable ? modelRegistry.getAll() : await modelRegistry.getAvailable();
  return models.map(toModelEntry);
}

async function invokeModel(modelId, request, writeLine) {
  const { modelRegistry } = await getServices();
  modelRegistry.refresh();
  const model = findModel(modelRegistry.getAll(), modelId);
  if (!model) {
    writeLine(providerResponse.error(`pi model not found: ${modelId}`));
    return;
  }

  const apiProvider = getApiProvider(model.api);
  if (!apiProvider?.streamSimple) {
    writeLine(providerResponse.error(`pi API provider not available for api '${model.api}'`));
    return;
  }

  const auth = await modelRegistry.getApiKeyAndHeaders(model);
  if (!auth.ok) {
    writeLine(providerResponse.error(auth.error));
    return;
  }

  const context = wireCompletionRequestToPiContext(request);
  const options = {
    temperature: request?.temperature ?? undefined,
    maxTokens: request?.max_tokens ?? undefined,
    apiKey: auth.apiKey,
    headers: auth.headers,
    reasoning: getReasoningLevel(model, request),
  };

  let stream;
  try {
    stream = apiProvider.streamSimple(model, context, options);
  } catch (error) {
    writeLine(providerResponse.error(errorMessage(error)));
    return;
  }

  writeLine(providerResponse.invokeStarted());
  const thinkingDeltasSeen = new Set();
  try {
    for await (const event of stream) {
      for (const item of eventToWireItems(event, thinkingDeltasSeen)) {
        writeLine(providerResponse.chunk(item));
      }
    }
  } catch (error) {
    writeLine(providerResponse.chunk(wireItem.error(errorMessage(error))));
  }
  writeLine(providerResponse.streamEnd());
}

function eventToWireItems(event, thinkingDeltasSeen) {
  switch (event.type) {
    case "text_delta":
      return event.delta ? [wireItem.message(event.delta)] : [];
    case "thinking_delta":
      if (event.delta) thinkingDeltasSeen?.add(event.contentIndex);
      return event.delta ? [wireItem.reasoningDelta(event.delta)] : [];
    case "thinking_end":
      // Most pi providers stream thinking via thinking_delta. If a provider only
      // reveals the finalized block/summary at thinking_end, forward it once.
      if (!thinkingDeltasSeen?.has(event.contentIndex) && event.content) {
        return [wireItem.reasoningSummary(event.content)];
      }
      return [];
    case "toolcall_end":
      return event.toolCall ? [wireItem.toolCall(event.toolCall)] : [];
    case "done":
      return [wireItem.final(event.message?.usage)];
    case "error":
      return [wireItem.error(event.error?.errorMessage ?? "pi provider error")];
    default:
      return [];
  }
}

function getReasoningLevel(model, request) {
  if (!model.reasoning) return undefined;

  const requested = request?.additional_params?.reasoning
    ?? request?.additional_params?.thinkingLevel
    ?? request?.additional_params?.thinking_level
    ?? process.env.PI_INFINITY_THINKING_LEVEL
    ?? "medium";

  if (requested === false || requested === "off" || requested === "none") return undefined;
  if (["minimal", "low", "medium", "high", "xhigh"].includes(requested)) return requested;

  console.error(`[pi warning] ignoring unsupported thinking level '${requested}' for ${modelKey(model)}`);
  return undefined;
}

function wireCompletionRequestToPiContext(request = {}) {
  const messages = [];
  for (const message of asMany(request.chat_history)) {
    const converted = wireMessageToPiMessage(message);
    if (converted) messages.push(converted);
  }
  for (const document of request.documents ?? []) {
    messages.push({ role: "user", content: stringifyDocument(document), timestamp: Date.now() });
  }
  return {
    systemPrompt: request.preamble ?? undefined,
    messages,
    tools: (request.tools ?? []).map(wireToolToPiTool).filter(Boolean),
  };
}

function wireMessageToPiMessage(message) {
  const [tag, value] = tagged(message);
  const role = tag?.toLowerCase?.();
  if (role === "user") {
    return { role: "user", content: contentToPi(value?.content ?? value), timestamp: Date.now() };
  }
  if (role === "assistant") {
    return {
      role: "assistant",
      content: assistantContentToPi(value?.content ?? value),
      api: "unknown",
      provider: "infinity",
      model: "history",
      usage: emptyUsage(),
      stopReason: "stop",
      timestamp: Date.now(),
    };
  }
  if (role === "toolresult" || role === "tool_result") {
    return {
      role: "toolResult",
      toolCallId: value?.id ?? value?.tool_call_id ?? value?.toolCallId ?? "tool-call",
      toolName: value?.name ?? value?.tool_name ?? value?.toolName ?? "tool",
      content: [{ type: "text", text: contentToText(value?.content ?? value) }],
      isError: Boolean(value?.is_error ?? value?.isError),
      timestamp: Date.now(),
    };
  }
  return { role: "user", content: contentToText(message), timestamp: Date.now() };
}

function assistantContentToPi(content) {
  const blocks = asMany(content).flatMap((part) => {
    const [tag, value] = tagged(part);
    const kind = tag?.toLowerCase?.();
    if (kind === "text") return [{ type: "text", text: typeof value === "string" ? value : value?.text ?? "" }];
    if (kind === "reasoning" || kind === "thinking") return [{ type: "thinking", thinking: typeof value === "string" ? value : value?.thinking ?? value?.reasoning ?? "" }];
    if (kind === "toolcall" || kind === "tool_call") {
      return [{ type: "toolCall", id: value?.id ?? randomUUID(), name: value?.name ?? value?.function?.name ?? "tool", arguments: value?.arguments ?? value?.function?.arguments ?? {} }];
    }
    return [{ type: "text", text: contentToText(part) }];
  });
  return blocks.length ? blocks : [{ type: "text", text: "" }];
}

function contentToPi(content) {
  const parts = asMany(content).map((part) => {
    const [tag, value] = tagged(part);
    if (tag?.toLowerCase?.() === "text") return { type: "text", text: typeof value === "string" ? value : value?.text ?? "" };
    if (tag?.toLowerCase?.() === "image") return { type: "image", data: value?.data ?? value?.base64 ?? "", mimeType: value?.mime_type ?? value?.mimeType ?? "image/png" };
    return { type: "text", text: contentToText(part) };
  });
  if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
  return parts;
}

function wireToolToPiTool(tool) {
  if (!tool?.name) return undefined;
  return { name: tool.name, description: tool.description ?? "", parameters: tool.parameters ?? tool.input_schema ?? {} };
}

function asMany(value) {
  if (value == null) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === "object") {
    if (Array.isArray(value.Many)) return value.Many;
    if (Array.isArray(value.many)) return value.many;
    if ("One" in value) return [value.One];
    if ("one" in value) return [value.one];
  }
  return [value];
}

function tagged(value) {
  if (typeof value === "string") return [value, null];
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    if (keys.length === 1) return [keys[0], value[keys[0]]];
    if (typeof value.type === "string") return [value.type, value];
    if (typeof value.role === "string") return [value.role, value];
  }
  return [undefined, value];
}

function contentToText(value) {
  if (typeof value === "string") return value;
  return asMany(value).map((part) => {
    const [tag, inner] = tagged(part);
    if (tag?.toLowerCase?.() === "text") return typeof inner === "string" ? inner : inner?.text ?? "";
    if (typeof part === "string") return part;
    return JSON.stringify(part);
  }).join("\n");
}

function stringifyDocument(document) {
  if (typeof document === "string") return document;
  const title = document?.title ? `# ${document.title}\n\n` : "";
  return `${title}${document?.text ?? document?.content ?? JSON.stringify(document)}`;
}

function emptyUsage() {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
}

function parseRequest(line) {
  const parsed = JSON.parse(line);
  if (parsed === "ListModels") return { type: "ListModels" };
  if (parsed?.InvokeModel) return { type: "InvokeModel", ...parsed.InvokeModel };
  throw new Error(`unknown provider request: ${line}`);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function handleConnection(socket) {
  let buffer = "";
  let handled = false;
  const writeLine = (value) => socket.write(`${JSON.stringify(value)}\n`);

  socket.setEncoding("utf8");
  socket.on("data", async (chunk) => {
    if (handled) return;
    buffer += chunk;
    const newline = buffer.indexOf("\n");
    if (newline === -1) return;
    handled = true;

    try {
      const request = parseRequest(buffer.slice(0, newline));
      if (request.type === "ListModels") {
        writeLine(providerResponse.models(await listModels()));
      } else {
        await invokeModel(request.model_id, request.request, writeLine);
      }
    } catch (error) {
      writeLine(providerResponse.error(errorMessage(error)));
    } finally {
      socket.end();
    }
  });
  socket.on("error", (error) => console.error(`[socket error] ${errorMessage(error)}`));
}

await fs.rm(socketPath, { force: true });
const server = net.createServer(handleConnection);
server.on("error", (error) => {
  console.error(`[server error] ${errorMessage(error)}`);
  process.exitCode = 1;
});

await new Promise((resolve, reject) => server.listen(socketPath, resolve).once("error", reject));

// Infinity provider stdout contract: first line is the Unix socket path.
console.log(socketPath);

async function cleanup() {
  server.close();
  await fs.rm(socketPath, { force: true }).catch(() => {});
}
process.on("SIGINT", async () => { await cleanup(); process.exit(130); });
process.on("SIGTERM", async () => { await cleanup(); process.exit(143); });
process.on("exit", () => { try { server.close(); } catch {} });
