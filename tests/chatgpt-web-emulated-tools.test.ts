import { describe, expect, test } from "bun:test";
import { ChatGptBrowserWorker, type BrowserTurn } from "../src/adapters/chatgpt-web/browser-worker";
import {
  buildChatGptEmulatedToolContract,
  chatGptEmulatedToolsForRequest,
  parseChatGptEmulatedToolResponse,
} from "../src/adapters/chatgpt-web/emulated-tools";
import { createChatGptWebAdapter } from "../src/adapters/chatgpt-web/index";
import { CHATGPT_WEB_MODEL_ID } from "../src/adapters/chatgpt-web/model";
import { compileChatGptWebPrompt } from "../src/adapters/chatgpt-web/prompt";
import { chatGptTurnSessions } from "../src/adapters/chatgpt-web/turn-execution";
import type { AdapterEvent, CodexParsedRequest, CodexProviderConfig, CodexTool } from "../src/types";

const tools: CodexTool[] = [
  {
    name: "exec_command",
    description: "Run a shell command",
    parameters: {
      type: "object",
      properties: { cmd: { type: "string", minLength: 1 } },
      required: ["cmd"],
      additionalProperties: false,
    },
  },
  {
    name: "apply_patch",
    description: "Apply a patch",
    parameters: {},
    freeform: true,
  },
  {
    name: "search_docs",
    namespace: "mcp__docs",
    description: "Search documentation",
    parameters: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    },
  },
];

function proRequest(): CodexParsedRequest {
  const turnId = "turn_emulated_tools_test";
  const threadId = "thread_emulated_tools_test";
  return {
    modelId: CHATGPT_WEB_MODEL_ID,
    stream: true,
    context: {
      tools,
      messages: [{ role: "user", content: "Inspect the repository and report back", timestamp: 1 }],
    },
    options: { reasoning: "max" },
    _rawBody: {
      prompt_cache_key: threadId,
      client_metadata: {
        "x-codex-turn-metadata": JSON.stringify({ thread_id: threadId, turn_id: turnId }),
      },
      input: [{
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Inspect the repository and report back" }],
        internal_chat_message_metadata_passthrough: { turn_id: turnId },
      }],
    },
  };
}

function relayBlock(nonce: string, payload: unknown): string {
  return `[[CODEX_TOOL_RELAY_BEGIN_${nonce}]]\n${JSON.stringify(payload)}\n[[CODEX_TOOL_RELAY_END_${nonce}]]`;
}

const capabilities = { localToolsEnabled: false, solAvailable: true, proAvailable: true };

describe("ChatGPT Pro emulated Codex tools", () => {
  test("builds a strict Toolify-style relay contract without an MCP turn token", () => {
    const request = proRequest();
    const lines = buildChatGptEmulatedToolContract(request, "relay_123456789abc");
    const text = lines.join("\n");
    expect(text).toContain("[[CODEX_TOOL_RELAY_BEGIN_relay_123456789abc]]");
    expect(text).toContain("[[CODEX_TOOL_RELAY_END_relay_123456789abc]]");
    expect(text).toContain('"name":"exec_command"');
    expect(text).toContain('"name":"apply_patch"');
    expect(text).toContain('"name":"mcp__docs__search_docs"');

    const compiled = compileChatGptWebPrompt(
      request,
      capabilities,
      undefined,
      { emulatedToolNonce: "relay_123456789abc" },
    );
    expect(compiled.text).toContain("outer Codex tool relay");
    expect(compiled.text).not.toContain("with no Codex Native bridge to the user's local computer");
    expect(compiled.text).not.toContain("turn_token");
  });

  test("parses declared JSON and freeform calls and rejects malformed arguments", () => {
    const request = proRequest();
    const nonce = "relay_123456789abc";
    const parsed = parseChatGptEmulatedToolResponse(
      relayBlock(nonce, {
        calls: [
          { name: "exec_command", arguments: { cmd: "pwd" } },
          { name: "apply_patch", input: "*** Begin Patch\n*** End Patch" },
        ],
      }),
      request,
      nonce,
    );
    expect(parsed).toHaveLength(2);
    expect(parsed?.[0]).toMatchObject({ wireName: "exec_command", freeform: false, arguments: { cmd: "pwd" } });
    expect(parsed?.[1]).toMatchObject({ wireName: "apply_patch", freeform: true, input: "*** Begin Patch\n*** End Patch" });
    expect(parsed?.[0]?.callId).toMatch(/^call_[a-f0-9]{32}$/);

    expect(() => parseChatGptEmulatedToolResponse(
      relayBlock(nonce, { calls: [{ name: "exec_command", arguments: {} }] }),
      request,
      nonce,
    )).toThrow("arguments.cmd is required");
    expect(() => parseChatGptEmulatedToolResponse(
      relayBlock(nonce, { calls: [{ name: "not_declared", arguments: {} }] }),
      request,
      nonce,
    )).toThrow("undeclared or disallowed");
    expect(() => parseChatGptEmulatedToolResponse(
      `prefix ${relayBlock(nonce, { calls: [{ name: "exec_command", arguments: { cmd: "pwd" } }] })}`,
      request,
      nonce,
    )).toThrow("malformed or mixed");
  });

  test("respects tool_choice restrictions", () => {
    const request = proRequest();
    request.options.toolChoice = { name: "exec_command" };
    expect(chatGptEmulatedToolsForRequest(request).map(tool => tool.name)).toEqual(["exec_command"]);
    expect(() => parseChatGptEmulatedToolResponse("normal answer", request, "relay_123456789abc"))
      .toThrow("tool_choice requires a tool call");

    request.options.toolChoice = "none";
    expect(chatGptEmulatedToolsForRequest(request)).toEqual([]);
  });

  test("turns a Pro browser control block into native Codex tool events, then resumes with the tool result", async () => {
    chatGptTurnSessions.clear();
    const provider: CodexProviderConfig = {
      adapter: "chatgpt-web",
      baseUrl: `browser://emulated-pro-test-${Date.now()}`,
      chatgptWeb: { localToolsEnabled: false, solAvailable: true, proAvailable: true },
    };
    const worker = ChatGptBrowserWorker.forProvider(provider);
    const originalRun = worker.run.bind(worker);
    let browserStarts = 0;
    (worker as unknown as { run: (turn: BrowserTurn) => Promise<string> }).run = async turn => {
      browserStarts += 1;
      const prepared = await turn.prepare();
      const nonce = prepared.text.match(/\[\[CODEX_TOOL_RELAY_BEGIN_([A-Za-z0-9_-]+)\]\]/)?.[1];
      if (!nonce) throw new Error("emulated tool nonce missing from Pro prompt");
      if (browserStarts === 1) {
        const answer = relayBlock(nonce, {
          calls: [{ name: "exec_command", arguments: { cmd: "pwd" } }],
        });
        // The adapter intentionally buffers this stream so the control block never reaches Codex as text.
        turn.onTextDelta(answer);
        return answer;
      }
      expect(prepared.text).toContain("tool_result");
      expect(prepared.text).toContain("/workspace/project");
      const answer = "Repository inspected successfully.";
      turn.onTextDelta(answer);
      return answer;
    };

    const adapter = createChatGptWebAdapter(provider);
    const first = proRequest();
    const firstEvents: AdapterEvent[] = [];
    try {
      await adapter.runTurn!(first, { headers: new Headers() }, event => firstEvents.push(event));
      const start = firstEvents.find(
        (event): event is Extract<AdapterEvent, { type: "tool_call_start" }> => event.type === "tool_call_start",
      );
      expect(start?.name).toBe("exec_command");
      expect(firstEvents.some(event => event.type === "text_delta" && event.text.includes("CODEX_TOOL_RELAY"))).toBe(false);
      expect(firstEvents.at(-1)).toMatchObject({ type: "done", stopReason: "tool_use", endTurn: false });

      const continuation = structuredClone(first);
      continuation.context.messages.push(
        {
          role: "assistant",
          content: [{ type: "toolCall", id: start!.id, name: "exec_command", arguments: { cmd: "pwd" } }],
          timestamp: 2,
        },
        {
          role: "toolResult",
          toolCallId: start!.id,
          toolName: "exec_command",
          content: JSON.stringify({ output: "/workspace/project", exit_code: 0 }),
          isError: false,
          timestamp: 3,
        },
      );

      const finalEvents: AdapterEvent[] = [];
      await adapter.runTurn!(continuation, { headers: new Headers() }, event => finalEvents.push(event));
      expect(browserStarts).toBe(2);
      expect(finalEvents.some(event => event.type === "text_delta" && event.text === "Repository inspected successfully.")).toBe(true);
      expect(finalEvents.at(-1)).toMatchObject({ type: "done", stopReason: "stop", endTurn: true });
    } finally {
      (worker as unknown as { run: (turn: BrowserTurn) => Promise<string> }).run = originalRun;
      chatGptTurnSessions.clear();
    }
  });
});
