"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ExampleChips from "./ExampleChips";
import ReportViewer from "./ReportViewer";

interface Message {
  role: "user" | "assistant";
  content: string;
  tools_used?: string[];
  sources?: string[];
}

interface ChatInterfaceProps {
  externalInput?: string;
  onExternalInputConsumed?: () => void;
}

const TOOL_BADGE_MAP: Record<string, string> = {
  query_campaign_data: "tool-badge-query",
  search_similar_campaigns: "tool-badge-search",
  compare_campaigns: "tool-badge-compare",
  generate_lci_report: "tool-badge-report",
  recommend_audience: "tool-badge-audience",
};

const TOOL_LABELS: Record<string, string> = {
  query_campaign_data: "SQL Query",
  search_similar_campaigns: "Semantic Search",
  compare_campaigns: "Compare",
  generate_lci_report: "Report",
  recommend_audience: "Audience",
};

export default function ChatInterface({
  externalInput,
  onExternalInputConsumed,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Handle external input from sidebar quick actions
  useEffect(() => {
    if (externalInput) {
      handleSubmitMessage(externalInput);
      onExternalInputConsumed?.();
    }
  }, [externalInput]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmitMessage(message: string) {
    if (!message.trim() || isStreaming) return;

    const userMessage: Message = { role: "user", content: message };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsStreaming(true);

    // Add empty assistant message for streaming
    const assistantMessage: Message = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const response = await fetch(`/api/chat?stream=true`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";
      let toolsUsed: string[] = [];
      let sources: string[] = [];

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          const lines = text.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.content) {
                  fullContent += data.content;
                  setMessages((prev) => {
                    const updated = [...prev];
                    updated[updated.length - 1] = {
                      role: "assistant",
                      content: fullContent,
                    };
                    return updated;
                  });
                }
                if (data.done) {
                  toolsUsed = data.tools_used || [];
                  sources = data.sources || [];
                }
              } catch {
                // Skip malformed JSON
              }
            }
          }
        }
      }

      // Update final message with metadata
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: fullContent || "No response received.",
          tools_used: toolsUsed,
          sources: sources,
        };
        return updated;
      });
    } catch (error) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    handleSubmitMessage(input);
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="border-b border-gray-200 px-6 py-4 bg-white">
        <h1 className="text-lg font-semibold text-gray-900">
          Campaign Intelligence Assistant
        </h1>
        <p className="text-sm text-gray-500">
          Ask questions about campaign performance, generate reports, and get
          insights.
        </p>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-4xl mb-4">&#x1f4ca;</div>
            <h2 className="text-xl font-semibold text-gray-700 mb-2">
              Campaign Intelligence Assistant
            </h2>
            <p className="text-gray-500 mb-6 max-w-md">
              Ask me about campaign performance, generate reports, compare
              campaigns, or get audience recommendations.
            </p>
            <ExampleChips
              onSelect={(query) => handleSubmitMessage(query)}
            />
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`chat-bubble flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-white border border-gray-200 text-gray-800"
              }`}
            >
              {msg.role === "assistant" ? (
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                </div>
              ) : (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              )}

              {/* Tool badges */}
              {msg.tools_used && msg.tools_used.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-gray-100">
                  {msg.tools_used.map((tool, j) => (
                    <span
                      key={j}
                      className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        TOOL_BADGE_MAP[tool] || "bg-gray-100 text-gray-700"
                      }`}
                    >
                      {TOOL_LABELS[tool] || tool}
                    </span>
                  ))}
                </div>
              )}

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="text-xs text-gray-400 mt-1">
                  Sources: {msg.sources.join(", ")}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {isStreaming &&
          messages.length > 0 &&
          messages[messages.length - 1].content === "" && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                <div className="flex space-x-1">
                  <div className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                  <div className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                  <div className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                </div>
              </div>
            </div>
          )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="border-t border-gray-200 bg-white px-6 py-4">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about campaign performance..."
            className="flex-1 rounded-xl border border-gray-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isStreaming}
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="bg-blue-600 text-white px-5 py-2.5 rounded-xl text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
