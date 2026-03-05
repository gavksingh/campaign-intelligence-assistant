"""LangGraph agent for the Campaign Intelligence Assistant.

Defines the stateful agent graph with nodes for routing, tool execution,
response synthesis, and error handling. The compiled graph processes
natural-language campaign queries end-to-end.

Usage::

    from app.agents.campaign_agent import invoke_agent

    response = await invoke_agent("How did Dunkin's Q4 compare to Q3?")
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from functools import partial
from typing import Annotated, Literal

import requests
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agents.tools import ALL_TOOLS
from app.config import settings

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

SYSTEM_PROMPT = """You are the Campaign Intelligence Assistant, an AI analyst for an adtech company (InMarket).
You help internal teams understand campaign performance, generate reports, and make data-driven decisions.

You have access to these tools:
- query_campaign_data: Query the campaign database with natural language (translates to SQL). Use for specific lookups by client, vertical, date range, status, or metrics.
- search_similar_campaigns: Semantic search to find campaigns by description. Use for exploratory queries or finding campaigns by characteristics.
- compare_campaigns: Compare two campaigns side-by-side. Needs two campaign integer database IDs (the "id" field from query/search results, NOT the UUID "campaign_id").
- generate_lci_report: Generate a Location Conversion Index attribution report for a campaign. Needs the integer database ID (the "id" field).
- recommend_audience: Get audience segment recommendations based on a description. Use when asked about targeting or audience strategy.

Guidelines:
- For comparison queries (e.g., "compare Q3 vs Q4 for Dunkin"), first use query_campaign_data or search_similar_campaigns to find the campaign IDs, then use compare_campaigns.
- For "generate a report for the best X campaign", first search/query to find it, then use generate_lci_report with the id.
- Always reference specific numbers and data points in your responses.
- If you cannot find the data, say so clearly rather than making up numbers.
- Be concise but thorough. Internal teams value actionable insights.
"""


# ── State ─────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """State passed between nodes in the agent graph.

    Attributes:
        messages: Conversation history managed by LangGraph's add_messages reducer.
        campaign_context: Retrieved campaign data used as grounding context.
        current_tool_results: Raw results from the most recent tool execution.
        report_data: Generated report data (if a report was requested).
        error_count: Number of consecutive errors for retry logic.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    campaign_context: str
    current_tool_results: str
    report_data: str
    error_count: int


# ── LLM helpers (direct Groq API via requests) ──────────────────────


# Pre-compute OpenAI-format tool definitions once
_OAI_TOOLS = [convert_to_openai_tool(t) for t in ALL_TOOLS]


def _messages_to_oai(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI-format dicts."""
    oai = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            oai.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            oai.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            m: dict = {"role": "assistant", "content": msg.content or ""}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            oai.append(m)
        elif isinstance(msg, ToolMessage):
            oai.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
    return oai


def _groq_chat_sync(
    oai_messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Synchronous HTTP POST to Groq API via requests (no httpx).

    Retries up to 3 times on 429 (rate limit) with exponential backoff.
    """
    import time

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": settings.llm_model,
        "messages": oai_messages,
        "temperature": 0.3,
    }
    if tools:
        payload["tools"] = tools

    max_retries = 3
    for attempt in range(max_retries):
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429 and attempt < max_retries - 1:
            retry_after = float(resp.headers.get("retry-after", str(2**attempt)))
            wait = min(retry_after, 8)
            logger.warning(
                "Groq 429 rate limit (attempt %d/%d), waiting %.1fs",
                attempt + 1,
                max_retries,
                wait,
            )
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            logger.error("Groq API error %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        return resp.json()

    resp.raise_for_status()
    return resp.json()


def _parse_groq_response(data: dict) -> AIMessage:
    """Convert a Groq API response dict to a LangChain AIMessage."""
    choice = data["choices"][0]["message"]
    content = choice.get("content", "") or ""
    raw_tool_calls = choice.get("tool_calls")

    additional_kwargs = {}
    lc_tool_calls = []
    if raw_tool_calls:
        additional_kwargs["tool_calls"] = raw_tool_calls
        for tc in raw_tool_calls:
            lc_tool_calls.append(
                {
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                    "id": tc["id"],
                    "type": "tool_call",
                }
            )

    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_calls=lc_tool_calls,
    )


async def _call_groq(
    messages: list[BaseMessage],
    with_tools: bool = True,
) -> AIMessage:
    """Call the Groq API asynchronously (runs sync request in thread pool)."""
    oai_messages = _messages_to_oai(messages)
    tools = _OAI_TOOLS if with_tools else None

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(
        None,
        partial(_groq_chat_sync, oai_messages, tools),
    )
    return _parse_groq_response(data)


# ── Nodes ─────────────────────────────────────────────────────────────


async def router_node(state: AgentState) -> dict:
    """Analyze the user query and decide which tool(s) to call.

    Sends the conversation history to the LLM with tool definitions.
    The LLM either produces tool calls or a direct response.

    Args:
        state: Current agent state.

    Returns:
        Updated state dict with the LLM's response appended to messages.
    """
    logger.info("router_node: processing %d messages", len(state["messages"]))

    # Ensure system prompt is first
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    # Add context if available
    if state.get("campaign_context"):
        ctx_msg = SystemMessage(
            content=f"Previously retrieved campaign context:\n{state['campaign_context']}"
        )
        messages.insert(1, ctx_msg)

    try:
        response = await _call_groq(messages, with_tools=True)
        logger.info(
            "router_node: LLM responded with %d tool calls, content=%s",
            len(response.tool_calls) if hasattr(response, "tool_calls") else 0,
            bool(response.content),
        )
        return {"messages": [response]}
    except requests.exceptions.HTTPError as e:
        logger.error("router_node HTTP error: %s", e, exc_info=True)
        status = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text[:300] if e.response is not None else "no body"
        error_msg = AIMessage(content=f"API error (HTTP {status}): {body}")
        return {"messages": [error_msg], "error_count": state.get("error_count", 0) + 1}
    except Exception as e:
        logger.error("router_node LLM call failed: %s: %s", type(e).__name__, e)
        error_msg = AIMessage(content=f"Error: {type(e).__name__}: {str(e)[:300]}")
        return {"messages": [error_msg], "error_count": state.get("error_count", 0) + 1}


async def tool_executor_node(state: AgentState) -> dict:
    """Execute tool calls from the most recent AI message.

    Looks at the last message for tool_calls, executes each one, and
    appends ToolMessage results to the conversation.

    Args:
        state: Current agent state.

    Returns:
        Updated state dict with tool results.
    """
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.warning("tool_executor_node: no tool calls found")
        return {"current_tool_results": "", "error_count": 0}

    tool_map = {t.name: t for t in ALL_TOOLS}
    tool_messages: list[ToolMessage] = []
    all_results: list[str] = []

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        logger.info("Executing tool: %s with args: %s", tool_name, tool_args)

        tool_fn = tool_map.get(tool_name)
        if not tool_fn:
            error_result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            tool_messages.append(
                ToolMessage(content=error_result, tool_call_id=tool_id)
            )
            all_results.append(error_result)
            continue

        try:
            result = await tool_fn.ainvoke(tool_args)
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            all_results.append(str(result))
            logger.info("Tool %s completed successfully", tool_name)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
            error_result = json.dumps({"error": f"Tool {tool_name} failed: {e}"})
            tool_messages.append(
                ToolMessage(content=error_result, tool_call_id=tool_id)
            )
            all_results.append(error_result)

    combined_results = "\n---\n".join(all_results)

    # Extract any campaign context from results for future turns
    campaign_context = state.get("campaign_context", "")
    for result_str in all_results:
        try:
            parsed = json.loads(result_str)
            if "campaigns" in parsed:
                campaign_context = result_str
            elif "campaign_name" in parsed:
                # Report or comparison result — store as report data
                return {
                    "messages": tool_messages,
                    "current_tool_results": combined_results,
                    "report_data": result_str,
                    "campaign_context": campaign_context,
                    "error_count": 0,
                }
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "messages": tool_messages,
        "current_tool_results": combined_results,
        "campaign_context": campaign_context,
        "error_count": 0,
    }


async def synthesizer_node(state: AgentState) -> dict:
    """Generate a final natural-language response from tool results.

    Takes the full conversation (including tool results) and produces
    a user-friendly summary.

    Args:
        state: Current agent state.

    Returns:
        Updated state dict with the synthesized response.
    """
    logger.info("synthesizer_node: synthesizing response")

    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    try:
        response = await _call_groq(messages, with_tools=True)

        # If the LLM wants more tool calls during synthesis, just use its content
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info("synthesizer_node: LLM wants more tool calls, routing back")
            return {"messages": [response]}

        return {"messages": [response]}
    except requests.exceptions.HTTPError as e:
        logger.error("synthesizer_node HTTP error: %s", e, exc_info=True)
        status = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text[:300] if e.response is not None else "no body"
        fallback = AIMessage(content=f"Synthesis API error (HTTP {status}): {body}")
        return {"messages": [fallback]}
    except Exception as e:
        logger.error("synthesizer_node failed: %s: %s", type(e).__name__, e)
        raw_results = state.get("current_tool_results", "")
        if raw_results:
            fallback = AIMessage(
                content=(
                    "I retrieved data but had trouble formatting. "
                    f"Error: {type(e).__name__}: {str(e)[:200]}\n\n"
                    "Raw results:\n" + raw_results[:2000]
                )
            )
        else:
            fallback = AIMessage(content=f"Error: {type(e).__name__}: {str(e)[:300]}")
        return {"messages": [fallback]}


async def error_handler_node(state: AgentState) -> dict:
    """Handle errors and decide whether to retry or give up.

    If under the retry limit, adds a message asking the LLM to try
    a different approach. Otherwise, generates a graceful failure response.

    Args:
        state: Current agent state.

    Returns:
        Updated state dict with error handling message.
    """
    error_count = state.get("error_count", 0)
    logger.warning("error_handler_node: error_count=%d", error_count)

    if error_count <= MAX_RETRIES:
        retry_msg = HumanMessage(
            content=(
                "The previous approach encountered an error. "
                "Please try a different strategy to answer the user's question. "
                "If a SQL query failed, try using search_similar_campaigns instead. "
                "If search failed, try query_campaign_data with a simpler query."
            )
        )
        return {"messages": [retry_msg]}
    else:
        give_up_msg = AIMessage(
            content=(
                "I apologize, but I'm having difficulty processing your request "
                "after multiple attempts. Here's what I'd suggest:\n\n"
                "1. Try rephrasing your question with more specific details\n"
                "2. Ask about a specific campaign by name\n"
                "3. Ask for a list of all campaigns to browse\n\n"
                "I'm ready to help with your next question."
            )
        )
        return {"messages": [give_up_msg], "error_count": 0}


# ── Routing logic ─────────────────────────────────────────────────────


def route_after_router(
    state: AgentState,
) -> Literal["tool_executor", "synthesizer", "error_handler"]:
    """Decide where to route after the router node.

    Args:
        state: Current agent state.

    Returns:
        Next node name.
    """
    last_message = state["messages"][-1]

    # Check for errors
    if state.get("error_count", 0) > MAX_RETRIES:
        return "error_handler"

    # Check if LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_executor"

    # No tool calls — go straight to END via synthesizer
    return "synthesizer"


def route_after_tools(
    state: AgentState,
) -> Literal["router", "synthesizer", "error_handler"]:
    """Decide where to route after tool execution.

    Args:
        state: Current agent state.

    Returns:
        Next node name.
    """
    results = state.get("current_tool_results", "")

    # Check if any tool returned an error
    has_error = False
    try:
        for chunk in results.split("\n---\n"):
            parsed = json.loads(chunk)
            if "error" in parsed:
                has_error = True
                break
    except (json.JSONDecodeError, TypeError):
        pass

    if has_error and state.get("error_count", 0) <= MAX_RETRIES:
        return "error_handler"

    # Send results back to the LLM for synthesis
    # Route to router so it can decide if more tools are needed
    return "router"


def route_after_error(state: AgentState) -> Literal["router", "__end__"]:
    """Decide where to route after error handling.

    Args:
        state: Current agent state.

    Returns:
        Next node name.
    """
    if state.get("error_count", 0) > MAX_RETRIES:
        return END
    return "router"


def route_after_synthesizer(state: AgentState) -> Literal["tool_executor", "__end__"]:
    """Decide if synthesis needs more tool calls or is done.

    Args:
        state: Current agent state.

    Returns:
        Next node name.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_executor"
    return END


# ── Graph construction ────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct the LangGraph agent graph.

    Graph structure::

        START → router → tool_executor → router (loop for multi-step)
                  ↓              ↓
              synthesizer   error_handler
                  ↓              ↓
                 END          router (retry) or END (give up)

    Returns:
        A compiled LangGraph StateGraph.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("error_handler", error_handler_node)

    # Entry point
    graph.add_edge(START, "router")

    # Router decides: tools, synthesize, or error
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "tool_executor": "tool_executor",
            "synthesizer": "synthesizer",
            "error_handler": "error_handler",
        },
    )

    # After tools: back to router (for multi-step), synthesize, or error
    graph.add_conditional_edges(
        "tool_executor",
        route_after_tools,
        {
            "router": "router",
            "synthesizer": "synthesizer",
            "error_handler": "error_handler",
        },
    )

    # After error: retry via router or give up (END)
    graph.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "router": "router",
            END: END,
        },
    )

    # After synthesis: done or more tools if LLM wants them
    graph.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {
            "tool_executor": "tool_executor",
            END: END,
        },
    )

    return graph


# Compile the graph once at module level
compiled_graph = build_graph().compile()


# ── Convenience entry point ───────────────────────────────────────────


async def invoke_agent(query: str, session_id: str | None = None) -> dict:
    """Run a user query through the campaign agent.

    Args:
        query: Natural-language question from the user.
        session_id: Optional session ID for conversation tracking.

    Returns:
        Dict with 'reply' (str), 'sources' (list[str]), and optionally 'data' (dict).
    """
    logger.info("invoke_agent: query='%s'", query[:100])

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "campaign_context": "",
        "current_tool_results": "",
        "report_data": "",
        "error_count": 0,
    }

    try:
        final_state = await compiled_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id or "default"}},
        )

        # Extract the final AI response
        messages = final_state.get("messages", [])
        reply = "I wasn't able to generate a response. Please try again."
        sources: list[str] = []

        # Walk messages in reverse to find the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        # Extract source campaign names from tool results
        try:
            ctx = final_state.get("campaign_context", "")
            if ctx:
                parsed = json.loads(ctx)
                campaigns = parsed.get("campaigns", [])
                sources = [
                    c.get("campaign_name", "Unknown")
                    for c in campaigns[:5]
                    if isinstance(c, dict)
                ]
        except (json.JSONDecodeError, TypeError):
            pass

        # Include report data if generated
        data = None
        if final_state.get("report_data"):
            try:
                data = json.loads(final_state["report_data"])
            except (json.JSONDecodeError, TypeError):
                pass

        return {"reply": reply, "sources": sources, "data": data}

    except Exception as e:
        logger.error("invoke_agent failed: %s", e, exc_info=True)
        return {
            "reply": (
                "I'm sorry, I encountered an unexpected error processing your request. "
                "Please try again or rephrase your question."
            ),
            "sources": [],
            "data": None,
        }


async def stream_agent(
    query: str, session_id: str | None = None
) -> AsyncGenerator[dict, None]:
    """Run the agent and stream the final reply in chunks via SSE.

    Runs the full agent graph (tool execution is non-streamable), then
    streams the synthesized reply using the LLM streaming API.

    Args:
        query: Natural-language question from the user.
        session_id: Optional session ID for conversation tracking.

    Yields:
        Dicts with 'content' (str), 'done' (bool), and optionally
        'tools_used' and 'sources' on the final chunk.
    """
    from app.services.llm_client import get_llm_client

    logger.info("stream_agent: query='%s'", query[:100])

    # Run the full agent graph first to get tool results
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "campaign_context": "",
        "current_tool_results": "",
        "report_data": "",
        "error_count": 0,
    }

    try:
        final_state = await compiled_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id or "default"}},
        )

        messages = final_state.get("messages", [])

        # Extract sources and tools
        sources: list[str] = []
        tools_used: list[str] = []
        try:
            ctx = final_state.get("campaign_context", "")
            if ctx:
                parsed = json.loads(ctx)
                campaigns = parsed.get("campaigns", [])
                sources = [
                    c.get("campaign_name", "Unknown")
                    for c in campaigns[:5]
                    if isinstance(c, dict)
                ]
        except (json.JSONDecodeError, TypeError):
            pass

        # Collect tool names from messages
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tools_used.append("tool_call")
            elif (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tc in msg.tool_calls:
                    tools_used.append(tc.get("name", "unknown"))

        tools_used = list(dict.fromkeys(tools_used))  # deduplicate preserving order
        tools_used = [t for t in tools_used if t != "tool_call"]

        # Build messages for streaming synthesis
        synth_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in messages:
            if isinstance(msg, HumanMessage):
                synth_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                synth_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                synth_messages.append(
                    {"role": "user", "content": f"[Tool result]: {msg.content[:2000]}"}
                )

        # Stream the final synthesis
        llm_client = get_llm_client()
        async for chunk in llm_client.stream_chat_completion(synth_messages):
            yield {"content": chunk, "done": False}

        # Final message with metadata
        yield {
            "content": "",
            "done": True,
            "tools_used": tools_used,
            "sources": sources,
        }

    except Exception as e:
        logger.error("stream_agent failed: %s", e, exc_info=True)
        yield {
            "content": "I'm sorry, I encountered an error processing your request.",
            "done": True,
            "tools_used": [],
            "sources": [],
        }
