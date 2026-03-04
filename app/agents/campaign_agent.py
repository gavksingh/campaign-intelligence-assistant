"""LangGraph agent definition for the Campaign Intelligence Assistant.

Defines the agent state schema, graph nodes (LLM reasoning, tool execution),
and edges (routing logic). The compiled graph is the main entry point for
processing user queries.
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State passed between nodes in the agent graph.

    Attributes:
        messages: Conversation history (LangGraph message format).
        campaign_context: Retrieved campaign data for grounding.
    """

    messages: Annotated[list, add_messages]
    campaign_context: str


async def reasoning_node(state: AgentState) -> AgentState:
    """LLM reasoning node — decides what action to take or generates a response.

    Args:
        state: Current agent state with messages and context.

    Returns:
        Updated state with the LLM's response appended to messages.
    """
    raise NotImplementedError("Reasoning node not yet implemented.")


async def tool_node(state: AgentState) -> AgentState:
    """Execute a tool call requested by the LLM.

    Args:
        state: Current agent state containing a pending tool call.

    Returns:
        Updated state with tool results appended to messages.
    """
    raise NotImplementedError("Tool node not yet implemented.")


def should_use_tool(state: AgentState) -> str:
    """Routing function: decide whether to call a tool or end the turn.

    Args:
        state: Current agent state.

    Returns:
        'tools' to route to tool_node, or END to finish.
    """
    raise NotImplementedError("Routing logic not yet implemented.")


def build_agent_graph() -> StateGraph:
    """Construct and compile the LangGraph agent.

    Graph structure::

        [reasoning] --should_use_tool--> [tools] --> [reasoning]
                   \\--END

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    raise NotImplementedError("Agent graph construction not yet implemented.")
