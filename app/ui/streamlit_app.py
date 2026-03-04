"""Streamlit chat interface for the Campaign Intelligence Assistant.

Provides a simple chat UI for internal teams to query campaign data,
request reports, and get audience recommendations via natural language.
"""

import streamlit as st

API_BASE_URL = "http://localhost:8080"

st.set_page_config(
    page_title="Campaign Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("Campaign Intelligence Assistant")
st.caption("Ask questions about campaign performance, generate reports, or get audience recommendations.")


def init_session_state() -> None:
    """Initialize Streamlit session state for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_chat_history() -> None:
    """Render all messages in the chat history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


async def send_message(user_message: str) -> str:
    """Send a message to the API and return the response.

    Args:
        user_message: The user's natural-language query.

    Returns:
        The assistant's reply text.
    """
    raise NotImplementedError("API integration not yet implemented.")


def main() -> None:
    """Main Streamlit app loop."""
    init_session_state()
    display_chat_history()

    if prompt := st.chat_input("Ask about campaign performance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.info("Chat functionality coming soon — backend integration pending.")

        st.session_state.messages.append(
            {"role": "assistant", "content": "_Chat functionality coming soon._"}
        )


if __name__ == "__main__":
    main()
