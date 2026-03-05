"""Streamlit chat interface for the Campaign Intelligence Assistant.

A professional internal tool UI for querying campaign performance data,
generating reports, comparing campaigns, and getting audience recommendations.
Communicates with the FastAPI backend via HTTP.

Usage::

    streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────

_api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
API_BASE = f"{_api_base_url.rstrip('/')}/api"
REQUEST_TIMEOUT = 120.0

st.set_page_config(
    page_title="Campaign Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background-color: #2a3040;
        border: 1px solid #3a4050;
        color: #e0e0e0;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        margin-bottom: 2px;
        transition: background-color 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #3a4a60;
        border-color: #5a8dee;
    }

    /* Main area */
    .main .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 8px;
    }

    /* Tool badge */
    .tool-badge {
        display: inline-block;
        background: #f0f2f6;
        color: #6b7280;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 4px;
        margin-top: 4px;
    }

    /* Example chip */
    .example-chip {
        display: inline-block;
        background: #f8f9fb;
        border: 1px solid #e2e5ea;
        border-radius: 16px;
        padding: 5px 14px;
        font-size: 0.82rem;
        color: #4a5568;
        cursor: pointer;
        margin: 3px;
        transition: all 0.15s;
    }
    .example-chip:hover {
        background: #edf2f7;
        border-color: #5a8dee;
        color: #2d3748;
    }

    /* Status dot */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-green { background-color: #48bb78; }
    .status-red { background-color: #f56565; }
    .status-yellow { background-color: #ecc94b; }

    /* Footer */
    .footer-badge {
        text-align: center;
        color: #a0aec0;
        font-size: 0.75rem;
        padding: 1rem 0;
        margin-top: 2rem;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ── API Client ────────────────────────────────────────────────────────


def _api_get(path: str, params: dict | None = None) -> dict | None:
    """Send a GET request to the API.

    Args:
        path: API path (e.g. '/health').
        params: Optional query parameters.

    Returns:
        JSON response dict, or None on failure.
    """
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            r = client.get(f"{API_BASE}{path}", params=params)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return None
    except httpx.HTTPStatusError as e:
        try:
            return e.response.json()
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def _api_post(path: str, json_body: dict) -> dict | None:
    """Send a POST request to the API.

    Args:
        path: API path (e.g. '/chat').
        json_body: Request body.

    Returns:
        JSON response dict, or None on failure.
    """
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            r = client.post(f"{API_BASE}{path}", json=json_body)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return None
    except httpx.HTTPStatusError as e:
        try:
            return e.response.json()
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def _api_post_bytes(path: str, json_body: dict) -> bytes | None:
    """Send a POST request expecting binary response (PDF).

    Args:
        path: API path.
        json_body: Request body.

    Returns:
        Response bytes, or None on failure.
    """
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            r = client.post(f"{API_BASE}{path}", json=json_body)
            r.raise_for_status()
            if "application/pdf" in r.headers.get("content-type", ""):
                return r.content
            return None
    except Exception:
        return None


# ── Session state init ────────────────────────────────────────────────


def _init_state() -> None:
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "recent_reports" not in st.session_state:
        st.session_state.recent_reports = []
    if "campaigns_cache" not in st.session_state:
        st.session_state.campaigns_cache = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def _load_campaigns() -> list[dict]:
    """Fetch all campaigns from the API (cached in session state).

    Returns:
        List of campaign dicts.
    """
    if st.session_state.campaigns_cache is not None:
        return st.session_state.campaigns_cache

    result = _api_get("/campaigns", {"limit": 100})
    if result and "campaigns" in result:
        st.session_state.campaigns_cache = result["campaigns"]
        return result["campaigns"]
    return []


# ── Sidebar ───────────────────────────────────────────────────────────


def _render_sidebar() -> None:
    """Render the sidebar with quick actions and system status."""
    with st.sidebar:
        st.markdown("## 📊 Campaign Intelligence")
        st.caption("AI-powered campaign analytics")
        st.markdown("---")

        # ── System Status ─────────────────────────────────────────
        health = _api_get("/health")
        if health and health.get("status") == "healthy":
            st.markdown(
                '<span class="status-dot status-green"></span> System Online',
                unsafe_allow_html=True,
            )
        elif health:
            st.markdown(
                '<span class="status-dot status-yellow"></span> System Degraded',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot status-red"></span> API Offline',
                unsafe_allow_html=True,
            )
            st.error(
                "Cannot connect to API. Start the server with `uvicorn app.main:app --port 8080`"
            )

        st.markdown("---")

        # ── Generate Report ───────────────────────────────────────
        st.markdown("### Generate Report")
        campaigns = _load_campaigns()
        campaign_names = {c["campaign_name"]: c["id"] for c in campaigns}

        if campaign_names:
            selected_campaign = st.selectbox(
                "Select campaign",
                options=list(campaign_names.keys()),
                key="report_campaign_select",
            )
            report_format = st.selectbox(
                "Format",
                options=["markdown", "pdf", "slack"],
                key="report_format_select",
            )

            if st.button("Generate Report", key="btn_generate_report"):
                cid = campaign_names[selected_campaign]
                if report_format == "pdf":
                    with st.spinner("Generating PDF..."):
                        pdf_data = _api_post_bytes(
                            "/reports/generate",
                            {"campaign_id": cid, "format": "pdf"},
                        )
                    if pdf_data:
                        st.download_button(
                            "Download PDF",
                            data=pdf_data,
                            file_name=f"report_{selected_campaign.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                        )
                        st.session_state.recent_reports.append(
                            {
                                "name": selected_campaign,
                                "format": "pdf",
                                "time": datetime.now(timezone.utc).strftime("%H:%M"),
                            }
                        )
                    else:
                        st.error("PDF generation failed.")
                else:
                    with st.spinner("Generating report..."):
                        result = _api_post(
                            "/reports/generate",
                            {"campaign_id": cid, "format": report_format},
                        )
                    if result and "content" in result:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["content"],
                                "tools": ["generate_lci_report"],
                            }
                        )
                        st.session_state.recent_reports.append(
                            {
                                "name": selected_campaign,
                                "format": report_format,
                                "time": datetime.now(timezone.utc).strftime("%H:%M"),
                            }
                        )
                        st.rerun()
                    elif result and "error" in result:
                        st.error(result["error"])
                    else:
                        st.error("Report generation failed.")
        else:
            st.caption("No campaigns available. Is the database seeded?")

        st.markdown("---")

        # ── Compare Campaigns ─────────────────────────────────────
        st.markdown("### Compare Campaigns")
        if len(campaign_names) >= 2:
            names = list(campaign_names.keys())
            camp_a = st.selectbox("Campaign A", options=names, key="compare_a")
            camp_b = st.selectbox(
                "Campaign B",
                options=[n for n in names if n != camp_a],
                key="compare_b",
            )
            if st.button("Compare", key="btn_compare"):
                with st.spinner("Comparing campaigns..."):
                    result = _api_post(
                        "/reports/compare",
                        {
                            "campaign_id_1": campaign_names[camp_a],
                            "campaign_id_2": campaign_names[camp_b],
                        },
                    )
                if result and "markdown_report" in result:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["markdown_report"],
                            "tools": ["compare_campaigns"],
                        }
                    )
                    st.rerun()
                elif result and "error" in result:
                    st.error(result["error"])
                else:
                    st.error("Comparison failed.")
        else:
            st.caption("Need at least 2 campaigns to compare.")

        st.markdown("---")

        # ── Recommend Audience ────────────────────────────────────
        st.markdown("### Recommend Audience")
        audience_desc = st.text_input(
            "Describe target audience",
            placeholder="e.g. lunch-time diners in Texas",
            key="audience_input",
        )
        audience_vertical = st.selectbox(
            "Vertical (optional)",
            options=["", "QSR", "Automotive", "CPG", "Retail", "Entertainment"],
            key="audience_vertical",
        )
        if st.button("Get Recommendations", key="btn_audience"):
            if audience_desc:
                with st.spinner("Generating recommendations..."):
                    body = {"description": audience_desc}
                    if audience_vertical:
                        body["vertical"] = audience_vertical
                    result = _api_post("/audience/recommend", body)
                if result and "recommendation" in result:
                    rec = result["recommendation"]
                    # Format as readable text
                    lines = [f"**Audience Recommendations** for: _{audience_desc}_\n"]
                    if "overall_strategy" in rec:
                        lines.append(f"**Strategy:** {rec['overall_strategy']}\n")
                    for seg in rec.get("recommended_segments", []):
                        conf = seg.get("confidence", 0)
                        lines.append(
                            f"- **{seg['segment_name']}** (confidence: {conf:.0%})\n"
                            f"  {seg.get('rationale', '')}"
                        )
                    if rec.get("segments_to_avoid"):
                        lines.append(
                            f"\n**Avoid:** {', '.join(rec['segments_to_avoid'])}"
                        )
                    content = "\n".join(lines)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tools": ["recommend_audience"],
                        }
                    )
                    st.rerun()
                elif result and "error" in result:
                    st.error(result["error"])
                else:
                    st.error("Recommendation failed.")
            else:
                st.warning("Enter a target audience description.")

        st.markdown("---")

        # ── Recent Reports ────────────────────────────────────────
        st.markdown("### Recent Reports")
        recent = st.session_state.get("recent_reports", [])
        if recent:
            for rpt in reversed(recent[-5:]):
                st.markdown(
                    f"<small>{rpt['time']} — {rpt['name']} ({rpt['format']})</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No reports generated yet.")


# ── Example queries ───────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "How did Dunkin's Q4 campaign perform?",
    "Compare our best QSR campaigns",
    "Generate LCI report for Toyota RAV4 campaign",
    "What audience should I use for a CPG lunch-time campaign?",
    "Which vertical had the highest ROAS last quarter?",
]


def _render_example_chips() -> None:
    """Render clickable example query chips."""
    cols = st.columns(len(EXAMPLE_QUERIES))
    for i, query in enumerate(EXAMPLE_QUERIES):
        with cols[i]:
            if st.button(query, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_query = query


# ── Chat display ──────────────────────────────────────────────────────


def _render_chat_history() -> None:
    """Render all messages in the chat history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show tool badges for assistant messages
            if msg["role"] == "assistant" and msg.get("tools"):
                badges = " ".join(
                    f'<span class="tool-badge">{t}</span>' for t in msg["tools"]
                )
                st.markdown(
                    f'<div style="margin-top: 2px">{badges}</div>',
                    unsafe_allow_html=True,
                )


def _process_query(query: str) -> None:
    """Send a query to the chat API and display the response.

    Args:
        query: The user's natural language query.
    """
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing campaign data..."):
            result = _api_post(
                "/chat",
                {
                    "message": query,
                    "conversation_id": st.session_state.conversation_id,
                },
            )

        if result is None:
            error_msg = (
                "Unable to connect to the API. Please ensure the backend is running:\n\n"
                "```\nuvicorn app.main:app --port 8080\n```"
            )
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
            return

        if "error" in result:
            detail = result.get("detail", result.get("error", "Unknown error"))
            st.error(f"Request failed: {detail}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {detail}"}
            )
            return

        # Successful response
        response_text = result.get("response", "No response received.")
        tools_used = result.get("tools_used", [])
        processing_ms = result.get("processing_time_ms", 0)

        # Update conversation ID
        if result.get("conversation_id"):
            st.session_state.conversation_id = result["conversation_id"]

        st.markdown(response_text)

        # Tool badges
        if tools_used:
            badges = " ".join(
                f'<span class="tool-badge">{t}</span>' for t in tools_used
            )
            st.markdown(
                f'<div style="margin-top: 2px">{badges}</div>',
                unsafe_allow_html=True,
            )

        # Timing
        if processing_ms:
            st.caption(f"Processed in {processing_ms:,}ms")

        # Store in history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "tools": tools_used,
            }
        )


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit application."""
    _init_state()
    _render_sidebar()

    # Header
    st.markdown("## Campaign Intelligence Assistant")
    st.caption(
        "Ask questions about campaign performance, generate reports, or get recommendations."
    )

    # Example chips (only show if no messages yet)
    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        _render_example_chips()
        st.markdown("---")

    # Chat history
    _render_chat_history()

    # Handle pending query from example chips
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        _process_query(query)
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about campaign performance..."):
        _process_query(prompt)
        st.rerun()

    # Footer
    st.markdown(
        '<div class="footer-badge">Powered by AI &middot; Campaign Intelligence Assistant v0.1</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
