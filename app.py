import streamlit as st
from pathlib import Path
import os
import asyncio
import turso_serverless
from dotenv import load_dotenv

from agent import run
from agents import call_report_agent
from ingest import ingest_files

load_dotenv()

# --- Event loop fix for Streamlit threads (Turso + ChromaDB) ---
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(
    page_title="CampusGPT — Multi-Agent Study Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load CSS ---
css_file = Path(__file__).parent / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper: fetch recent rows from Turso ---
def fetch_recent_entries(limit=10):
    try:
        conn = turso_serverless.connect(
            os.getenv("TURSO_DATABASE_URL"),
            auth_token=os.getenv("TURSO_AUTH_TOKEN"),
        )
        cursor = conn.execute(
            "SELECT timestamp, subject, topic, question FROM study_log "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return [("Error", str(e), "", "")]

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🎓 CampusGPT")
    st.caption("Multi-Agent Study Platform")

    st.markdown("---")

    # Quick Action: Report
    st.markdown("### ⚡ Quick Actions")
    if st.button("📊 Generate Weekly Report", use_container_width=True):
        with st.spinner("Report Analyst is compiling..."):
            report = call_report_agent.invoke({"query": "Generate my weekly study report"})
        st.session_state["latest_report"] = report

    if "latest_report" in st.session_state:
        with st.expander("📄 Latest Report", expanded=True):
            st.markdown(st.session_state["latest_report"])

    st.markdown("---")

    # PDF Upload + Auto Ingest
    st.markdown("### 📤 Upload Study Material")
    uploaded_files = st.file_uploader(
        "Add PDFs to the knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        kb_dir = Path("knowledge_base")
        kb_dir.mkdir(exist_ok=True)

        saved_any = False
        for uploaded in uploaded_files:
            target = kb_dir / uploaded.name
            if target.exists():
                st.warning(f"{uploaded.name} already exists — replacing.")
            target.write_bytes(uploaded.getbuffer())
            st.success(f"Saved {uploaded.name}")
            saved_any = True

        if saved_any and st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            with st.spinner("Ingesting PDFs into ChromaDB..."):
                try:
                    ingest_files(str(kb_dir))
                    st.success("Knowledge base rebuilt. New PDFs are now searchable.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("### 🧠 Architecture")
    st.markdown("""
    - **Supervisor** routes
    - **Study Assistant** logs + RAG
    - **Report Analyst** summarizes
    """)

    st.caption(f"Session questions: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")

# --- Main Layout: Two Columns ---
col_chat, col_data = st.columns([2, 1])

# === Left: Chat ===
with col_chat:
    st.markdown("""
    <div class="hero">
        <h1>CampusGPT</h1>
        <p>A multi-agent platform that answers academic questions, logs your study trail, and generates weekly reports.</p>
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a study question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Routing..."):
                response = run(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        # Force refresh so the data panel updates
        st.rerun()

# === Right: Live Data Panel ===
with col_data:
    st.markdown("### 📊 Live Study Log")
    st.caption("Rows written to Turso by the Study Assistant agent")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    rows = fetch_recent_entries(limit=10)

    if not rows or rows[0][0] == "Error":
        st.warning(f"Could not load data: {rows[0][1] if rows else 'No rows'}")
    else:
        for ts, subject, topic, question in rows:
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(30,30,50,0.6); border-left: 3px solid #a78bfa;
                            padding: 10px 14px; border-radius: 10px; margin-bottom: 8px;">
                    <div style="font-size: 0.75rem; color: #9ca3af;">{ts}</div>
                    <div style="font-weight: 600; color: #c4b5fd;">{subject} · {topic}</div>
                    <div style="font-size: 0.85rem; color: #d1d5db;">{question[:100]}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Every question is logged by the agent, not by you.")