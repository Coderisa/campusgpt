import os
import asyncio
import turso_serverless
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "campus_kb"


# ---------- Tool 1: RAG retrieval (event loop fixed) ----------
@tool
def search_knowledge_base(query: str) -> str:
    """Search university course notes, syllabus, faculty info, and academic rules.
    Use this for any question about course content, faculty details, or exam preparation."""
    # Create a fresh event loop for this worker thread (Python 3.12 fix)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "No relevant documents found in the knowledge base."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(results)
    finally:
        loop.close()


# ---------- Turso helper ----------
def _get_conn():
    return turso_serverless.connect(
        os.getenv("TURSO_DATABASE_URL"),
        auth_token=os.getenv("TURSO_AUTH_TOKEN"),
    )


# ---------- Tool 2: Log a question ----------
@tool
def log_question(subject: str, topic: str, question: str, reasoning: str) -> str:
    """Log an academic question to the Turso database.
    Call this whenever the user asks a course-related question.
    Extract the subject (e.g., DBMS, OS, CN, AI) and topic (e.g., Normalization, Deadlock)
    from the question. Also provide a short reasoning explaining why you classified it as academic."""
    try:
        conn = _get_conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS study_log ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT, subject TEXT, topic TEXT, question TEXT, agent_reasoning TEXT)"
        )
        conn.execute(
            "INSERT INTO study_log (timestamp, subject, topic, question, agent_reasoning) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), subject, topic, question, reasoning)
        )
        conn.commit()
        conn.close()
        return f"Logged: {subject} / {topic}"
    except Exception as e:
        return f"Logging failed: {e}"


# ---------- Tool 3: Weekly report ----------
@tool
def generate_weekly_report(days: int = 7) -> str:
    """Generate a weekly study report by querying Turso.
    Use this when the user asks for a study report, weekly summary, or learning analysis."""
    try:
        conn = _get_conn()
        cutoff = datetime.now() - timedelta(days=days)
        cursor = conn.execute(
            "SELECT subject, topic FROM study_log WHERE timestamp >= ?",
            (cutoff.strftime("%Y-%m-%d %H:%M"),)
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return f"No study records found in the last {days} days."

        subject_counts = {}
        topic_counts = {}
        for s, t in rows:
            subject_counts[s] = subject_counts.get(s, 0) + 1
            topic_counts[t] = topic_counts.get(t, 0) + 1

        top_subjects = sorted(subject_counts.items(), key=lambda x: -x[1])[:3]
        top_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:3]

        report = f"Weekly Study Report ({days} days)\nTotal questions: {len(rows)}\n\nTop subjects:\n"
        report += "\n".join(f"  - {s}: {c}" for s, c in top_subjects)
        report += "\n\nTop topics:\n"
        report += "\n".join(f"  - {t}: {c}" for t, c in top_topics)
        return report
    except Exception as e:
        return f"Report generation failed: {e}"