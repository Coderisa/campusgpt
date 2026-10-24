"""
Worker agents for CampusGPT multi-agent system.
Each worker is a self-contained agent with its own prompt and tools.
They are wrapped as tools so the supervisor can call them.
"""
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from tools import search_knowledge_base, log_question, generate_weekly_report

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3,
)

# --- Study Assistant: always searches RAG, logs only academic questions ---
study_agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, log_question],
    system_prompt=(
        "You are a study assistant with access to university course notes and uploaded PDFs.\n\n"
        "RULES:\n"
        "1. For EVERY question, ALWAYS call `search_knowledge_base` FIRST to look in the knowledge base.\n"
        "2. The retrieved content is the source of truth. If ANY chunk contains information related "
        "to the question—even loosely, even with spelling variations, even if it's not academic—"
        "USE IT to answer. The knowledge base contains both academic notes AND personal documents "
        "uploaded by the user. Do NOT dismiss a chunk as 'unrelated' just because it's not academic.\n"
        "3. ONLY call `log_question` if the question is clearly academic (DBMS, OS, CN, AI, faculty, "
        "syllabus, academic rules). For casual questions (names, greetings, stories, random facts), do NOT log.\n"
        "4. When you log, extract the subject and topic accurately and give a short reasoning.\n"
        "5. Always mention what you did: whether you searched, and whether you logged."
    ),
)

# --- Report Analyst: weekly reports only ---
report_agent = create_agent(
    model=llm,
    tools=[generate_weekly_report],
    system_prompt=(
        "You are a report analyst. When the user asks for a study report, weekly "
        "summary, or learning analysis, call generate_weekly_report and present "
        "the result clearly."
    ),
)

# --- Wrap workers as tools for the supervisor ---
@tool(
    "study_assistant",
    description=(
        "Handles ALL factual and academic questions. It always searches the uploaded PDFs "
        "and only logs academic questions to the database. Use this for any question about "
        "DBMS, OS, CN, AI, faculty, syllabus, rules, names, stories, or general facts."
    ),
)
def call_study_agent(query: str) -> str:
    result = study_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool(
    "report_analyst",
    description=(
        "Generates weekly study reports and learning summaries from the logged questions. "
        "Use this ONLY when the user asks for a report, summary, or analysis of study activity."
    ),
)
def call_report_agent(query: str) -> str:
    result = report_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content