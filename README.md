Accessible on :https://campusina.streamlit.app/

# CampusGPT — University Study Assistant Agent

A LangGraph agent that:
- Answers course questions using RAG over your PDFs
- Autonomously logs subject/topic/reasoning to Google Sheets
- Generates weekly study reports by querying, analyzing, and writing back

## Quick Start

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in:
   - `GOOGLE_API_KEY` (from https://aistudio.google.com/)
   - `GOOGLE_SHEET_ID` (from your Sheet URL)
   - `GOOGLE_CREDENTIALS_PATH` (path to service account JSON)
3. Put PDFs in `knowledge_base/`
4. `python ingest.py` (build ChromaDB once)
5. `python agent.py --question "Explain normalization in DBMS"`

## Sample Output

(paste real output here after running)

## Evaluation

| # | Question | Type | Answer Correct? | Logged Correctly? |
|---|----------|------|-----------------|-------------------|
| 1 | Explain normalization in DBMS | Academic | ✅ | ✅ DBMS/Normalization |
| 2 | Who teaches ML? | Faculty | ✅ | ✅ Faculty/Machine Learning |
| 3 | Hi | Non-academic | ✅ | ❌ correctly NOT logged |
| ... | | | | |

**Accuracy**: X/10 correct answers, Y/10 correct logs, 0 incorrect logs.

## What Makes It an Agent

- LLM decides when to log, what subject/topic to assign, and whether retrieval is needed
- Multi-step task: weekly report = query → analyze → generate → write
- Decision logging: every log entry includes `Agent Reasoning`
- Failure handling: retrieval retries 3 times with backoff

