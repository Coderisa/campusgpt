""" # docstring :a block of text describing the file. 
CampusGPT — Multi-Agent University Study Assistant.

Architecture:
    Supervisor (routes every message)
        ├── study_assistant  → always searches RAG, logs only academic questions
        └── report_analyst   → weekly reports

Usage:
    python agent.py --question "Explain normalization in DBMS"
    python agent.py --question "Generate my weekly study report"
    python agent.py --question "who is bablu"
"""
import os
import argparse#module-built in py : helps you read command-line arguments.
from dotenv import load_dotenv #2)function 1)module: This loads variables from a .env file.
from langchain.agents import create_agent #function :lanchain=package is a collection of modules; agents is a module
from langchain_openai import ChatOpenAI #2)Class while 1)-Package/Module

from agents import call_study_agent, call_report_agent #Functions--<snake_case

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3,
) #llm is obj passes these values to Chatcopenai class

SUPERVISOR_PROMPT = """You are the CampusGPT coordinator.

You have exactly TWO tools: `study_assistant` and `report_analyst`.
You MUST call one of them for EVERY user message.

Routing rules:
- If the user asks for a study report, weekly summary, or learning analysis
  → call `report_analyst`.
- For EVERY other message (academic questions, "who is X", "what is Y", greetings,
  names, stories, facts, chit-chat, anything) → call `study_assistant`.

The Study Assistant will search the uploaded PDFs for EVERY question and decide
whether to log it.

After the worker returns its answer, present that answer to the user EXACTLY as
received. Do NOT say "I can't answer directly." Do NOT replace the worker's answer
with your own text. Relay the worker's response verbatim.
"""
#The LLM reads this and behaves accordingly. If you change this text, the supervisor's routing behavior changes.

supervisor = create_agent(
    model=llm,
    tools=[call_study_agent, call_report_agent],
    system_prompt=SUPERVISOR_PROMPT,
)

"""Run the agent and return the final answer string."""
def run(question: str) -> str: #It takes a string (question) and returns a string."
    
    print(f"\n❓ Question: {question}\n") #Print the question to the terminal, with a blank line before and after.
    result = supervisor.invoke({"messages": [{"role": "user", "content": question}]})#Send this question to the supervisor. Wait for the full result."

    final_answer = ""
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:#"If this message has tool calls (i.e., the supervisor decided to call a worker), do the following."
            for tc in msg.tool_calls:#Loop through each tool call.
                print(f"🔀 SUPERVISOR ROUTED TO: {tc['name']}")#Print which worker was called—study_assistant or report_analyst.
                print(f"   Args: {tc['args']}")#Print the arguments passed to the worker (usually {'query': '...'}).
        elif msg.type == "tool":#If this message is a tool result (i.e., the worker returned an answer), do the following.
            preview = msg.content[:300].replace("\n", " ") #Take the first 300 characters of the worker's answer, replace newlines with spaces (for cleaner terminal output).
            print(f"📥 WORKER RETURNED: {preview}...") #Print a preview of the worker's answer.
        elif msg.type == "ai" and msg.content:  #"If this message is the AI's final answer (not a tool call), do the following."
            final_answer = msg.content #Print a separator line (60 equals signs).
            print(f"\n🤖 CampusGPT:\n{final_answer}")

    print("\n" + "=" * 60)
    return final_answer  #Return the answer string to whoever called run()—Streamlit, or the command line.


# Handles command-line arguments. When you run python agent.py --question "Explain DBMS", this function reads --question and passes its value to run().
def main(): #Define a function called main.
    parser = argparse.ArgumentParser(description="CampusGPT Multi-Agent Assistant") #Create an argument parser with a description.
    parser.add_argument("--question", required=True, help="Your question to the agent") #Expect a --question argument. It's required (the script will error if you don't provide it).
    args = parser.parse_args() #Read the actual arguments from the command line.
    run(args.question)  #Call run() with whatever the user typed after --question.


if __name__ == "__main__":
    main()
    #If this file is being run directly (not imported by another file), call main()."

#Why it matters: When Streamlit does from agent import run, this block is skipped—only run() is imported. When you run python agent.py --question "...", this block executes, calling main(). This is standard Python practice: it lets a file be both a script and an importable module.