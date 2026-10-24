"""
Ingest PDF, Excel, Word files from knowledge_base/ into ChromaDB.
Run once before using the agent:
    python ingest.py
"""
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "campus_kb"

def ingest_files(folder_path: str = "./knowledge_base"):
    all_docs = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        loader = None

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith((".xlsx", ".xls")):
            loader = UnstructuredExcelLoader(filepath, mode="elements")
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        try:
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = filename
            all_docs.extend(pages)
            print(f"Loaded {filename}: {len(pages)} documents")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    if not all_docs:
        print("No supported files found in knowledge_base/. Add PDF, Excel, or Word files first.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
       model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
         output_dimensionality=768  # Force output to 768 dimensions
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Ingested {len(chunks)} chunks into ChromaDB at {CHROMA_DIR}")

if __name__ == "__main__":
    ingest_files()