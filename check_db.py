import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=768,
)
db = Chroma(
    collection_name="campus_kb",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

data = db.get()
print(f"Total chunks: {len(data['documents'])}\n")
for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
    print(f"[{i}] Source: {meta.get('source')}")
    print(doc[:300].replace("\n", " "))
    print("---")

# Search for babalu specifically
print("\n=== Searching for 'babalu' ===")
results = db.similarity_search("babalu", k=4)
for i, doc in enumerate(results, 1):
    print(f"[{i}] Source: {doc.metadata.get('source')}")
    print(doc.page_content[:300])
    print("---")