# rag.py — Modern LangChain (LCEL style, no deprecated chains)

import os
import sys
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "llama3")
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# ── Check FAISS index exists ─────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    print("❌ FAISS index not found. Run:  python ingest.py  first!")
    sys.exit(1)

# ── Load embeddings ──────────────────────────────────────────
print(f"🤖 Loading embeddings: {EMBED_MODEL}")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ── Load FAISS ───────────────────────────────────────────────
print(f"📂 Loading FAISS index...")
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("   ✅ Vector store ready")

# ── Load Ollama ──────────────────────────────────────────────
print(f"🦙 Connecting to Ollama: {OLLAMA_MODEL}")
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model=OLLAMA_MODEL)
print("   ✅ LLM ready")

# ── Prompt ───────────────────────────────────────────────────
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = PromptTemplate.from_template("""
You are a financial education assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the documents to answer this."
Do not make up information.

Context:
{context}

Question:
{question}

Answer:""")

# ── Helper to format retrieved docs ─────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Build chain (LCEL style) ─────────────────────────────────
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Chat loop ────────────────────────────────────────────────
print("\n" + "="*55)
print("💰 Finance RAG — Equity Analysis Assistant")
print("   Ask anything about stock markets & equity.")
print("   Type  'exit'  to quit.")
print("="*55 + "\n")

while True:
    question = input("You: ").strip()

    if not question:
        continue
    if question.lower() in ["exit", "quit", "q"]:
        print("Bye! 👋")
        break

    print("\n🔍 Searching documents...\n")

    # Get answer
    answer = chain.invoke(question)
    print(f"Assistant: {answer}")

    # Show sources separately
    docs = retriever.invoke(question)
    print("\n📎 Sources used:")
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        pg  = doc.metadata.get("page",   "?")
        key = f"{src}:{pg}"
        if key not in seen:
            seen.add(key)
            print(f"   • {os.path.basename(src)}  (page {pg})")
    print()