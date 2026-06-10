# app.py — Streamlit UI for Finance RAG

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ── Config ───────────────────────────────────────────────────
load_dotenv()
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "llama3")
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Finance RAG Assistant",
    page_icon  = "💰",
    layout     = "centered"
)

# ── Header ───────────────────────────────────────────────────
st.title("💰 Finance RAG Assistant")
st.caption("Ask anything about stock markets & equity analysis — powered by SEBI, NSE & CFA documents")
st.divider()

# ── Load models (cached so they load only once) ──────────────
@st.cache_resource
def load_rag_chain():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # FAISS
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = OllamaLLM(model=OLLAMA_MODEL)

    # Prompt
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ── Check FAISS exists ───────────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("❌ FAISS index not found. Please run `python ingest.py` first.")
    st.stop()

# ── Load chain ───────────────────────────────────────────────
with st.spinner("Loading models... (first load takes ~30 seconds)"):
    chain, retriever = load_rag_chain()

# ── Sample questions ─────────────────────────────────────────
st.markdown("**💡 Try asking:**")
sample_questions = [
    "What is fundamental analysis?",
    "What is the role of SEBI?",
    "How is P/E ratio calculated?",
    "What is the difference between primary and secondary market?",
    "What is DCF valuation?",
]
cols = st.columns(2)
for i, q in enumerate(sample_questions):
    if cols[i % 2].button(q, use_container_width=True):
        st.session_state["prefill"] = q

st.divider()

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['file']}** — page {src['page']}")

# ── Handle prefill from sample buttons ───────────────────────
prefill = st.session_state.pop("prefill", None)

# ── Chat input ───────────────────────────────────────────────
question = st.chat_input("Ask a question about equity & stock markets...")

if prefill:
    question = prefill

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            answer = chain.invoke(question)
            docs   = retriever.invoke(question)

        st.markdown(answer)

        # Sources
        sources = []
        seen = set()
        for doc in docs:
            src  = doc.metadata.get("source", "unknown")
            pg   = doc.metadata.get("page",   "?")
            key  = f"{src}:{pg}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": os.path.basename(src),
                    "page": pg
                })

        with st.expander("📎 Sources used"):
            for src in sources:
                st.markdown(f"- **{src['file']}** — page {src['page']}")

    # Save to history
    st.session_state.messages.append({
        "role"    : "assistant",
        "content" : answer,
        "sources" : sources
    })

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Finance RAG Assistant**

Answers questions using official financial documents.

**Knowledge Base:**
- NSE Basics of Financial Markets
- SEBI Financial Education Booklet
- SEBI Beginner's Guide to Capital Markets
- SEBI Securities Market Booklet
- CFA Equity Valuation
- SEBI Financial Education Part A
- SEBI College Students Booklet

**Stack:**
- LangChain (LCEL)
- FAISS vector database
- HuggingFace Embeddings
- Ollama + Llama 3
""")
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()