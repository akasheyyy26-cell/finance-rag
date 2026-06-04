import os
import sys

print("✅ Script started")

# ── Check pdfs folder ────────────────────────────────────────
if not os.path.exists("pdfs"):
    print("❌ ERROR: 'pdfs/' folder does not exist")
    print("   Fix: run   mkdir pdfs   in your terminal")
    sys.exit(1)

pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
print(f"📁 Found {len(pdf_files)} PDF files in pdfs/:")
for f in pdf_files:
    size = os.path.getsize(f"pdfs/{f}")
    print(f"   {f}  →  {size} bytes")

if len(pdf_files) == 0:
    print("❌ ERROR: No PDFs found in pdfs/ folder")
    print("   Fix: re-run the curl download commands")
    sys.exit(1)

print("\n⬇️  Importing LangChain...")
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    print("   ✅ PyPDFDirectoryLoader OK")
except Exception as e:
    print(f"   ❌ {e}")
    sys.exit(1)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("   ✅ RecursiveCharacterTextSplitter OK")
except Exception as e:
    print(f"   ❌ {e}")
    sys.exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("   ✅ HuggingFaceEmbeddings OK")
except Exception as e:
    print(f"   ❌ {e}")
    sys.exit(1)

try:
    from langchain_community.vectorstores import FAISS
    print("   ✅ FAISS OK")
except Exception as e:
    print(f"   ❌ {e}")
    sys.exit(1)

print("\n📄 Loading PDFs...")
loader = PyPDFDirectoryLoader("pdfs")
documents = loader.load()
print(f"   ✅ Loaded {len(documents)} pages")

if len(documents) == 0:
    print("❌ ERROR: PDFs found but 0 pages loaded")
    print("   The PDF files may be empty or corrupted")
    print("   Fix: re-download them using the curl commands")
    sys.exit(1)

print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"   ✅ {len(chunks)} chunks created")

print("\n🤖 Loading embedding model (may take 2-3 min first time)...")
from dotenv import load_dotenv
load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
print("   ✅ Embeddings ready")

print("\n💾 Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("   ✅ Saved to faiss_index/")

print("\n🎉 DONE! Now run:  python rag.py")