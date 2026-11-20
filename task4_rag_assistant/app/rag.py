
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# PATHS AND ENVIRONMENT VARIABLES
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")

PDF_PATH = ROOT_DIR / "task4_rag_assistant" / "data" / "product_manual.pdf"

CHROMA_PATH = ROOT_DIR / "task4_rag_assistant" / "chroma_db"

EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text"
)

OLLAMA_BASE_URL = "http://localhost:11434"


# --------------------------------------------------
# 1. INITIALIZE EMBEDDING MODEL
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)


# --------------------------------------------------
# 2. LOAD PDF DOCUMENT
# --------------------------------------------------

def load_pdf():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))

    documents = loader.load()

    print(f"PDF loaded successfully: {PDF_PATH.name}")
    print(f"Total PDF pages: {len(documents)}")

    return documents


# --------------------------------------------------
# 3. SPLIT DOCUMENT INTO CHUNKS
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    print(f"Total text chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# 4. CREATE CHROMA VECTOR DATABASE
# --------------------------------------------------

def create_vector_database():

    print("\nCreating RAG vector database...")

    documents = load_pdf()

    chunks = split_documents(documents)

    # Rebuild the database to avoid duplicate chunks
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    vector_database = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_name="support_manual"
    )

    print("Chroma vector database created successfully.")

    return vector_database


# --------------------------------------------------
# 5. LOAD EXISTING VECTOR DATABASE
# --------------------------------------------------

def get_vector_database():

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            "Chroma database not found. Build the database first."
        )

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name="support_manual"
    )


# --------------------------------------------------
# 6. RETRIEVE RELEVANT DOCUMENT CHUNKS
# --------------------------------------------------

def retrieve_documents(question, k=3):

    vector_database = get_vector_database()

    results = vector_database.similarity_search(
        question,
        k=k
    )

    return results


# --------------------------------------------------
# 7. GENERATE ANSWER USING RETRIEVED CONTEXT
# --------------------------------------------------

def generate_answer(question, documents):

    import requests

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY the context provided below.

STRICT RULES:
1. Use only information explicitly available in the context.
2. Do not use outside knowledge.
3. Do not invent product features, prices, policies, or instructions.
4. If the answer is not available in the context,
   respond exactly with: I don't know
5. Give a clear and concise answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        os.getenv("OLLAMA_URL"),
        json={
            "model": os.getenv("OLLAMA_MODEL"),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


# --------------------------------------------------
# 8. ASK QUESTION AND RETURN SOURCES
# --------------------------------------------------

def ask_question(question):

    documents = retrieve_documents(question, k=3)

    if not documents:
        return {
            "answer": "I don't know",
            "sources": []
        }

    answer = generate_answer(question, documents)

    sources = []

    for document in documents:

        page_number = document.metadata.get("page", 0) + 1

        source = {
            "file": PDF_PATH.name,
            "page": page_number
        }

        if source not in sources:
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources
    }


# --------------------------------------------------
# 9. BUILD DATABASE WHEN RUN DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    create_vector_database()

    print("\nRAG document processing completed.")
