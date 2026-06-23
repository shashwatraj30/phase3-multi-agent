import fitz
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@tool
def extract_pdf_text(pdf_path: str) -> str:
    """Extracts and returns all text from a PDF file given its path."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text[:8000]
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def build_paper_index(pdf_path: str) -> FAISS:
    """Chunks a PDF and builds an in-memory FAISS index."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    words = full_text.split()
    chunks = []
    chunk_size = 200
    overlap = 50
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    index = FAISS.from_texts(chunks, embeddings)
    return index


def query_paper(index: FAISS, query: str, k: int = 4) -> str:
    """Queries the FAISS index and returns top k relevant chunks."""
    docs = index.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])