import os
import logging
import warnings
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Suppress noisy HuggingFace/sentence-transformers warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")


KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge"
INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "faiss_index"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_vector_store() -> FAISS:
    """Load knowledge documents, chunk them, and create a FAISS vector store."""
    if INDEX_DIR.exists():
        return FAISS.load_local(
            str(INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    loader = DirectoryLoader(
        str(KNOWLEDGE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n---", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))

    return vector_store


def get_retriever(k: int = 5):
    """Return a retriever over the medical knowledge base."""
    vector_store = build_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
