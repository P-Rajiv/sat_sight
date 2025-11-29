# vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # or any embedding model
from langchain_core.documents import Document

def build_vector_store(documents: List[Document], embeddings_model=None, **kwargs):
    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings()
    vs = FAISS.from_documents(documents, embeddings_model, **kwargs)
    return vs
