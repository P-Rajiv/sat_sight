# src/build_text_index.py

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOC_DIR = "data/text_docs"
CHROMA_DIR = "data/chroma_store"

def main():
    # 1) load docs
    loader = DirectoryLoader(DOC_DIR, glob="*.txt", show_progress=True)
    docs = loader.load()

    # 2) split
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3) embeddings (LangChain wrapper over sentence-transformers)
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},                # or "cpu" if needed
        encode_kwargs={"normalize_embeddings": True}    # important for cosine/IP
    )

    # 4) build/store Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,               # <—— use 'embedding', not 'embedding_function'
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print(f"Chroma store built at {CHROMA_DIR} with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
