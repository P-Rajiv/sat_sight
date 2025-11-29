# retrieval_agent.py
from graph_state import SatSightState
from vector_store import build_vector_store
from typing import List, Dict

# You should initialize once with your docs; for demo we assume it's ready
# e.g. vs = build_vector_store(my_docs)

# But for simplicity in this example, assume you already have a global retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Example: load & build index (do once offline ideally)
def init_global_retriever(documents: List[Dict]):
    """
    documents: list of dict {'id':..., 'text':...}
    """
    docs = [Document(page_content=d['text'], metadata={'id': d['id']}) for d in documents]
    return FAISS.from_documents(docs, OpenAIEmbeddings())

RETRIEVER = None

def retrieval_node(state: SatSightState) -> SatSightState:
    global RETRIEVER
    if RETRIEVER is None:
        # load your metadata documents from storage
        docs = load_my_documents()  # you define this
        RETRIEVER = init_global_retriever(docs)

    query = state.user_query or ""
    results = RETRIEVER.similarity_search(query, k=5)  # top-5
    # each result is a Document, with .page_content and .metadata
    state.retrieved_docs = [{
        'id': doc.metadata.get('id'),
        'text': doc.page_content
    } for doc in results]

    # build fused context as a concatenated string
    context_texts = "\n\n".join([d['text'] for d in state.retrieved_docs])
    state.fused_context = f"Relevant documents:\n{context_texts}\n\nUser query: {state.user_query}"
    return state
