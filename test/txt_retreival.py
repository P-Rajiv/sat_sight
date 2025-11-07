from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                            model_kwargs={"device":"cuda"},
                            encode_kwargs={"normalize_embeddings":True})

db = Chroma(persist_directory="data/chroma_store", embedding_function=emb)
docs = db.similarity_search("Why do deltas experience flooding?", k=3)
for i,d in enumerate(docs,1):
    print(f"{i}. {d.page_content[:150]}...")