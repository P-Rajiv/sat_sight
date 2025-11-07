import pickle, faiss, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load components
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
clip_proc  = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
index = faiss.read_index("data/image_index.faiss")
meta = pickle.load(open("data/image_meta.pkl","rb"))
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                            model_kwargs={"device":"cuda"},
                            encode_kwargs={"normalize_embeddings":True})
chroma = Chroma(persist_directory="data/chroma_store", embedding_function=emb)

# 1️⃣ image embedding
img = Image.open("data/images/forest_10.jpg").convert("RGB")
inputs = clip_proc(images=img, return_tensors="pt")
with torch.no_grad():
    q = clip_model.get_image_features(**inputs)
    q = q / q.norm(p=2, dim=-1, keepdim=True)
q = q.cpu().numpy().astype("float32")

# 2️⃣ retrieve similar images
D,I = index.search(q, k=3)
image_contexts = [meta[i] for i in I[0]]
context_str = " ".join([m["caption"] + " " + " ".join(m["tags"]) for m in image_contexts])
print("\nImage context:", context_str)

# 3️⃣ construct text query
user_query = "Explain environmental risks for this region."
full_query = f"{user_query} Region: {image_contexts[0]['region_hint']}, Biome: {image_contexts[0]['biome']}, Tags: {', '.join(image_contexts[0]['tags'])}."

# 4️⃣ text retrieval
docs = chroma.similarity_search(full_query, k=3)
text_context = " ".join([d.page_content for d in docs])
print("\nText context:", text_context[:500])

# 5️⃣ ready for LLM tomorrow (we'll pass full_query + both contexts)
