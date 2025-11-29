# src/nodes.py
import faiss, pickle, torch, numpy as np
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModelForCausalLM, AutoTokenizer, pipeline
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from graph_state import SatSightState
import requests

# ---- cache heavy objects once ----
_CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
_CLIP_PROC  = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
_LLM_PIPE   = "meta-llama/Meta-Llama-3-8B-Instruct"
_EMB        = "BAAI/bge-large-en-v1.5"
_DB         = "data/chroma_store"
_INDEX      = "data/image_index.faiss"
_META       = "data/image_meta.pkl" 
_PLANNER_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct" 
_REASONER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" 
_JUDGE_MODEL    = "meta-llama/Meta-Llama-3-8B-Instruct"
_VISION_VLM     = "some-rs-vlm-or-llava"  # if you add a VLM

_PLANNER_PIPE   = None
_REASONER_PIPE  = None
_JUDGE_PIPE     = None
_VISION_PIPE    = None



def _clip():
    global _CLIP_MODEL, _CLIP_PROC
    _CLIP_MODEL = CLIPModel.from_pretrained(_CLIP_MODEL)
    _CLIP_PROC  = CLIPProcessor.from_pretrained(_CLIP_PROC)
    return _CLIP_MODEL, _CLIP_PROC

def _llm():
    global _LLM_PIPE
    model_id = _LLM_PIPE
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    _LLM_PIPE = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512)
    return _LLM_PIPE

def _textdb():
    global _EMB, _DB
    _EMB = HuggingFaceEmbeddings(
        model_name=_EMB,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    _DB = Chroma(persist_directory=_DB, embedding_function=_EMB)
    return _DB

def _image_index():
    global _INDEX, _META
    _INDEX = faiss.read_index(_INDEX)
    _META = pickle.load(open(_META, "rb"))
    return _INDEX, _META

# ---------- NODES (take/return SatSightState) ----------

def vision_encoder_node(state: SatSightState) -> SatSightState:
    clip_model, clip_proc = _clip()
    img = Image.open(state.image_path).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    state.image_embedding = emb.cpu().numpy().astype("float32")
    return state

def image_retriever_node(state: SatSightState) -> SatSightState:
    index, meta = _image_index()
    D, I = index.search(state.image_embedding, k=3)
    results = [meta[i] | {"score": float(D[0][idx])} for idx, i in enumerate(I[0])]
    state.retrieved_images = results
    return state

def text_retriever_node(state: SatSightState) -> SatSightState:
    db = _textdb()
    meta0 = state.retrieved_images[0]
    q = state.user_query or "Describe this region."
    region = meta0.get("region_hint", "")
    biome = meta0.get("biome", "")
    tags = ", ".join(meta0.get("tags", []))
    constructed = f"{q} Region: {region}. Biome: {biome}. Tags: {tags}."
    docs = db.similarity_search(constructed, k=3)
    state.constructed_query = constructed
    state.retrieved_texts = [d.page_content for d in docs]
    return state

def fusion_node(state: SatSightState) -> SatSightState:
    imgs = " ".join([m["caption"] + " " + " ".join(m["tags"]) for m in state.retrieved_images])
    texts = " ".join(state.retrieved_texts)
    state.fused_context = f"Image context: {imgs}\n\nText context: {texts}"
    return state

def reasoning_node(state: SatSightState) -> SatSightState:
    pipe = _llm()
    prompt = (
        "You are SatSight, a geospatial analyst.\n"
        f"User query: {state.user_query}\n"
        f"Context:\n{state.fused_context}\n\n"
        "Provide a concise environmental interpretation.\n"
    )
    out = pipe(prompt)[0]["generated_text"]
    state.answer = out
    scores = [m.get("score", 0.0) for m in state.retrieved_images]
    conf = float(np.clip(np.mean(scores) if scores else 0.0, 0, 1))
    state.confidence = conf
    uq = (state.user_query or "").lower()
    state.mcp_needed = (conf < 0.6) or any(w in uq for w in ["recent", "current", "today", "latest"])
    return state

# reasoning llms
def _planner_llm():
    global _PLANNER_PIPE
    if _PLANNER_PIPE is None:
        tok = AutoTokenizer.from_pretrained(_PLANNER_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            _PLANNER_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _PLANNER_PIPE = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=256)
    return _PLANNER_PIPE

def _reasoner_llm():
    global _REASONER_PIPE
    if _REASONER_PIPE is None:
        # could be same as planner model or different
        tok = AutoTokenizer.from_pretrained(__REASONER_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            __REASONER_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _REASONER_PIPE = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512)
    return _REASONER_PIPE

def _judge_llm():
    global _JUDGE_PIPE
    if _JUDGE_PIPE is None:
        tok = AutoTokenizer.from_pretrained(_JUDGE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            _JUDGE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _JUDGE_PIPE = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=384)
    return _JUDGE_PIPE

def _vision_vlm():
    global _VISION_PIPE
    if _VISION_PIPE is None:
        # load RS-LLaVA / LLaVA / Qwen2-VL etc, depending on what you choose
        _VISION_PIPE = ...  # whatever API that VLM exposes
    return _VISION_PIPE


def mcp_node(state: SatSightState) -> SatSightState:
    if not state.mcp_needed:
        return state
    meta = state.retrieved_images[0] if state.retrieved_images else {}
    lat, lon = meta.get("lat"), meta.get("lon")
    region = meta.get("region_hint", "Earth")
    print("\n[MCP] Fetching external data for:", region)
    ext_facts = []
    # Weather (Open-Meteo)
    try:
        if lat and lon:
            r = requests.get(f"http://127.0.0.1:5001/earthdata?lat={lat}&lon={lon}", timeout=10)
            data = r.json()
            forecast = data.get("forecast", [])
            ext_facts.append(f"Weather forecast (min,max °C next 3 days): {forecast}")
    except Exception as e:
        ext_facts.append(f"Weather fetch failed: {e}")
    # Wikipedia summary
    try:
        r = requests.get(f"http://127.0.0.1:5001/wiki?region={region}", timeout=10)
        data = r.json()
        ext_facts.append("Wikipedia summary: " + data.get("summary", "N/A"))
    except Exception as e:
        ext_facts.append(f"Wiki fetch failed: {e}")
    state.fused_context += "\n\n[MCP external enrichment]\n" + "\n".join(ext_facts)
    pipe = _llm()
    prompt = (
        "You are SatSight, a geospatial analyst.\n"
        f"User query: {state.user_query}\n"
        f"Updated context:\n{state.fused_context}\n\n"
        "Using the external data above, update or refine your previous interpretation."
    )
    out = pipe(prompt)[0]["generated_text"]
    state.answer = out
    return state