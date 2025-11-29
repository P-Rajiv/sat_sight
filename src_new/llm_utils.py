# llm_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def get_llm_pipeline(model_name: str, max_new_tokens: int = 256):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # You may need adjust model_kwargs depending on your setup
    llm = pipeline("text-generation", model=model, tokenizer=tok,
                   model_kwargs={"max_new_tokens": max_new_tokens})
    return llm
