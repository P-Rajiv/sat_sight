# agents.py
import json
from graph_state import SatSightState
from llm_utils import get_llm_pipeline

# You can replace model_name with your preferred one
PLANNER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
REASONER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

planner_llm = get_llm_pipeline(PLANNER_MODEL, max_new_tokens=128)
reasoner_llm = get_llm_pipeline(REASONER_MODEL, max_new_tokens=512)

def planner_node(state: SatSightState) -> SatSightState:
    prompt = f"""
        You are the Planner for SatSight.
        User query:
        {state.user_query}
        Decide how to handle this query. Return a JSON object with:
        - "plan": a short textual plan
        - (optionally) other flags if needed
        For example:
        {{ "plan": "simple_reasoning" }}
        """
    raw = planner_llm(prompt)[0]["generated_text"]
    try:
        info = json.loads(raw.strip())
    except json.JSONDecodeError:
        info = {}

    state.planner_plan = info.get("plan", "simple_reasoning")
    return state

def reasoner_node(state: SatSightState) -> SatSightState:
    prompt = f"""
        You are the Reasoner for SatSight.
        User query:
        {state.user_query}
        Based on plan: {state.planner_plan}
        Provide a reasoned answer.
        """
    out = reasoner_llm(prompt)[0]["generated_text"]
    state.answer = out.strip()
    state.confidence = 0.9
    return state
