# graph_state.py
from pydantic import BaseModel
from typing import Optional, List, Any

class SatSightState(BaseModel):
    # Input
    image_path: Optional[str] = None
    user_query: Optional[str] = None

    # Retrieval results
    retrieved_docs: Optional[List[Any]] = None  # e.g. list of dicts { 'id', 'text', ... }
    fused_context: Optional[str] = None

    # Plan / metadata
    planner_plan: Optional[str] = None

    # Output
    answer: Optional[str] = None
    confidence: Optional[float] = None
