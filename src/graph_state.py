from typing import Optional, Dict, Any
from pydantic import BaseModel

class SatSightState(BaseModel):
    image_path: Optional[str] = None
    user_query: Optional[str] = None
    image_embedding: Optional[Any] = None
    retrieved_images: Optional[list] = None
    constructed_query: Optional[str] = None
    retrieved_texts: Optional[list] = None
    fused_context: Optional[str] = None
    answer: Optional[str] = None
    confidence: Optional[float] = 1.0
    mcp_needed: Optional[bool] = False