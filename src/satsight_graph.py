from langgraph.graph import StateGraph, END
from graph_state import SatSightState
from nodes import (
    vision_encoder_node, image_retriever_node,
    text_retriever_node, fusion_node, reasoning_node, mcp_node
)

def build_graph():
    g = StateGraph(SatSightState)
    g.add_node("vision_encoder", vision_encoder_node)
    g.add_node("image_retriever", image_retriever_node)
    g.add_node("text_retriever", text_retriever_node)
    g.add_node("fusion", fusion_node)
    g.add_node("reasoning", reasoning_node)
    g.add_node("mcp", mcp_node)

    g.add_edge("vision_encoder", "image_retriever")
    g.add_edge("image_retriever", "text_retriever")
    g.add_edge("text_retriever", "fusion")
    g.add_edge("fusion", "reasoning")
    g.add_conditional_edges(
        "reasoning",
        lambda state: "mcp" if state.mcp_needed else END,
        {"mcp": "mcp", END: END}
    )
    g.set_entry_point("vision_encoder")
    return g
