# satsight_graph.py
from langgraph.graph import StateGraph, END
from graph_state import SatSightState
from agents import planner_node
from retrieval_agent import retrieval_node
from agents import reasoner_node  # same as before

def build_rag_graph():
    g = StateGraph(SatSightState)
    g.add_node("planner", planner_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("reasoner", reasoner_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "retrieval")
    g.add_edge("retrieval", "reasoner")
    g.add_edge("reasoner", END)

    return g
