from satsight_graph import build_graph
from graph_state import SatSightState

def main():
    g = build_graph()
    app = g.compile()

    image_path = "dataset/EuroSAT_RGB/River/River_18.jpg"
    query = "Explain the recent environmental characteristics of this region."
    state = SatSightState(image_path=image_path, user_query=query)

    result_dict = app.invoke(state.model_dump())  # LangGraph expects dict input
    final_state = SatSightState(**result_dict)    # rebuild model from dict

    print("\n=== FINAL OUTPUT ===")
    print((final_state.answer or "")[:1500])
    print("\nConfidence:", final_state.confidence)
    print("MCP needed:", final_state.mcp_needed)

if __name__ == "__main__":
    main()
