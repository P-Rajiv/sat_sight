# run_graph.py
from satsight_graph import build_rag_graph
from graph_state import SatSightState

def main():
    g = build_rag_graph()
    app = g.compile()

    state = SatSightState(
        image_path=None,
        user_query="What are the known threats to tropical rainforest ecosystems?"
    )
    result = app.invoke(state.model_dump())
    final = SatSightState(**result)

    print("Answer:\n", final.answer)
    print("Confidence:", final.confidence)
    print("Used documents:", final.retrieved_docs)

if __name__ == "__main__":
    main()
