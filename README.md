# SatSight — Agentic AI Project

**SatSight** is an **Agentic AI system** built using **graph-based orchestration** of reasoning and tool-using agents.  
It demonstrates how intelligent workflows can be dynamically compiled and executed for reasoning, decision-making, and environmental or satellite-sight analysis tasks.

---

## Architecture Overview

```
┌──────────────────────┐
│  Input / Task State  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Reasoning Node      │
│  (LLM / Custom Logic)│
└──────────┬───────────┘
           │ Conditional Edge
           ▼
┌──────────────────────┐
│  MCP / Tool Node     │
│  (External Actions)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  End / Output Node   │
└──────────────────────┘
```

---

## Project Structure

```
sat_sight/
├── src/
│   ├── run_graph.py          # Entry point for running the reasoning graph
│   ├── graph_builder.py      # Graph definition and node composition
│   ├── reasoning.py          # Reasoning agent logic
│   ├── mcp_module.py         # MCP / tool interaction logic
│   └── utils/                # Helper functions and config
├── env_sat_sight/            # Virtual environment
├── README.md
└── requirements.txt
```

---

## How It Works

1. **Define nodes** for reasoning, perception, or tool use  
2. **Add conditional edges** between them, e.g.:
   ```python
   g.add_conditional_edges(
       "reasoning",
       lambda state: "mcp" if state.mcp_needed else END,
       {"mcp": "mcp", END: None}
   )
   ```
3. **Compile and run**:
   ```bash
   python src/run_graph.py
   ```
4. The graph dynamically executes reasoning and tool actions until completion.

---

## Installation

```bash
git clone https://github.com/<your-username>/sat_sight.git
cd sat_sight
python -m venv env_sat_sight
source env_sat_sight/bin/activate  # On Windows: env_sat_sight\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

To run the full Agentic workflow:

```bash
python src/run_graph.py
```

You should see logs detailing graph compilation, reasoning steps, and final outputs.

---

## Tech Stack

- **Python 3.9+**
- **Graph-Oriented Agent Framework**
- **LLM (via API or local model)**
- **Custom MCP Integration**
