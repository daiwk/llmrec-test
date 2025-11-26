# LangGraph LLM Recommender Demo

A minimal, torch-free recommendation flow built on [LangGraph](https://github.com/langchain-ai/langgraph).
Multiple agents share a single LLM, use different prompts, pull tool outputs, and merge their views to produce movie suggestions.

## Project layout

- `data/movies.csv` &mdash; small public-style catalog used by the graph.
- `src/llmrec/graph.py` &mdash; LangGraph pipeline wiring multiple agents.
- `src/llmrec/llm.py` &mdash; simple local LLM stub; swap with a real model if desired.
- `run_demo.py` &mdash; example entry point.

## Installing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the demo

```bash
python run_demo.py
```

You should see a bullet list of recommendations produced by the master agent after
combining the preference agent, retrieval agent, and RAG/context agent.
