# Smartphone Catalog Agent — LLM Evals

A LangChain agent that recommends smartphones from a Markdown catalog, instrumented with Langfuse for tracing, prompt management, user feedback, and automated model-based evaluation.

## How it works

- **Retrieval:** `smartphones_catalog.md` is split by `##` model headers, embedded with `text-embedding-3-small`, and stored in a Qdrant collection (`smartphones`). Ingestion happens once, on first search.
- **Agent:** built with `create_agent` (model `gpt-5.4-mini`) and two tools:
  - `SearchCatalog` — semantic search over the catalog (top 4 results).
  - `EndSession` — sentinel tool; when called, the loop asks for a 1–5 rating and a comment, then exits.
- **System prompt:** fetched from Langfuse (`agent_system_prompt`, text, label `production`) and linked to each generation.
- **Tracing:** each user turn runs inside a `main_agent_loop` root observation tagged with `user_id` and `session_id`.
- **Scores:** `user_feedback` (numeric, API) is attached to the trace; `Context Precision` and `Faithfulness` are added automatically by Langfuse evaluators.

## Prerequisites

- Python 3.10+
- Qdrant at `http://localhost:6333`
- Langfuse at `http://localhost:3000` (project `llm-evals`)
- An OpenAI API key

## Setup

1. Install dependencies in a virtualenv: `langfuse`, `langchain`, `langchain-openai`, `langchain-qdrant`, `langchain-text-splitters`, `langgraph`, `qdrant-client`, `python-dotenv`, `hstest`, `requests`.
2. Create `.env` in this directory:

   ```
   OPENAI_API_KEY=...
   LANGFUSE_SECRET_KEY=...
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_BASE_URL="http://localhost:3000"
   ```

3. In Langfuse, create the `agent_system_prompt` text prompt and the score configs `response_relevance` (numeric) and `tool_usage_success` (boolean).

## Run

```bash
python agent.py
```

Chat with the agent. Say thanks/goodbye to trigger `EndSession`, then enter a rating and comment.

## Model-based evaluation (Langfuse UI)

1. **Project Settings → LLM Connections:** add an OpenAI connection with a valid key.
2. **Evaluation → Evaluators → New Evaluator**, once each for the managed templates **Context Precision** and **Faithfulness**:
   - Target: live observations
   - Filter: `Type = generation`
   - Variable mapping: `input` → observation input (last user message), `output` → observation output, `context` → retrieved catalog text (the `SearchCatalog` tool message in the generation input; JsonPath optional)
   - Status: active
3. Run the agent to generate traffic and confirm that scores appear on traces (source `EVAL`).

## Verify

- `python check_v2.py` — counts recent `main_agent_loop` observations via the Langfuse metrics API.
- `python tests.py` — runs the full stage test suite (needs Qdrant, Langfuse, and a valid OpenAI key).

## Files

| File | Purpose |
|------|---------|
| `agent.py` | The agent and chat loop |
| `smartphones_catalog.md` | Source catalog |
| `check_v2.py` | Quick metrics check |
| `tests.py`, `test/tests.py` | Stage tests |
| `.env` | Secrets (git-ignored) |
