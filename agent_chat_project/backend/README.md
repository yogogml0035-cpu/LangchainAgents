# Agent Chat Backend

Local FastAPI backend for the v1 Agent Chat workspace.

## Scope

- Stores every substantive task in a local task directory.
- Accepts Markdown uploads for the v1 document-analysis workflow.
- Generates an execution plan before acting.
- Runs category sub-agents concurrently with retry-on-failure behavior.
- Persists tool-style events, sub-agent reports, evidence records, final Markdown summary, and an interactive HTML report.
- Exposes only safe model IDs to the frontend. Provider secrets stay in backend `.env`.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

## Test

```bash
uv run pytest
```

## Environment

Copy `.env.example` to `.env` and configure:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `TAVILY_API_KEY`
- `AGENT_CHAT_ACCESS_TOKEN`
- `AGENT_CHAT_TASK_ROOT`

The default model registry exposes `deepseek-reasoner`.
The backend loads `backend/.env` on startup before reading process environment values.
Task APIs accept local loopback clients by default. If `AGENT_CHAT_ACCESS_TOKEN` is set,
all task APIs require either `Authorization: Bearer <token>` or `X-Agent-Chat-Token`.
