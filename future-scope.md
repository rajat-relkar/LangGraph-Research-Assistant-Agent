# Future Scope

This document describes the planned next steps for the LangGraph Research Assistant. The four items below close the biggest gaps between "working demo" and "production-quality project":

1. [Automated tests](#1-automated-tests)
2. [LLM-driven tool calling](#2-llm-driven-tool-calling)
3. [Evaluation of report quality](#3-evaluation-of-report-quality)
4. [Deployment](#4-deployment)

Smaller improvements (CLI runner, configurable model, deduplicating the search nodes, and so on) are listed under **Future Improvements** in [README.md](README.md). Background on the concepts used here is in [notes.md](notes.md).

> API names and snippets below were checked against the versions in `uv.lock` (`langgraph` 1.2.17, `langchain` 1.4.7, `langchain-groq` 1.1.3, `langchain-tavily` 0.2.18). Re-check them if you upgrade.

---

## Where the project stands today

| Area | Status |
|---|---|
| Tests | None. Behaviour has only been checked by hand and with a throw-away fake-LLM script |
| Tools | The LLM never calls tools. Tavily is called directly by `search_web` / `search_web2`; `src/utils/tools.py` is an empty placeholder |
| Evaluation | None. Report quality is judged by reading the output |
| Deployment | Local only: `uv run langgraph dev` with an in-memory checkpointer |

## Suggested order

| Step | Item | Effort | Why in this position |
|---|---|---|---|
| 1 | Automated tests | Small | Protects every later change |
| 2 | Tool calling | Medium | Biggest code change, so make it with tests in place |
| 3 | Report evaluation | Medium | Measures whether tool calling and prompt changes actually helped |
| 4 | Deployment | Medium to large | Deploy something that is tested and measured |

---

## 1. Automated tests

**Goal:** catch regressions automatically, without spending API credits.

### Setup

`pytest` is not installed yet:

```bash
uv add --dev pytest
```

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Suggested layout:

```text
tests/
|-- conftest.py          # fake LLM + fake Tavily fixtures
|-- test_edges.py        # routers
|-- test_nodes.py        # individual nodes and regression tests
`-- test_graph_e2e.py    # whole graph with fakes + MemorySaver
```

### Three layers of tests

**a) Routers and pure functions (no LLM needed).** `Send` objects expose `.node` and `.arg`, so fan-out is easy to assert on:

```python
from langchain_core.messages import AIMessage
from src.utils.edges import routes_messages, initiate_all_interviews

def test_routes_messages_stops_after_max_turns():
    msgs = [AIMessage(content="q"), AIMessage(content="a", name="expert"),
            AIMessage(content="q2"), AIMessage(content="a2", name="expert")]
    assert routes_messages({"messages": msgs}) == "save_interview"

def test_fans_out_one_send_per_analyst(analyst_a, analyst_b):
    sends = initiate_all_interviews({"topic": "t", "analysts": [analyst_a, analyst_b]})
    assert [s.node for s in sends] == ["conduct_interview"] * 2
```

**b) Whole graph with fakes.** The nodes look up `llm` and `TavilySearch` as module globals at call time, so a fixture can replace them with `monkeypatch`:

```python
# conftest.py (sketch)
import pytest
import src.utils.nodes as N

@pytest.fixture
def fake_services(monkeypatch):
    monkeypatch.setattr(N, "llm", FakeLLM())
    monkeypatch.setattr(N, "TavilySearch", FakeTavily)
```

```python
# test_graph_e2e.py (sketch)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from src.agent import builder

def test_full_run_produces_one_section_per_analyst(fake_services):
    graph = builder.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"topic": "demo", "max_analysts": 2}, cfg)   # stops at the interrupt
    out = graph.invoke(Command(resume="yes"), cfg)            # approve and continue
    assert len(out["sections"]) == 2
    assert out["final_report"]
```

The full `FakeLLM` / `FakeTavily` code is in [notes.md](notes.md), section 11 ("How to test without spending API credits").

**c) Regression tests for the bugs already fixed** (see `notes.md`, section 11.1):

- `write_section` returns a section for an `Analyst` object **and** for a plain dict.
- `finalize_report` keeps a final `https://example.org` source URL intact.
- The `human_feedback` hint lists the word `continue`.

### Extras

- Mark a few real-API tests with `@pytest.mark.live` and skip them by default (they cost money).
- Add a GitHub Actions workflow that runs `uv sync` and `uv run pytest` on every push, plus a build badge in the README.

### Done when

- [ ] `uv run pytest` passes with no network access and no API keys.
- [ ] Routers, `finalize_report`, and the three bug fixes each have a test.
- [ ] One end-to-end test covers interrupt, resume, and final report.
- [ ] CI runs the suite on every push.

---

## 2. LLM-driven tool calling

**Goal:** let the expert decide *whether* to search and *what* to search for, instead of always running two fixed search nodes. This turns the interview step from a fixed workflow into a small tool-using agent.

**What is already in place** (verified): `TavilySearch` is a standard LangChain tool (named `tavily_search`), `ChatGroq` supports `bind_tools`, and `ToolNode` / `tools_condition` import from `langgraph.prebuilt`.

### Changes in the interview subgraph

```python
# src/utils/tools.py  (currently empty)
from langchain_tavily import TavilySearch

tools = [TavilySearch(max_results=3)]
```

```python
# src/utils/nodes.py: the expert binds the tools
from .tools import tools
llm_with_tools = llm.bind_tools(tools)

def generate_answer(state: InterviewState):
    ...
    answer = llm_with_tools.invoke([SystemMessage(content=system_message)] + messages)
    return {"messages": [answer]}      # may carry answer.tool_calls
```

```python
# src/answering_question.py
from langgraph.prebuilt import ToolNode

interview_builder.add_node("tools", ToolNode(tools))
interview_builder.add_edge("tools", "answer_question")   # tool results return to the expert
```

```python
# src/utils/edges.py: search first if the model asked for it
def route_after_expert(state: InterviewState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return routes_messages(state)
```

Then replace the conditional edge on `answer_question` with `route_after_expert`, adding `"tools"` to its destination list.

### Details to handle

- **Remove `search_web` and `search_web2`.** This also resolves the duplicate-search issue listed in `notes.md` (section 11.2).
- **Tag only the final answer** with `name="expert"`. Otherwise `routes_messages` would count tool-calling messages as turns.
- **Cap tool calls per turn** to control cost and avoid loops.
- **Update `answer_instructions`.** It currently cites from a `{context}` block; it must cite from the tool results instead.
- **Confirm the chosen Groq model supports tool use** (the `gpt-oss` models are documented as doing so, but verify on Groq's docs before relying on it).
- **Trade-off:** behaviour becomes less predictable and harder to test. This is why tests (step 1) and evaluation (step 3) come first or alongside.

### Done when

- [ ] The expert can answer with zero, one, or several searches in a turn.
- [ ] Turn counting still works with tool-calling messages in the history.
- [ ] Citations in the final report point to the tool-retrieved sources.
- [ ] Tests cover the "model calls a tool" and "model answers directly" paths (use a fake LLM that returns an `AIMessage` with `tool_calls`).

---

## 3. Evaluation of report quality

**Goal:** replace "I read it and it looks fine" with repeatable scores, so prompt, model, and architecture changes can be compared.

### Layout

```text
evals/
|-- dataset.json      # 5-10 diverse topics
|-- run_evals.py      # runs research_agent per topic and saves the reports
|-- judge.py          # scoring logic
`-- results/          # saved scores per run, to compare across changes
```

`run_evals.py` needs a helper that runs the graph and auto-approves the analysts: invoke, then resume with `Command(resume="yes")`.

### Two kinds of checks

**a) Cheap, deterministic checks (no LLM):**

- The report has a title, `## Introduction`, `## Conclusion`, and `## Sources`.
- Every `[n]` citation has a matching entry in the sources list.
- No analyst names appear (the report prompt forbids them).
- Source URLs are well-formed and unique.

**b) LLM-as-judge**, using the same structured-output technique as the rest of the project:

```python
class Score(BaseModel):
    relevance: int = Field(description="1-5: does the report answer the topic?")
    grounding: int = Field(description="1-5: are claims supported by the cited sources?")
    coverage: int = Field(description="1-5: are all analyst perspectives reflected?")
    clarity: int = Field(description="1-5: well organised and readable?")
    reasoning: str = Field(description="Short justification for the scores.")

judge = judge_llm.with_structured_output(Score)   # temperature 0
```

Use a **different, ideally stronger, model as the judge** than the one that wrote the report. Models tend to rate their own output too generously.

### One code change this needs

The retrieved web text lives only in each interview's private `context`, so the parent graph never sees it. To verify citations against real source text, add a key to `ResearchGraphState`, for example `all_context: Annotated[list, operator.add]`, and return it from the interview subgraph.

### Tooling

`langsmith` is already installed (a dependency of LangChain), so datasets and results can be stored there. A plain script writing JSON into `results/` works equally well. Keep evals **out of CI by default**, since every run costs LLM and search credits; run them manually or on a schedule.

### Done when

- [ ] One command runs the full eval set and writes a results file.
- [ ] Deterministic checks and judge scores are reported per topic and as an average.
- [ ] A before/after comparison exists for at least one real change (for example, tool calling vs. fixed search).

---

## 4. Deployment

**Goal:** run the assistant as a service that other people (or a front end) can call, with pause/resume surviving restarts.

### Two paths

| Path | What it involves | Notes |
|---|---|---|
| **A. LangGraph's own tooling** | `langgraph dockerfile` and `langgraph build -t research-agent` generate and build an image from `langgraph.json`; `langgraph up` runs it locally; `langgraph deploy` (marked Beta in the installed CLI) deploys to LangSmith Deployment | Least code. Needs Docker. Check current pricing and self-hosting licence terms before depending on it |
| **B. Own FastAPI service** | Add `fastapi`; expose `POST /research` (start a run), `POST /research/{thread_id}/feedback` (resume with `Command`), `GET /research/{thread_id}` (status or final report); package with a Dockerfile | Best for showing backend design in a portfolio |

### Path B essentials

- **Persistent checkpointer.** Replace `MemorySaver` with Postgres (`langgraph-checkpoint-postgres`). With an in-memory saver, every paused human-feedback step is lost when the server restarts.
- **Progress streaming.** Use `graph.astream(...)` and stream node updates to the client.
- **Hosting.** Any container host works: Render, Railway, Fly.io, or Oracle Cloud (OCI), among others.

### Production basics (either path)

- **Secrets:** pass `GROQ_API_KEY` and `TAVILY_API_KEY` as environment variables on the host. Never bake `.env` into the image; add `.env`, `.venv`, and `.langgraph_api` to a `.dockerignore`.
- **Cost guard rails:** cap `max_analysts` and interview turns, and add authentication and rate limiting. Without them, anyone who finds the URL can run up the API bill.
- **Reliability:** add retries and timeouts (`RetryPolicy` on the search and LLM nodes) and a health-check endpoint.
- **Observability:** turn on LangSmith tracing to inspect every LLM and search call.

### Done when

- [ ] The service starts from a single Docker command.
- [ ] A run can be started, paused at human feedback, the server restarted, and the run resumed.
- [ ] Keys come from the environment only, and the image contains no `.env`.
- [ ] Limits and authentication prevent unbounded spend.
