"""
agent.py - the MAIN graph: the complete research assistant, end to end.

Registered in langgraph.json as the graph id "research_agent". It combines everything:
    1. generate analysts           (create_analysts)
    2. pause for human approval    (human_feedback)
    3. interview each analyst IN PARALLEL using the interview subgraph (conduct_interview)
    4. write report body + introduction + conclusion IN PARALLEL
    5. glue everything into the final Markdown report (finalize_report)

    START -> create_analysts -> human_feedback --(feedback)--> create_analysts   (loop)
                                      |
                                      +--(approved: one Send per analyst)--> conduct_interview x N
                                                                                   |
                                         +-------------------+----------------------+
                                         v                   v                      v
                                   write_report      write_introduction      write_conclusion
                                         +-------------------+----------------------+
                                                             v
                                                      finalize_report -> END

This is the MAP-REDUCE pattern: MAP = N parallel interviews, REDUCE = merge all sections
into a single report.
"""

from src.utils.states import GenerateAnalystsState, ResearchGraphState
# Wildcard import: pulls in every public name from nodes.py (create_analysts, human_feedback,
# write_report, ...). Explicit imports are easier to trace, but this keeps the file short.
from src.utils.nodes import *
from src.utils.edges import initiate_all_interviews
# The *uncompiled* interview builder, so it can be compiled and embedded as a node below.
from src.answering_question import interview_builder
from langgraph.graph import START, END, StateGraph


# building our graph
# The state schema is ResearchGraphState: it holds the topic, analysts, the accumulated
# `sections` list, and the pieces of the final report.
builder = StateGraph(ResearchGraphState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
# SUBGRAPH AS A NODE: a compiled graph can be passed to add_node like any function. Its state
# schema (InterviewState) differs from the parent's; the two only share the key `sections`,
# which is how each interview's result flows back up (and gets merged by operator.add).
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# Logic

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
# After the human reviews: initiate_all_interviews() either returns "create_analysts"
# (feedback -> redo) or a list of Send objects (approved -> parallel interviews).
# The list gives the possible destinations (for validation / drawing the graph).
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
# FAN-OUT: once ALL interviews are finished, these three nodes start in the same step.
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
# FAN-IN with a LIST of sources: finalize_report waits until ALL THREE have finished.
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
# `graph` is what langgraph.json exposes as "research_agent". Like the other graphs, no
# checkpointer is given here because the LangGraph dev server supplies one (needed by interrupt()).
graph = builder.compile()
