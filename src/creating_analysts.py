"""
creating_analysts.py - the SMALL graph: generate analyst personas, then ask a human.

Registered in langgraph.json as the graph id "create_analysts", so you can run just this
part on its own in LangGraph Studio to tune the persona-generation step.

    START -> create_analysts -> human_feedback --(feedback given)--> create_analysts  (loop)
                                                +--(approved)------> END
"""

from src.utils.nodes import create_analysts, human_feedback
# StateGraph builds a graph around a State schema; START / END are the built-in entry and exit markers.
from langgraph.graph import START, END, StateGraph
from src.utils.states import GenerateAnalystsState
from dotenv import load_dotenv
from src.utils.edges import should_continue

load_dotenv()

# creating our graph
# GenerateAnalystsState
# node create_analysts
# edges no need / simple

# 1. Create a builder bound to the state schema. Every node will read/write this state.
builder = StateGraph(GenerateAnalystsState)

# 2. Register nodes: add_node("name used in edges", python_function).
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback )

# 3. Wire the edges.
builder.add_edge(START, "create_analysts")             # entry point
builder.add_edge("create_analysts", "human_feedback")  # always review after generating
# Conditional edge: after human_feedback, should_continue() returns either
# "create_analysts" (redo with the feedback) or END (approved).
builder.add_conditional_edges("human_feedback", should_continue)

# 4. compile() validates the graph and turns the builder into a runnable object.
# `graph` is the variable name langgraph.json points to ("...:graph").
# No checkpointer is passed here: the LangGraph dev server attaches one, which
# interrupt() in human_feedback needs. Running this file standalone would require
# graph = builder.compile(checkpointer=MemorySaver()) plus a thread_id in the config.
graph = builder.compile()
