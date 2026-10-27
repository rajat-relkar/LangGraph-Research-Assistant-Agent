"""
answering_question.py - the INTERVIEW subgraph (one analyst <-> one expert).

This is a complete graph in its own right, registered in langgraph.json as
"question_answer_graph", AND it is reused as a single node ("conduct_interview") inside
the big graph in agent.py. agent.py imports `interview_builder` from here and compiles
it itself.

Flow (loops until the expert has answered `max_num_turns` times):

    START -> ask_question --+--> search_web   --+--> answer_question --(routes_messages)--> ask_question (loop)
                            |                   |                                      +--> save_interview
                            +--> search_web2  --+                                             -> write_section -> END
"""

from src.utils.states import InterviewState
from src.utils.nodes import generate_questions, search_web, search_web2, generate_answer, save_interview, write_section
from src.utils.states import InterviewState  # (duplicate import of the line above - harmless)
from src.utils.edges import  routes_messages
from langgraph.graph import StateGraph, START, END

# Add nodes and edges
# Same pattern as always: builder around a state schema, then nodes, then edges.
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_questions)  # analyst asks (LLM)
interview_builder.add_node("search_web", search_web)            # Tavily search, branch 1
interview_builder.add_node("search_web2", search_web2)          # Tavily search, branch 2
interview_builder.add_node("answer_question", generate_answer)  # expert answers from the search results (LLM)
interview_builder.add_node("save_interview", save_interview)    # turn messages into a transcript string
interview_builder.add_node("write_section", write_section)      # LLM writes the memo for this interview

# Flow
interview_builder.add_edge(START, "ask_question")
# FAN-OUT: two edges leave ask_question, so search_web and search_web2 run in the SAME
# step, in parallel. Their results are merged by the `context` reducer (operator.add).
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_web2")
# FAN-IN: both searches point to answer_question. Because both finish in the same step,
# answer_question runs once, after both are done. (If the branches had different lengths,
# you would use the list form add_edge(["a", "b"], "c") to make c wait for ALL of them -
# agent.py does exactly that for finalize_report.)
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_web2", "answer_question")
# Conditional edge = the interview LOOP: routes_messages() returns "ask_question" (go again)
# or "save_interview" (enough turns). The list is the set of allowed destinations; it is
# used for validation and for drawing the graph.
interview_builder.add_conditional_edges("answer_question", routes_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Compiled standalone version, for langgraph.json / Studio. (agent.py re-compiles interview_builder.)
question_answer_graph = interview_builder.compile()
