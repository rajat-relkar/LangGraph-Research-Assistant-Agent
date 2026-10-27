"""
edges.py - the ROUTING functions used by conditional edges.

A normal edge (add_edge("a", "b")) always goes a -> b. A CONDITIONAL edge instead runs
a function after node "a" and lets that function decide where to go next. The function
receives the State and returns either:
    - the NAME of the next node (a string, or END), or
    - a list of `Send(...)` objects to launch several parallel runs (map-reduce).

Routers in this file:
    should_continue        -> used by creating_analysts.py  (approve or regenerate?)
    routes_messages        -> used by answering_question.py (ask again or finish interview?)
    initiate_all_interviews-> used by agent.py              (regenerate or fan out interviews?)
"""

from dotenv import load_dotenv
from src.utils.states import GenerateAnalystsState, InterviewState, ResearchGraphState
from typing import Literal
from langgraph.graph import END  # END = special node name that terminates the graph ("__end__")
from langchain.messages import AIMessage, HumanMessage
# Send(node_name, state_for_that_run) = "run this node with THIS input", used to fan out in parallel.
from langgraph.types import Send

load_dotenv()

# conditional edges
def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", END]:
    """Return the next node to execute"""

    # The Literal[...] return annotation lists the possible destinations. It is mainly
    # documentation here, but it also lets LangGraph draw the possible edges in Studio.

    human_analyst_feedback = state.get("human_analyst_feedback", None)

    # Non-empty feedback (truthy string) -> loop back and regenerate the analysts.
    if human_analyst_feedback:
        return "create_analysts"

    # No feedback means the human approved -> finish the graph.
    return END


def routes_messages(state: InterviewState, name: str="expert"):
    """Route messages between question and answer"""

    # Decides what happens after the expert answers: ask another question, or wrap up.
    # `name="expert"` is a default argument; LangGraph only passes `state`, so it always equals "expert".

    # get messages
    messages = state["messages"]
    # max_num_turns is not set by agent.py, so this default of 2 is what actually applies
    # (= each analyst gets two question/answer exchanges).
    max_num_turns = state.get("max_num_turns", 2)

    # check number of expert answers
    # Count only AI messages tagged name == "expert" (generate_answer sets that tag).
    # The analyst's own questions are also AIMessages, but have no name, so they are not counted.
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    # Enough answers collected -> go write down the interview.
    if num_responses >= max_num_turns:
        return "save_interview"

    # Otherwise loop: the analyst asks the next question.
    return "ask_question"

def initiate_all_interviews(state: ResearchGraphState):
    """This is the map step where we run each interview in sub-graph using Send API"""

    human_analyst_feedback = state.get("human_analyst_feedback")

    # Feedback present -> the human wants different analysts, so go back and regenerate.
    if human_analyst_feedback:
        return "create_analysts"

    else:
        topic = state["topic"]
        # MAP step: build one Send per analyst. Each Send starts a separate run of the
        # "conduct_interview" subgraph with its OWN private input state (that analyst + a
        # first message). LangGraph runs all of them in parallel, and the number of
        # branches is decided at runtime by len(state["analysts"]) - something a fixed
        # add_edge() cannot express.
        # The first HumanMessage is the interview's opening line; the analyst node then
        # sees it and responds with its self-introduction + first question.
        return [
            Send("conduct_interview", {
                "analyst": analyst,
                "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}")]
            }) for analyst in state["analysts"]
        ]
