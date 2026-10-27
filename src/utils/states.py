"""
states.py - the "shared memory" schemas for every graph in this project.

In LangGraph, a graph is a set of nodes that all read from / write to ONE shared
State object. The State schema declares which keys exist and (optionally) HOW
updates to a key are merged. Three schemas live here, one per graph:

    GenerateAnalystsState -> the small "create + review analysts" graph
                             (src/creating_analysts.py)
    InterviewState        -> the interview subgraph, one run per analyst
                             (src/answering_question.py)
    ResearchGraphState    -> the full end-to-end graph (src/agent.py)

Key idea - REDUCERS:
    By default a node's returned value OVERWRITES the key in the state.
    If a key is declared as Annotated[type, reducer_fn], LangGraph calls
    reducer_fn(old_value, new_value) instead. `operator.add` on lists means
    "concatenate", which is what lets several parallel branches each append
    their own result without clobbering each other.
"""

# TypedDict: a dict with a fixed set of typed keys (checked by type checkers only,
#            at runtime it is just a normal dict).
# NotRequired: marks a TypedDict key that may be missing from the dict.
# Annotated: attaches extra metadata (here, a reducer function) to a type.
from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Optional, List
from .objects import Analyst  # relative import: "objects.py" in this same package
# MessagesState is LangGraph's ready-made state with a `messages` key whose reducer
# (add_messages) APPENDS new chat messages instead of overwriting the list.
from langgraph.graph import MessagesState
import operator  # operator.add(a, b) == a + b -> list concatenation for lists


class GenerateAnalystsState(TypedDict):
    """State for the analyst-creation graph (creating_analysts.py)."""
    topic: str # Research Topic
    max_analysts: int # Number of analysts
    # NotRequired[...] because on the very first run the user hasn't given feedback yet.
    # Optional[str] because "approved" is represented as None.
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback for what is generated
    analysts: NotRequired[List[Analyst]] # List of all our analysts


class InterviewState(MessagesState):
    """State for ONE analyst <-> expert interview (the subgraph).

    Inherits `messages` (with the add_messages reducer) from MessagesState, so the
    whole conversation accumulates there turn after turn.
    """
    max_num_turns: int # Number of turns of conversation
    # reducer = operator.add: both search nodes run in parallel and each returns
    # {"context": [text]}; the reducer concatenates them instead of one overwriting the other.
    context: Annotated[list, operator.add] # source of docs
    analyst: Analyst # my analyst
    interview: str # interview transcript
    sections: list # final key we duplicate in outer state for Send() api


class ResearchGraphState(TypedDict):
    """State for the full research graph (agent.py)."""
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    # This is the "reduce" half of map-reduce: every interview subgraph returns
    # {"sections": [one_section]} and operator.add stitches all of them into one list.
    # The key name matches InterviewState.sections - that shared name is how data
    # flows from the subgraph back up into this parent state.
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report
