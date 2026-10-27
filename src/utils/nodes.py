"""
nodes.py - the NODES of every graph (the units of work).

A LangGraph node is just a Python function with this contract:
    input : the current State (a dict-like object)
    output: a dict containing ONLY the keys it wants to change (a "partial update")
LangGraph merges that dict into the shared state (using the key's reducer, if any),
then decides which node runs next based on the edges (see edges.py and the graph files).

Nodes in this file, grouped by the graph they belong to:

    Analyst-creation graph      : create_analysts, human_feedback
    Interview subgraph (x N)    : generate_questions, search_web, search_web2,
                                  generate_answer, save_interview, write_section
    Report-writing (map-reduce) : write_report, write_introduction,
                                  write_conclusion, finalize_report
"""

from dotenv import load_dotenv
from .states import GenerateAnalystsState, InterviewState, ResearchGraphState
from .models import llm  # the shared ChatGroq client
from .objects import Analyst, Perspectives, SearchQuery  # Pydantic schemas for structured output
from .prompts import analyst_instructions, question_instructions, search_instructions, answer_instructions
from .prompts import section_writer_instructions, report_writer_instructions, intro_conclusion_instructions
# LangChain message classes. SystemMessage = instructions/rules, HumanMessage = a user turn.
# (AI replies come back from llm.invoke() as AIMessage objects.)
from langchain.messages import SystemMessage, HumanMessage
# interrupt() pauses the graph and waits for a human; see human_feedback() below.
from langgraph.types import interrupt
# Tavily = a search API built for LLM apps (returns clean text snippets + URLs).
from langchain_tavily import TavilySearch
# Converts a list of messages into one "Human: ...\nAI: ..." transcript string.
from langchain_core.messages import get_buffer_string


# Loads TAVILY_API_KEY / GROQ_API_KEY from .env into the environment so the clients can find them.
load_dotenv()


# nodes

# ============================================================================
# ANALYST-CREATION NODES
# ============================================================================
def create_analysts(state: GenerateAnalystsState):
    """Generate Analysts"""

    # Read inputs from the shared state. `.get(key, default)` is used for the optional key,
    # because on the first run `human_analyst_feedback` does not exist yet.
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # Enforce structured output
    # with_structured_output(Perspectives) wraps the LLM so that .invoke() returns a
    # validated `Perspectives` object instead of raw text.
    structured_llm = llm.with_structured_output(Perspectives)

    # System Message
    # Fill the {placeholders} of the prompt template with this run's values.
    system_message = analyst_instructions.format(topic=topic,
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts)

    # Generate analysts
    # A chat call takes a list of messages: the rules (system) + a short user turn that triggers the work.
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Please generate the set of analysts.")])

    # Partial state update: only the `analysts` key changes. On a regeneration this
    # simply overwrites the previous list (no reducer is declared for `analysts`).
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """This is where human gives feedback about the analysts given."""

    # HUMAN-IN-THE-LOOP. interrupt(payload) does three things:
    #   1. pauses the whole graph right here and saves its state (needs a checkpointer;
    #      `langgraph dev` / LangGraph Studio provides one automatically),
    #   2. shows `payload` to whoever is driving the graph (the Studio UI),
    #   3. when the human replies (Command(resume=<value>)), the graph continues and
    #      interrupt() RETURNS that value - it is what ends up in `feedback`.
    # Gotcha: on resume this whole node function is re-run from the top, so keep
    # anything with side effects (API calls, writes) out of the code before interrupt().
    feedback = interrupt({
        "question": "Are these analysts good enough for you?",
        # model_dump() turns a Pydantic object into a plain dict so the payload is JSON-friendly.
        # hasattr() check = also accept analysts that are already plain dicts.
        "analysts": [
            analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
            for analyst in state.get("analysts", [])
        ],
        # The approval words listed here must match the set checked below.
        "instructions": "Return feedback to regenerate analysts or "
        "return empty/perfect/continue/okay/yes to approve and continue the graph"
    })

    # Interpret the human's reply. Convention used here:
    #   None means "approved, no changes" -> the router sends the graph forward.
    #   A non-empty string is real feedback -> the router loops back to create_analysts.
    if feedback is None:
        return {"human_analyst_feedback": None}

    if isinstance(feedback, str):
        feedback = feedback.strip()  # remove leading/trailing whitespace

        # empty string = approve
        if feedback == "":
            return {"human_analyst_feedback": None}

        # magic approval words (case-insensitive) = approve
        if feedback.lower() in {"perfect", "okay", "continue", "yes"}:
            return {"human_analyst_feedback": None}

        # anything else is treated as instructions for how to change the analysts
        return {"human_analyst_feedback": feedback}

    # Non-string replies (e.g. a dict) are treated as approval.
    return {"human_analyst_feedback": None}


# ============================================================================
# INTERVIEW SUBGRAPH NODES  (each analyst gets its own run of these)
# ============================================================================
def generate_questions(state: InterviewState):
    """node to generate the question"""

    # get state analyst
    analyst = state["analyst"]

    # The analyst may arrive as a plain dict (e.g. after JSON serialization through
    # Studio / checkpoints). model_validate() rebuilds the Pydantic object so that
    # `.persona` works. This defensive pattern repeats in the nodes below.
    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    messages = state["messages"]

    # generate question
    # System prompt (analyst role + goals) followed by the whole conversation so far,
    # so the analyst can ask a follow-up based on the expert's last answer.
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)

    # Returning {"messages": [...]} APPENDS (add_messages reducer) rather than replaces.
    return {"messages": [question]}


def search_web(state: InterviewState):
    """Retrieve docs from the web"""

    # Search query
    # Step 1: ask the LLM to turn the conversation into one search query (structured output).
    structured_llm = llm.with_structured_output(SearchQuery)

    # search instruction
    search_instruction_system_message = SystemMessage(content=search_instructions)

    # Step 2: the Tavily client; max_results=3 keeps the context small and cheap.
    tavily_search = TavilySearch(max_results=3)

    search_query = structured_llm.invoke([search_instruction_system_message]+state["messages"])

    # search
    # The langchain-tavily tool returns a dict like {"query": ..., "results": [{url, content, ...}, ...]}.
    # `.get("results", data)` falls back to the raw value if the "results" key is missing.
    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data)

    # Format
    # Step 3: flatten the hits into ONE string, each wrapped in <Document href="url"> tags
    # so the answering prompt can see (and cite) where each snippet came from.
    # (The tag is written as a self-closing <Document .../> then closed again - a quirk,
    # but harmless for an LLM reader.)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    # `context` has the operator.add reducer: this one-item list is APPENDED to what the
    # other search node (and earlier turns) already put there.
    return {"context": [formatted_search_docs]}


def search_web2(state: InterviewState):
    """Retrieve docs from the web"""

    # NOTE: this is a line-for-line copy of search_web(). The graph runs both in PARALLEL
    # after ask_question. Because temperature=0, both usually generate the same query, so
    # you pay for two LLM calls + two Tavily calls and get near-duplicate context. It is
    # a good demo of parallel branches + a reducer, but a candidate for deduplication
    # (e.g. one node using a different source such as Wikipedia).

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)

    # search instruction
    search_instruction_system_message = SystemMessage(content=search_instructions)

    tavily_search = TavilySearch(max_results=3)

    search_query = structured_llm.invoke([search_instruction_system_message]+state["messages"])

    # search
    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}



def generate_answer(state: InterviewState):
    """Node to answer a question"""

    # get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    # answer the question
    # The answer is grounded: the prompt tells the model to use ONLY `context` (RAG pattern).
    # NOTE: `context` is a Python list, so .format() inserts its repr (['...', '...']).
    # The LLM copes, but "\n\n".join(context) would give it cleaner text.
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)

    # name the message as comming from the expert
    # Setting .name tags this AIMessage as the expert's. routes_messages() in edges.py counts
    # messages with name == "expert" to know how many turns have happened.
    answer.name = "expert"

    # append to the state
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """save interviews"""

    messages = state["messages"]

    # Convert the message list into a single readable transcript string
    # ("Human: ...\nAI: ...") and store it in the `interview` key.
    interview = get_buffer_string(messages)

    return {"interview": interview}



def write_section(state: InterviewState):
    """Node to answer a question"""

    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    # Rebuild the Analyst object if state holds a plain dict. Only this conversion is
    # conditional: the rest of the function must run for BOTH cases, otherwise an analyst that
    # is already an Analyst object would silently produce no section (the node would return None).
    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    # The section is written from the retrieved SOURCES (`context`), steered by the analyst's
    # focus; the `interview` transcript is read above but not used in the prompt.
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+
                         [HumanMessage(content=f"Use this source to write your section: {context}")])

    # Returned under `sections`. In the PARENT graph this key has an operator.add reducer,
    # so the sections of all analysts get concatenated into one list (the "reduce" half).
    return {"sections": [section.content]}



# ============================================================================
# REPORT-WRITING NODES  (run in the parent graph after all interviews finish)
# ============================================================================
def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")])
    # `report.content` is the text of the AIMessage; stored in `content` (the report body).
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    # NOTE: `instructions` is a plain string, not a SystemMessage. LangChain converts a bare
    # string in a message list into a HumanMessage, so it works - just not as a "system" role.
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")])
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    # Same prompt as write_introduction; only the HumanMessage below tells the model which part to write.
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # This node runs only after write_report, write_introduction AND write_conclusion are all
    # done (the graph joins those three branches). It does plain string assembly - no LLM call.

    # Save full final report
    content = state["content"]
    # Drop the "## Insights" title the prompt asks for (the intro already provides the title).
    # removeprefix() removes exactly that text. (Do NOT use str.strip("## Insights") here: strip
    # treats its argument as a set of CHARACTERS and would also chop letters such as the "g" off
    # a final "....org" source URL.)
    if content.startswith("## Insights"):
        content = content.removeprefix("## Insights")
    # The report prompt asks the model to end with a "## Sources" section. Split it off so the
    # sources can be re-attached at the very end, after the conclusion.
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            # (bare except: hides the real error, e.g. when the header appears more than once)
            sources = None
    else:
        sources = None

    # Final layout: introduction --- body --- conclusion --- sources.
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}




