"""
objects.py - Pydantic models used for STRUCTURED OUTPUT from the LLM.

Normally an LLM returns free text. Here we want predictable, typed data (a list of
analysts, a search query). We hand a Pydantic model to `llm.with_structured_output(...)`
(see nodes.py) and LangChain makes the model reply in that exact shape and parses it
back into a real Python object - no manual string/JSON parsing needed.

The `Field(description=...)` text is NOT just documentation: it is sent to the LLM as
part of the schema, so it actively steers what the model writes in each field.
"""

from pydantic import BaseModel, Field
from typing import List

# State Creation
class Analyst(BaseModel):
    """One AI analyst persona (who is asking the questions in an interview)."""
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    # @property = a computed, read-only attribute: access it as `analyst.persona`
    # (no parentheses). It is NOT a model field, so the LLM never fills it in - we
    # build it ourselves from the four real fields. It is the text that gets dropped
    # into prompts as "{goals}".
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\n Role: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    """Wrapper so the LLM returns the WHOLE panel of analysts in a single call."""
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

class SearchQuery(BaseModel):
    """The web-search query the LLM derives from the interview so far."""
    search_query: str = Field(description="Search query for retrieval")

