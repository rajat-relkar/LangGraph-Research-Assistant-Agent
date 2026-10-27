# tools.py - intentionally empty placeholder.
#
# In LangChain/LangGraph a "tool" is a function the LLM can decide to call (search, calculator, ...).
# This project does not use LLM-driven tool calling: the Tavily search is called directly
# from the `search_web` / `search_web2` nodes in nodes.py. If you later let the model choose
# its own tools (e.g. with @tool + bind_tools + ToolNode), define them here.
