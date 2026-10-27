"""
models.py - creates the single LLM client shared by every node.

Importing `llm` from here (instead of building a model inside each node) means the
provider/model is configured in exactly ONE place: change it here and the whole
project switches model.
"""

from dotenv import load_dotenv
# ChatGroq is LangChain's wrapper around Groq's hosted-inference API. Because it is
# a LangChain "chat model", it exposes the same interface as any other provider
# (.invoke, .with_structured_output, ...), so swapping providers later is easy.
from langchain_groq import ChatGroq

# Reads the .env file and puts its KEY=VALUE lines into os.environ.
# ChatGroq automatically looks for GROQ_API_KEY there - we never pass the key in code.
load_dotenv()

# Alternative model ids noted for quick swapping:
# openai/gpt-oss-20b
# qwen/qwen3.8-27b

# model       - Groq model id. "openai/gpt-oss-120b" is OpenAI's open-weight model served by Groq.
# temperature - 0 means (almost) deterministic output: the model always picks its most
#               likely tokens. Good for structured output, factual answers and repeatable runs.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# if __name__ == "__main__":
#     response = llm.invoke("Say hello in one sentence.")
#     print(response.content)
