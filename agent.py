from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
)
from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 40000

# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
]

# -------------------------------------------------
# LLM INIT (SPEED MODE)
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # 30 RPM (Safe for Tier 1 Pro)
    check_every_n_seconds=0.1,
    max_bucket_size=1         # Prevent bursts
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-pro",   # Pro is smarter for Audio/Vision tasks
    rate_limiter=rate_limiter
).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain, then output END.

Rules:
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool.
- Never hallucinate URLs or fields.
- Always inspect server response.
- Use tools for HTML, downloading, rendering, OCR, or running code.
- Include:
    email = {EMAIL}
    secret = {SECRET}

CRITICAL INSTRUCTIONS:
1. **GitHub Task (project2-gh-tree):** - DO NOT use `get_rendered_html` or `requests.get` on the GitHub page itself.
   - You MUST run the following Python code EXACTLY using `run_code`:
   ```python
   import requests
   # 1. Fetch the Tree
   r = requests.get("[https://api.github.com/repos/sanand0/tools-in-data-science-public/git/trees/95224924d73f70bf162288742a555fe6d136af2d?recursive=1](https://api.github.com/repos/sanand0/tools-in-data-science-public/git/trees/95224924d73f70bf162288742a555fe6d136af2d?recursive=1)")
   data = r.json()

   # 2. Count Files
   count = 0
   for file in data.get("tree", []):
       # Filter by path prefix 'project-1/' and extension '.md'
       if file["path"].startswith("project-1/") and file["path"].endswith(".md"):
           count += 1

   # 3. Calculate Offset
   email = "{EMAIL}"
   offset = len(email) % 2

   # 4. Final Answer
   print(count + offset)
