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
MAX_TOKENS = 40000 # Kept moderate to prevent Token Quota crashes

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
# LLM INIT (SAFE MODE)
# -------------------------------------------------
# We set this to 0.2 req/s (~12 RPM) to stay safely under the 14 RPM limit.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,   # SLOW: 1 request every 2 seconds
    check_every_n_seconds=0.1,
    max_bucket_size=1          # No bursts allowed
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-pro",    # Back to Pro for Audio/Image capability
    rate_limiter=rate_limiter
).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT (OPTIMIZED)
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
- **GitHub/JSON Tasks:** If the URL looks like an API (ends in .json) or is a GitHub Tree, **DO NOT use `get_rendered_html`**. Instead, use `run_code` to fetch it using `requests.get(url).json()`. This is much faster and saves memory.
- **File Counting:** When counting files in a list, write Python code to count them programmatically.
- **Audio:** Use the `transcribe_audio` tool.
"""

# -------------------------------------------------
# NODES & GRAPH
# -------------------------------------------------
def handle_malformed_node(state: AgentState):
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            {
                "role": "user", 
                "content": "SYSTEM ERROR: Your last tool call was Malformed (Invalid JSON). Please rewrite the code and try again."
            }
        ]
    }

def agent_node(state: AgentState):
    cur_time = time.time()
    cur_url = os.getenv("url")
    prev_time = url_time.get(cur_url) 
    offset = os.getenv("offset", "0")

    if prev_time is not None:
        prev_time = float(prev_time)
        diff = cur_time - prev_time
        # Increased timeout for the slower rate limit
        if diff >= 400 or (offset != "0" and (cur_time - float(offset)) > 300):
            print(f"Timeout exceeded ({diff}s).")
            fail_instruction = "Timeout exceeded. Submit a placeholder answer to skip."
            fail_msg = HumanMessage(content=fail_instruction)
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}

    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm, 
    )
    
    has_human = any(msg.type == "human" for msg in trimmed_messages)
    if not has_human:
        print("WARNING: Context was trimmed too far. Injecting state reminder.")
        current_url = os.getenv("url", "Unknown URL")
        reminder = HumanMessage(content=f"Context cleared. Continue processing URL: {current_url}")
        trimmed_messages.append(reminder)

    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    result = llm.invoke(trimmed_messages)
    return {"messages": [result]}

def route(state):
    last = state["messages"][-1]
    if "finish_reason" in last.response_metadata and last.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
        return "handle_malformed"
    if getattr(last, "tool_calls", None):
        print("Route → tools")
        return "tools"
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip() == "END":
        return END
    print("Route → agent")
    return "agent"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", "agent": "agent", "handle_malformed": "handle_malformed", END: END})
app = graph.compile()

def run_agent(url: str):
    initial_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": url}]
    app.invoke({"messages": initial_messages}, config={"recursion_limit": RECURSION_LIMIT})
    print("Tasks completed successfully!")
