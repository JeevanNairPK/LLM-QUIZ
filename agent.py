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
MAX_TOKENS = 50000  # Increased slightly for safety

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
# 0.5 req/s = 30 RPM. Safe for Pro Tier but much faster than before.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-pro",  # Pro model for better reasoning
    rate_limiter=rate_limiter
).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = """
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

CRITICAL:
- If a task takes too long, the system will instruct you to submit a WRONG answer. Do this immediately to skip to the next task.
- For GitHub API urls, use `run_code` with `requests`, NOT `get_rendered_html`.
"""

# -------------------------------------------------
# NODES
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
    # --- TIME HANDLING (AGGRESSIVE SKIP) ---
    cur_time = time.time()
    cur_url = os.getenv("url")
    prev_time = url_time.get(cur_url) 
    offset = os.getenv("offset", "0")

    if prev_time is not None:
        prev_time = float(prev_time)
        diff = cur_time - prev_time

        # FAST SKIP: If > 60 seconds, force a wrong answer to move on.
        if diff >= 180 or (offset != "0" and (cur_time - float(offset)) > 180):
            print(f"Timeout exceeded ({diff}s) — instructing LLM to SKIP.")
            fail_instruction = "TIMEOUT: Submit a WRONG answer (e.g. 'skip') immediately to proceed to the next question."
            fail_msg = HumanMessage(content=fail_instruction)
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}
    # ---------------------

    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm, 
    )
    
    # Context Safety Check
    has_human = any(msg.type == "human" for msg in trimmed_messages)
    if not has_human:
        print("WARNING: Context trimmed. Injecting reminder.")
        current_url = os.getenv("url", "Unknown URL")
        reminder = HumanMessage(content=f"Context cleared. Continue processing URL: {current_url}")
        trimmed_messages.append(reminder)

    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    result = llm.invoke(trimmed_messages)
    return {"messages": [result]}

# -------------------------------------------------
# ROUTING
# -------------------------------------------------
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
    
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END

    print("Route → agent")
    return "agent"

# -------------------------------------------------
# GRAPH COMPILE
# -------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges(
    "agent", 
    route,
    {"tools": "tools", "agent": "agent", "handle_malformed": "handle_malformed", END: END}
)

app = graph.compile()

# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_agent(url: str):
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]
    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT}
    )
    print("Tasks completed successfully!")
