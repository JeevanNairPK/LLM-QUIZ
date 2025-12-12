from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin

@tool
def get_rendered_html(url: str) -> dict:
    """
    Fetch and return the fully rendered HTML of a webpage.
    """
    # --- SAFETY INTERVENTION ---
    # Prevents memory crashes on API/JSON endpoints
    if "api.github.com" in url or url.endswith(".json"):
        return {
            "error": (
                "INTERVENTION: You are trying to render a JSON/API URL with a browser. "
                "This will crash your memory. "
                "STOP. Use the `run_code` tool instead to fetch this data using "
                "`import requests; data = requests.get(url).json()`."
            )
        }
    # ---------------------------

    print("\nFetching and rendering:", url)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # 30s timeout to fail fast
            page.goto(url, wait_until="networkidle", timeout=30000)
            content = page.content()
            browser.close()

            soup = BeautifulSoup(content, "html.parser")
            imgs = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
            
            if len(content) > 300000:
                    print("Warning: HTML too large, truncating...")
                    content = content[:300000] + "... [TRUNCATED]"
            
            return {
                "html": content,
                "images": imgs,
                "url": url
            }

    except Exception as e:
        return {"error": f"Error fetching/rendering page: {str(e)}"}
