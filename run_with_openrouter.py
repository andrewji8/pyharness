"""Start PyHarness Web UI with OpenRouter nvidia/nemotron-3-ultra-550b-a55b:free."""
import os
from pyharness import Harness
from pyharness.plugins.llm.entry import use_http
from pyharness.plugins.web_ui import serve as serve_web_ui

api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

# Configure OpenRouter provider
use_http(
    models=("nvidia/nemotron-3-ultra-550b-a55b:free",),
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Start Web UI
harness = Harness()
serve_web_ui(harness, host="127.0.0.1", port=3080)
