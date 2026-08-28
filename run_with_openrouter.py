"""Start PyHarness Web UI with OpenRouter nvidia/nemotron-3-ultra-550b-a55b:free."""
import os
from pyharness.factory import build_harness
from pyharness.plugins.web_ui import serve as serve_web_ui

api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

harness = build_harness(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    provider="http",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)
serve_web_ui(harness, host="127.0.0.1", port=3080)
