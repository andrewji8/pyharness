"""Shared pytest fixtures for PyHarness tests."""

from __future__ import annotations

import pytest

from pyharness.plugins.llm import entry as llm
from pyharness.schema import AgentConfig, HarnessConfig, LLMResponse


@pytest.fixture()
def mock_llm():
    """Fixture that provides a network-free dummy LLM provider.

    Usage::

        async def test_something(mock_llm):
            h = Harness(config=HarnessConfig(auto_load_entry_points=False))
            h.register_plugin(mock_llm)
            h.initialize()
    """
    llm.clear()
    provider = llm.use_dummy(
        models=("mock-model",),
        plan=[LLMResponse(model="mock-model", content="Mock LLM response")],
    )
    return provider
