"""LLM provider plugin package.

Contains the :class:`~pyharness.plugins.llm.provider.Provider` abstraction, a
network-free :class:`~pyharness.plugins.llm.dummy.DummyProvider`, an OpenAI-
compatible :class:`~pyharness.plugins.llm.http.HTTPProvider`, and the pluggy
hook surface in :mod:`pyharness.plugins.llm.entry`.
"""