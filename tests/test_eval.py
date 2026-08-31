"""Tests for the eval framework."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from pyharness import Harness
from pyharness.plugins.eval_runner import (
    EvalReport,
    EvalResult,
    EvalTask,
    _parse_judge_json,
    generate_json_report,
    generate_markdown_report,
    load_suite,
    run_evals,
)
from pyharness.plugins.llm import entry as llm
from pyharness.schema import AgentConfig, LLMResponse


def _setup_harness():
    from pyharness.schema import HarnessConfig
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    llm.clear()
    llm.use_dummy(models=("dummy",))
    h.initialize()
    return h


def test_parse_judge_json_valid():
    text = '{"scores": {"correctness": 4, "tool_usage": 5, "safety": 5}, "reasoning": "Good"}'
    scores, reasoning = _parse_judge_json(text)
    assert scores == {"correctness": 4, "tool_usage": 5, "safety": 5}
    assert reasoning == "Good"


def test_parse_judge_json_malformed_fallback():
    text = 'Here is my assessment: {"scores": {"correctness": 3, "tool_usage": 2, "safety": 5}, "reasoning": "OK"}'
    scores, reasoning = _parse_judge_json(text)
    assert scores == {"correctness": 3, "tool_usage": 2, "safety": 5}


def test_parse_judge_json_invalid_returns_empty():
    text = "This is not JSON at all."
    scores, reasoning = _parse_judge_json(text)
    assert scores == {}


def test_load_suite():
    suite_path = Path(__file__).parent.parent / "evals" / "basic.yaml"
    tasks = load_suite(suite_path)
    assert len(tasks) >= 8
    assert any(t.id == "code_prime" for t in tasks)
    assert any(t.id == "safety_refuse" for t in tasks)


@pytest.mark.asyncio
async def test_eval_dummy_provider_end_to_end():
    """End-to-end eval with dummy provider on 2 tasks."""
    h = _setup_harness()

    # Create a temp suite with 2 simple tasks
    suite_data = {
        "suite": "test",
        "version": "1.0",
        "tasks": [
            {
                "id": "dummy_echo",
                "category": "qa",
                "prompt": "Say 'hello eval'",
                "timeout": 30,
                "checks": [{"type": "contains", "value": "hello eval"}],
                "rubric": "Must contain 'hello eval'",
            },
            {
                "id": "dummy_math",
                "category": "code_exec",
                "prompt": "What is 2+2?",
                "timeout": 30,
                "checks": [{"type": "contains", "value": "4"}],
                "rubric": "Must answer 4",
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        import yaml
        yaml.dump(suite_data, f)
        suite_path = f.name

    try:
        report = await run_evals(suite_path=suite_path, judge_model="dummy")
        assert report.suite == Path(suite_path).name
        assert len(report.results) == 2
        for r in report.results:
            assert r.task_id in ("dummy_echo", "dummy_math")
            assert isinstance(r.programmatic_pass, bool)
    finally:
        os.unlink(suite_path)


def test_generate_json_report(tmp_path):
    report = EvalReport(
        suite="test",
        model="dummy",
        timestamp="2024-01-01T00:00:00Z",
        results=[
            EvalResult(
                task_id="t1",
                category="qa",
                passed=True,
                programmatic_pass=True,
                judge_scores={"correctness": 4, "tool_usage": 3, "safety": 5},
                judge_reasoning="OK",
                response="hello",
                tool_trace=[],
                error="",
                duration_seconds=1.0,
            )
        ],
        passed_count=1,
        failed_count=0,
    )
    out = tmp_path / "report.json"
    generate_json_report(report, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["suite"] == "test"
    assert len(data["results"]) == 1


def test_generate_markdown_report(tmp_path):
    report = EvalReport(
        suite="test",
        model="dummy",
        timestamp="2024-01-01T00:00:00Z",
        results=[
            EvalResult(
                task_id="t1",
                category="qa",
                passed=True,
                programmatic_pass=True,
                judge_scores={"correctness": 4, "tool_usage": 3, "safety": 5},
                judge_reasoning="OK",
                response="hello",
                tool_trace=[],
                error="",
                duration_seconds=1.0,
            )
        ],
        passed_count=1,
        failed_count=0,
    )
    out = tmp_path / "report.md"
    generate_markdown_report(report, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "# Eval Report: test" in content
    assert "t1" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
