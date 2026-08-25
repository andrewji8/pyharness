"""Tests for the web tool plugin (tool_web.py).

Covers:
- Tavily search returns formatted results.
- web_search returns a friendly ToolResult when no results are found.
- web_search handles network errors gracefully.
- web_search returns error when TAVILY_API_KEY is missing.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from pyharness.plugins.tool_web import WebPlugin
from pyharness.schema import ToolResultStatus


@pytest.fixture()
def web_plugin() -> WebPlugin:
    return WebPlugin()


@pytest.mark.asyncio
async def test_search_empty_results_returns_ok_with_empty_list(web_plugin: WebPlugin) -> None:
    """web_search with Tavily empty results should return OK with '未找到相关结果'."""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {"answer": "", "results": []}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
        with patch("pyharness.plugins.tool_web.httpx.AsyncClient", return_value=mock_client):
            result = await web_plugin._search({"query": "python 3.13 news"})

    assert result.status == ToolResultStatus.OK
    assert result.output["count"] == 0
    assert result.output["results"] == []
    assert "未找到相关结果" in result.output["text"]


@pytest.mark.asyncio
async def test_search_network_error_returns_friendly_error(web_plugin: WebPlugin) -> None:
    """web_search should return a friendly error when the network fails."""
    import httpx

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
        with patch("pyharness.plugins.tool_web.httpx.AsyncClient", return_value=mock_client):
            result = await web_plugin._search({"query": "python 3.13 news"})

    assert result.status == ToolResultStatus.ERROR
    assert "Tavily 搜索失败" in result.error
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_search_missing_api_key_returns_error(web_plugin: WebPlugin) -> None:
    """web_search should return error when TAVILY_API_KEY is not configured."""
    with patch.dict(os.environ, {}, clear=True):
        result = await web_plugin._search({"query": "python 3.13 news"})

    assert result.status == ToolResultStatus.ERROR
    assert "TAVILY_API_KEY" in result.error
    assert result.output["query"] == "python 3.13 news"


@pytest.mark.asyncio
async def test_search_tavily_formatting(web_plugin: WebPlugin) -> None:
    """web_search should format Tavily response with AI summary and numbered results."""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {
        "answer": "Python 3.13 was released in October 2024.",
        "results": [
            {"title": "Python 3.13 Release", "url": "https://python.org/3.13", "content": "Python 3.13 is now available.", "score": 0.95},
            {"title": "What's New", "url": "https://docs.python.org/3/whatsnew/3.13.html", "content": "Highlights of Python 3.13.", "score": 0.90},
        ],
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
        with patch("pyharness.plugins.tool_web.httpx.AsyncClient", return_value=mock_client):
            result = await web_plugin._search({"query": "python 3.13"})

    assert result.status == ToolResultStatus.OK
    assert result.output["count"] == 2
    assert "【AI 摘要】" in result.output["text"]
    assert "1. Python 3.13 Release" in result.output["text"]
    assert "2. What's New" in result.output["text"]


@pytest.mark.asyncio
async def test_fetch_missing_url_returns_error(web_plugin: WebPlugin) -> None:
    """web_fetch with a missing URL should return an error ToolResult."""
    result = await web_plugin._fetch({})
    assert result.status == ToolResultStatus.ERROR
    assert "url" in result.error.lower()


@pytest.mark.asyncio
async def test_fetch_timeout_returns_error(web_plugin: WebPlugin) -> None:
    """web_fetch should return a friendly error on timeout."""
    import httpx

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("pyharness.plugins.tool_web.httpx.AsyncClient", return_value=mock_client):
        result = await web_plugin._fetch({"url": "http://example.com"})

    assert result.status == ToolResultStatus.ERROR
    assert "超时" in result.error


__all__ = [
    "test_search_empty_results_returns_ok_with_empty_list",
    "test_search_network_error_returns_friendly_error",
    "test_search_missing_api_key_returns_error",
    "test_search_tavily_formatting",
    "test_fetch_missing_url_returns_error",
    "test_fetch_timeout_returns_error",
]
