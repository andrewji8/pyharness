"""Web tools plugin for PyHarness.

Provides web content retrieval tools so agents can fetch and read live web
pages. Uses ``httpx`` for HTTP and ``beautifulsoup4`` for HTML-to-text
extraction.

Tools
-----
* ``web_fetch(url)`` — fetch a URL and return its text content.
* ``web_search(query)`` — search the web using Tavily Search API and return
  the top results with an AI summary. Requires ``TAVILY_API_KEY`` environment
  variable.

Dependencies
------------
* Required: ``httpx``, ``beautifulsoup4``. Install with::

      pip install pyharness[web]
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import ToolArg, ToolResult, ToolResultStatus, ToolSpec

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

try:
    import httpx  # noqa: F401

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup  # noqa: F401

    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BS4_AVAILABLE = False


# SSRF guard: loopback / private / link-local / cloud-metadata ranges are never
# fetched. Redirects are re-validated on every hop (see ``_fetch``).
_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
        "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.168.0.0/16",
        "198.18.0.0/15", "224.0.0.0/4", "240.0.0.0/4",
        "::1/128", "fc00::/7", "fe80::/10",
    )
)
_MAX_REDIRECTS = 3


def _assert_public_http_url(url: str) -> None:
    """Raise ``PermissionError`` unless ``url`` is http(s) to a public address.

    Every A/AAAA record of the host is resolved and checked, so a DNS name
    pointing at an internal address is blocked just like a literal IP.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PermissionError(f"仅允许 http/https 协议: {url!r}")
    host = parsed.hostname
    if not host:
        raise PermissionError(f"URL 缺少主机名: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise PermissionError(f"无法解析主机 {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if any(ip in net for net in _BLOCKED_NETWORKS):
            raise PermissionError(f"禁止访问内网/保留地址: {host} -> {ip}")


class WebPlugin:
    """Web content retrieval tool provider.

    Parameters
    ----------
    timeout:
        HTTP request timeout in seconds.
    max_content_chars:
        Maximum characters to return from a fetched page to avoid token
        explosions.
    user_agent:
        User-Agent header sent with requests.

    Notes
    -----
    If ``httpx`` or ``beautifulsoup4`` is not installed, this plugin will
    silently register no tools and emit a warning on construction. This
    allows the entry-point to be loaded safely even when the ``web`` extra
    is not installed.
    """

    def __init__(
        self,
        timeout: float = 15.0,
        max_content_chars: int = 8000,
        user_agent: str = "PyHarness/0.1 (web-tool)",
    ) -> None:
        self.timeout = timeout
        self.max_content_chars = max_content_chars
        self.user_agent = user_agent
        self._available = _HTTPX_AVAILABLE and _BS4_AVAILABLE
        if not self._available:
            import warnings

            warnings.warn(
                "Web 插件依赖未安装，请运行: pip install pyharness[web]；"
                "web_fetch / web_search 工具将不会注册。",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # Tool specs
    # ------------------------------------------------------------------ #
    def _fetch_spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_fetch",
            description="抓取指定 URL 的网页内容并提取正文文本。",
            parameters=(
                ToolArg(name="url", type="string", description="要抓取的完整 URL", required=True),
            ),
            timeout_seconds=self.timeout,
        )

    def _search_spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description=(
                "使用 Tavily Search API 搜索网页并返回前 5 条结果摘要。"
                "需要环境变量 TAVILY_API_KEY。"
            ),
            parameters=(
                ToolArg(name="query", type="string", description="搜索关键词", required=True),
            ),
            timeout_seconds=self.timeout,
        )

    @hookimpl
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        if not self._available:
            return ()
        return (self._fetch_spec(), self._search_spec())

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    @hookimpl
    async def execute_tool(
        self, context: SessionContext, tool: ToolSpec, arguments: dict[str, object]
    ) -> ToolResult | None:
        if not self._available:
            return None
        if tool.name == "web_fetch":
            return await self._fetch(arguments)
        if tool.name == "web_search":
            return await self._search(arguments)
        return None

    # ------------------------------------------------------------------ #
    # web_fetch
    # ------------------------------------------------------------------ #
    async def _fetch(self, arguments: dict[str, object]) -> ToolResult:
        url = str(arguments.get("url", ""))
        if not url:
            return ToolResult(tool_name="web_fetch", status=ToolResultStatus.ERROR, error="缺少 'url' 参数。", output={})

        try:
            async with httpx.AsyncClient(
                follow_redirects=False,  # redirects re-validated hop by hop (SSRF)
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            ) as client:
                current = url
                for _ in range(_MAX_REDIRECTS + 1):
                    # Validate EVERY hop: a redirect can point back inside the
                    # private network even when the first URL was public.
                    _assert_public_http_url(current)
                    resp = await client.get(current)
                    if resp.is_redirect:
                        location = resp.headers.get("location", "")
                        if not location:
                            break
                        current = urljoin(current, location)
                        continue
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    text = self._extract_text(resp.text, content_type)
                    if len(text) > self.max_content_chars:
                        text = text[: self.max_content_chars] + f"\n\n...[已截断，共 {len(text)} 字符]"
                    return ToolResult(
                        tool_name="web_fetch",
                        status=ToolResultStatus.OK,
                        output={"url": current, "title": self._guess_title(resp.text), "content": text, "content_type": content_type},
                    )
                return ToolResult(
                    tool_name="web_fetch",
                    status=ToolResultStatus.ERROR,
                    error=f"重定向次数超过上限（{_MAX_REDIRECTS}）。",
                    output={"url": url},
                )
        except PermissionError as exc:
            logger.warning("web_fetch blocked: %s", exc)
            return ToolResult(tool_name="web_fetch", status=ToolResultStatus.ERROR, error=f"已拦截: {exc}", output={"url": url})
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                tool_name="web_fetch", status=ToolResultStatus.ERROR, error=f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}", output={"url": url}
            )
        except httpx.TimeoutException:
            return ToolResult(tool_name="web_fetch", status=ToolResultStatus.ERROR, error="请求超时。可能是网络不稳定或目标服务器响应缓慢。", output={"url": url})
        except Exception as exc:
            logger.exception("web_fetch failed")
            return ToolResult(tool_name="web_fetch", status=ToolResultStatus.ERROR, error=f"抓取失败: {exc}", output={"url": url})

    # ------------------------------------------------------------------ #
    # web_search (Tavily)
    # ------------------------------------------------------------------ #
    async def _search(self, arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query", ""))
        if not query:
            return ToolResult(tool_name="web_search", status=ToolResultStatus.ERROR, error="缺少 'query' 参数。", output={})

        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return ToolResult(
                tool_name="web_search",
                status=ToolResultStatus.ERROR,
                error="错误: 未配置 TAVILY_API_KEY（请写入 .env）",
                output={"query": query},
            )

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 5,
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return ToolResult(
                tool_name="web_search",
                status=ToolResultStatus.ERROR,
                error=f"Tavily 搜索失败: {type(exc).__name__}: {exc}",
                output={"query": query},
            )

        results = data.get("results") or []
        parts = []
        if data.get("answer"):
            parts.append(f"【AI 摘要】{data['answer']}")
        for i, r in enumerate(results, 1):
            parts.append(
                f"{i}. {r.get('title', '')}\n"
                f"   {r.get('url', '')}\n"
                f"   {(r.get('content') or '')[:300]}"
            )
        output_text = "\n\n".join(parts) if parts else "未找到相关结果"
        return ToolResult(
            tool_name="web_search",
            status=ToolResultStatus.OK,
            output={"query": query, "results": results, "count": len(results), "text": output_text},
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(html: str, content_type: str) -> str:
        if "text/html" not in content_type and not html.lstrip().startswith("<"):
            return html

        if not _BS4_AVAILABLE:
            raise RuntimeError(
                "需要安装 beautifulsoup4 才能提取网页内容。"
                "请运行: pip install pyharness[web]"
            )

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style/nav elements.
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    @staticmethod
    def _guess_title(html: str) -> str:
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


__all__ = ["WebPlugin"]
