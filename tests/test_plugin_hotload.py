"""Tests for runtime plugin hot-reload (load / unload / reload).

These exercise the harness-level API directly (no running web server):

* ``load_plugin`` makes a plugin's tools available immediately and survives.
* ``unload_plugin`` removes them; core plugins are protected.
* ``reload_plugin`` re-reads the source file and picks up new behaviour.
* A failed load rolls back and leaves the registry intact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pyharness import Harness
from pyharness.context import open_session
from pyharness.core import (
    PluginCoreProtectedError,
    PluginError,
    PluginNotFoundError,
    _settle,
)
from pyharness.schema import ToolArg, ToolResult, ToolResultStatus, ToolSpec  # noqa: F401

PLUGIN_V1 = '''
from pluggy import HookimplMarker
from pyharness.context import SessionContext
from pyharness.schema import ToolArg, ToolResult, ToolResultStatus, ToolSpec

hookimpl = HookimplMarker("pyharness")


class HelloPlugin:
    @hookimpl
    def get_tool_specs(self, context):
        return (
            ToolSpec(
                name="hello_dyn",
                description="dynamic hello",
                parameters=(ToolArg(name="name", type="string", required=True),),
            ),
        )

    @hookimpl
    async def execute_tool(self, context, tool, arguments):
        if tool.name != "hello_dyn":
            return None
        return ToolResult(
            tool_name="hello_dyn",
            status=ToolResultStatus.OK,
            output={"content": f"hello {arguments.get('name')}"},
        )


PLUGIN = HelloPlugin()
'''

PLUGIN_V2 = PLUGIN_V1.replace('"hello {arguments.get(\'name\')}"', '"HI {arguments.get(\'name\')}"')

PLUGIN_NO_OBJECT = '''
# intentionally has no PLUGIN / create_plugin / @hookimpl class
FOO = 1
'''

PLUGIN_BROKEN = '''
# import-time failure -> must roll back cleanly
raise RuntimeError("boom at import time")
'''


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


async def _collect_tool_names(harness: Harness) -> set[str]:
    async with open_session() as ctx:
        names: set[str] = set()
        for specs in harness.bus.pm.hook.get_tool_specs(context=ctx):
            names.update(s.name for s in specs)
        return names


async def _exec_hello(harness: Harness, name: str) -> str:
    async with open_session() as ctx:
        specs = {}
        for ps in harness.bus.pm.hook.get_tool_specs(context=ctx):
            specs.update({s.name: s for s in ps})
        spec = specs["hello_dyn"]
        results = await _settle(
            harness.bus.pm.hook.execute_tool(context=ctx, tool=spec, arguments={"name": name})
        )
        res = next((r for r in results if r is not None), None)
        assert res is not None
        return res.output["content"]


@pytest.fixture
def harness() -> Harness:
    return Harness()


def test_load_makes_tool_available(harness: Harness, tmp_path: Path) -> None:
    path = _write(tmp_path / "hello_dyn.py", PLUGIN_V1)
    before = harness._plugin_registry.keys()
    assert "hello_dyn" not in before

    result = asyncio.run(harness.load_plugin(path))
    assert result["ok"] is True
    assert result["name"] == "hello_dyn"
    assert "hello_dyn" in asyncio.run(_collect_tool_names(harness))
    assert asyncio.run(_exec_hello(harness, "x")) == "hello x"


def test_unload_removes_tool(harness: Harness, tmp_path: Path) -> None:
    path = _write(tmp_path / "hello_dyn.py", PLUGIN_V1)
    asyncio.run(harness.load_plugin(path))
    assert "hello_dyn" in asyncio.run(_collect_tool_names(harness))

    result = asyncio.run(harness.unload_plugin("hello_dyn"))
    assert result["ok"] is True
    assert "hello_dyn" not in harness._plugin_registry
    assert "hello_dyn" not in asyncio.run(_collect_tool_names(harness))


def test_reload_picks_up_new_behaviour(harness: Harness, tmp_path: Path) -> None:
    path = _write(tmp_path / "hello_dyn.py", PLUGIN_V1)
    asyncio.run(harness.load_plugin(path))
    assert asyncio.run(_exec_hello(harness, "x")) == "hello x"

    _write(Path(path), PLUGIN_V2)
    result = asyncio.run(harness.reload_plugin("hello_dyn"))
    assert result["ok"] is True
    assert asyncio.run(_exec_hello(harness, "x")) == "HI x"


def test_core_plugins_cannot_be_unloaded(harness: Harness) -> None:
    # "llm" entry-point module is a protected core plugin.
    assert "llm" in harness._plugin_registry
    assert harness._plugin_registry["llm"]["core"] is True
    with pytest.raises(PluginCoreProtectedError):
        asyncio.run(harness.unload_plugin("llm"))


def test_unload_unknown_plugin_raises(harness: Harness) -> None:
    with pytest.raises(PluginNotFoundError):
        asyncio.run(harness.unload_plugin("does_not_exist"))


def test_load_exception_rolls_back(harness: Harness, tmp_path: Path) -> None:
    before = dict(harness._plugin_registry)
    count_before = len(harness._plugin_registry)

    broken = _write(tmp_path / "broken.py", PLUGIN_BROKEN)
    with pytest.raises(PluginError):
        asyncio.run(harness.load_plugin(broken))

    # registry must be unchanged and still usable
    assert len(harness._plugin_registry) == count_before
    assert harness._plugin_registry == before
    # a valid plugin can still be loaded afterwards
    good = _write(tmp_path / "hello_dyn.py", PLUGIN_V1)
    asyncio.run(harness.load_plugin(good))
    assert "hello_dyn" in harness._plugin_registry


def test_load_no_plugin_object_raises(harness: Harness, tmp_path: Path) -> None:
    empty = _write(tmp_path / "empty.py", PLUGIN_NO_OBJECT)
    with pytest.raises(PluginError):
        asyncio.run(harness.load_plugin(empty))


def test_list_plugins_reports_core_flag(harness: Harness) -> None:
    plugins = asyncio.run(harness.list_plugins())
    by_name = {p["name"]: p for p in plugins}
    assert by_name["llm"]["core"] is True
    assert by_name["builtin"]["core"] is False
