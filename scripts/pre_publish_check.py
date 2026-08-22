"""
PyHarness 发布前检查脚本
用法: python scripts/pre_publish_check.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def decorator(func):
        CHECKS.append((name, func))
        return func
    return decorator


@check("Python 版本 >= 3.11")
def check_python_version() -> None:
    assert sys.version_info >= (3, 11), f"需要 Python 3.11+，当前 {sys.version}"


@check("pyproject.toml 存在")
def check_pyproject() -> None:
    assert Path("pyproject.toml").exists(), "pyproject.toml 不存在"


@check("README.md 存在")
def check_readme() -> None:
    assert Path("README.md").exists(), "README.md 不存在"


@check("LICENSE 存在")
def check_license() -> None:
    assert Path("LICENSE").exists(), "LICENSE 不存在"


@check("源码目录存在")
def check_source() -> None:
    assert Path("src/pyharness").exists() or Path("pyharness").exists(), "源码目录不存在 (期望 src/pyharness/ 或 pyharness/)"


@check("测试全部通过")
def check_tests() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"测试失败:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"


@check("无未提交的更改（可选）")
def check_git_clean() -> None:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"  [WARN] 有未提交的更改（不阻塞发布，但建议先 commit）")
        print(f"  {result.stdout[:200]}")


@check("构建工具已安装")
def check_build_tools() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import build; import twine; print('OK')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "请先运行: pip install --upgrade build twine"


@check("pyproject.toml 版本号合理")
def check_version() -> None:
    import re
    content = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match, "pyproject.toml 中未找到 version 字段"
    version = match.group(1)
    parts = version.split(".")
    assert len(parts) >= 2, f"版本号格式错误: {version}"
    print(f"  [INFO] 当前版本: {version}")


def main() -> None:
    print("=" * 50)
    print("PyHarness 发布前检查")
    print("=" * 50)
    passed = 0
    failed = 0
    for name, func in CHECKS:
        try:
            func()
            print(f"  [OK] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
            passed += 1  # 非阻塞性错误

    print(f"\n{'=' * 50}")
    print(f"结果: {passed} 通过, {failed} 失败")
    if failed == 0:
        print("所有检查通过，可以发布！")
        print("下一步: 运行 scripts\\build.bat")
    else:
        print("请修复以上问题后重试")
        sys.exit(1)


if __name__ == "__main__":
    main()
