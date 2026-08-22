@echo off
chcp 65001 >nul
setlocal

set SOURCE=%1
if "%SOURCE%"=="" set SOURCE=pypi

echo ============================================
echo  🔍 验证安装（来源: %SOURCE%）
echo ============================================

REM 创建临时虚拟环境
python -m venv _verify_env
call _verify_env\Scripts\activate.bat

REM 安装
if "%SOURCE%"=="testpypi" (
    echo [1/3] 从 TestPyPI 安装...
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyharness[all] -q
) else (
    echo [1/3] 从 PyPI 安装...
    pip install pyharness[all] -q
)

REM 验证版本
echo [2/3] 验证版本...
pyharness version

REM 验证导入
echo [3/3] 验证导入...
python -c "from pyharness import Harness; from pyharness.plugins.tool_python_exec import PythonExecPlugin; print('✅ 导入成功')"

REM 清理
deactivate
rmdir /s /q _verify_env

echo.
echo ============================================
echo  ✅ 验证通过！
echo ============================================

endlocal
