@echo off
chcp 65001 >nul
echo ============================================
echo  📦 PyHarness 构建脚本
echo ============================================

REM 1. 清理旧产物
echo [1/4] 清理旧的构建产物...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d"

REM 2. 安装/升级构建工具
echo [2/4] 安装构建工具...
pip install --upgrade build twine -q

REM 3. 构建
echo [3/4] 构建 wheel 和 sdist...
python -m build

REM 4. 验证
echo [4/4] 验证构建产物...
twine check dist/*

echo.
echo ============================================
echo  ✅ 构建完成！产物在 dist/ 目录：
dir /b dist\
echo ============================================
echo.
echo  下一步: 运行 scripts\publish.bat testpypi
echo ============================================
pause
