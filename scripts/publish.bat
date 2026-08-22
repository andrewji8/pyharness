@echo off
chcp 65001 >nul
setlocal

set TARGET=%1

if "%TARGET%"=="" (
    echo 用法: scripts\publish.bat [testpypi^|pypi]
    echo.
    echo   testpypi  - 上传到 TestPyPI（测试）
    echo   pypi      - 上传到正式 PyPI（发布）
    exit /b 1
)

if "%TARGET%"=="testpypi" (
    echo ============================================
    echo  🧪 上传到 TestPyPI（测试环境）
    echo ============================================
    twine upload --repository testpypi dist/*
    echo.
    echo  验证安装:
    echo  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyharness[all]
) else if "%TARGET%"=="pypi" (
    echo ============================================
    echo  🚀 上传到正式 PyPI
    echo ============================================
    echo  ⚠️  确认版本号正确！此操作不可撤销！
    echo.
    set /p CONFIRM="确认发布？(y/N): "
    if /i not "%CONFIRM%"=="y" (
        echo 已取消。
        exit /b 0
    )
    twine upload dist/*
    echo.
    echo  🎉 发布成功！
    echo  查看: https://pypi.org/project/pyharness/
    echo.
    echo  验证安装:
    echo  pip install pyharness[all]
) else (
    echo 未知目标: %TARGET%
    echo 用法: scripts\publish.bat [testpypi^|pypi]
    exit /b 1
)

endlocal
