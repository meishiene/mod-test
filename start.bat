@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul 2>nul
title ML Test Platform
cd /d "%~dp0"

rem One-click Windows launcher.
rem   start.bat
rem   start.bat --host 127.0.0.1 --port 8085
rem   start.bat --bootstrap-only
rem   start.bat --check
rem
rem Python 3.12+ is located automatically. If no suitable Python is found,
rem this script tries to install Python 3.12 with winget. All remaining
rem bootstrap work is done by scripts\start_windows.py.

set "ROOT=%~dp0"
set "PYTHON_EXE="
set "PYTHON_ARGS="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if defined PYTHON_BIN call :try "%PYTHON_BIN%"
if not defined PYTHON_EXE call :find_python
if not defined PYTHON_EXE call :install_python
if not defined PYTHON_EXE goto :no_python

"%PYTHON_EXE%" %PYTHON_ARGS% "%ROOT%scripts\start_windows.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Startup failed with exit code %EXIT_CODE%.
    echo Check the messages above, then run start.bat again.
    pause
)
exit /b %EXIT_CODE%


:find_python
if defined PYTHON_EXE exit /b 0

rem Prefer the Python launcher and the default per-user/all-users install paths.
rem 3.14+ is tried last: some native dependencies publish Windows wheels later.
call :try "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" -3.13
call :try "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" -3.12
call :try "%SystemRoot%\py.exe" -3.13
call :try "%SystemRoot%\py.exe" -3.12
for %%V in (313 312 311 314) do (
    call :try "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
    call :try "%ProgramFiles%\Python%%V\python.exe"
    call :try "C:\Python%%V\python.exe"
)
call :try "%SystemRoot%\py.exe" -3
call :try_where python
call :try_where python3
exit /b 0


:try_where
rem Skip the Microsoft Store alias: running it without Python installed
rem opens the Store instead of reporting a usable interpreter.
for /f "delims=" %%P in ('where %~1 2^>nul') do (
    echo "%%P" | findstr /i /c:"WindowsApps" >nul
    if errorlevel 1 call :try "%%P"
)
exit /b 0


:try
if defined PYTHON_EXE exit /b 0
if "%~1"=="" exit /b 0
"%~1" %~2 -c "import sys;sys.exit(0 if sys.version_info >= (3,12) and sys.maxsize == 2**63 - 1 else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_EXE=%~1"
set "PYTHON_ARGS=%~2"
exit /b 0


:install_python
if defined PYTHON_EXE exit /b 0
echo.
echo [bootstrap] Python 3.12+ not found.
if /i "%MLTP_AUTO_INSTALL_PYTHON%"=="0" (
    echo [bootstrap] MLTP_AUTO_INSTALL_PYTHON=0, skipping automatic installation.
    exit /b 0
)
where winget >nul 2>nul
if errorlevel 1 (
    echo [bootstrap] winget is not available on this machine.
    exit /b 0
)
echo [bootstrap] Installing Python 3.12 with winget, please wait...
winget install --id Python.Python.3.12 -e --source winget --scope user --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo [bootstrap] Per-user install failed, retrying with the default scope...
    winget install --id Python.Python.3.12 -e --source winget --silent --accept-package-agreements --accept-source-agreements
)
call :try "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" -3.12
call :try "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :try "%ProgramFiles%\Python312\python.exe"
call :try "%SystemRoot%\py.exe" -3.12
call :try "%SystemRoot%\py.exe" -3
exit /b 0


:no_python
echo.
echo [ERROR] Python 3.12+ is required but was not found.
echo.
echo Install Python from:
echo   https://www.python.org/downloads/windows/
echo During setup, enable "Add python.exe to PATH", then run start.bat again.
echo.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1
