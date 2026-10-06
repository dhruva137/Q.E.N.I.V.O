@echo off
rem Start the QENIVO planner console on this machine only (http://127.0.0.1:8765).
rem Audit trail goes to %LOCALAPPDATA%\QENIVO\audit (kept 180 days).
set AUDIT=%LOCALAPPDATA%\QENIVO\audit
if not exist "%AUDIT%" mkdir "%AUDIT%"
start "" http://127.0.0.1:8765
qenivo serve --port 8765 --audit-dir "%AUDIT%"
if errorlevel 1 python -m qenivo serve --port 8765 --audit-dir "%AUDIT%"
pause
