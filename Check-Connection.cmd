@echo off
setlocal
echo Keep the SkillBridge server window open while running this check.
echo.
echo === Listener on port 8765 ===
netstat -ano -p tcp | findstr ":8765"
echo.
echo === Direct HTTP request, bypassing proxy ===
curl.exe --noproxy "*" --connect-timeout 5 --max-time 8 -v http://127.0.0.1:8765/api/health
echo.
echo === Alternative localhost address ===
curl.exe --noproxy "*" --connect-timeout 5 --max-time 8 -v http://localhost:8765/api/health
echo.
echo Send the output above if the page still cannot be opened.
pause
