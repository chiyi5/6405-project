@echo off
setlocal
cd /d "%~dp0submission\repository"
if errorlevel 1 goto failed
echo Prepared commit:
git log -1 --oneline
echo.
echo Pushing to https://github.com/chiyi5/6405-project.git main...
git push origin main
if errorlevel 1 goto failed
echo.
echo Push completed successfully.
pause
exit /b 0
:failed
echo.
echo Push did not complete. Keep the error above for diagnosis.
pause
exit /b 1
