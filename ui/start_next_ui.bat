@echo off
chcp 65001 >nul
echo ========================================
echo   Next.js Live Trading UI (Dev)
echo ========================================
cd /d "%~dp0\next-live"
set "NEXT_PUBLIC_API_BASE=http://localhost:8000"
start "" "http://localhost:3000"
npm run dev

