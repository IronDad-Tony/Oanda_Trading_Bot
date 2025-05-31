@echo off
echo ==== 備份到GitHub ====
cd /d "%~dp0"

git add .
git commit -m "解決Streamlit運行警告並優化代碼"
git push

echo 備份完成！
pause