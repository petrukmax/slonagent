@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cls
.venv\Scripts\python -m scripts.diary_enrichment %*
