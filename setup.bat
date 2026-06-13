@echo off
:: ─────────────────────────────────────────────────────────────
:: NLP Wellness Assistant — Windows Setup
:: Double-click or run: setup.bat
:: ─────────────────────────────────────────────────────────────

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║      NLP Wellness Assistant — Windows Setup          ║
echo ╚══════════════════════════════════════════════════════╝
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt --quiet
echo     Done!

echo.
echo [Kaggle Setup - Optional]
echo   1. Visit: https://www.kaggle.com/settings
echo   2. API section - Create New Token
echo   3. Save kaggle.json to C:\Users\%USERNAME%\.kaggle\kaggle.json
echo.
pause

echo [2/3] Downloading dataset...
python dataset\download_dataset.py

echo.
echo [3/3] Training model...
python backend\train_model.py

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║  Setup Complete!                                     ║
echo ║  Run: python backend\app.py                          ║
echo ║  Then open: frontend\index.html                      ║
echo ╚══════════════════════════════════════════════════════╝
pause
