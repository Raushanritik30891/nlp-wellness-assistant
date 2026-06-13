#!/bin/bash
# ─────────────────────────────────────────────────────────────
# NLP Wellness Assistant — One-Command Setup
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────

set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      NLP Wellness Assistant — Setup Script           ║"
echo "║      Bhagavad Gita & Puranas · NLP + Flask           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# 1. Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "   ✅ Dependencies installed"

# 2. Kaggle setup reminder
echo ""
echo "🔑 Kaggle API Setup (for richer dataset):"
echo "   1. Visit: https://www.kaggle.com/settings → API → Create New Token"
echo "   2. Move kaggle.json → ~/.kaggle/kaggle.json"
echo "   3. chmod 600 ~/.kaggle/kaggle.json"
echo "   (Press Enter to continue — script works even without Kaggle)"
read -r

# 3. Download dataset
echo ""
echo "📥 Downloading & building dataset..."
python dataset/download_dataset.py

# 4. Train model
echo ""
echo "🧠 Training ML models + generating visualizations..."
python backend/train_model.py

# 5. Done
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                                  ║"
echo "║                                                      ║"
echo "║  To start the API:                                   ║"
echo "║    python backend/app.py                             ║"
echo "║                                                      ║"
echo "║  Then open:                                          ║"
echo "║    frontend/index.html  (in your browser)            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
