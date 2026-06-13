# 🕉️ NLP Wellness Assistant
### AI-powered emotional support using Bhagavad Gita & Puranas wisdom

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://kaggle.com)

---

## 📌 What It Does

User types how they feel → system classifies emotion → retrieves matching scripture verse → returns wellness guidance.

```
"I am feeling very stressed about work"
        ↓
Emotion: Stress (86% confidence)
Verse:   Bhagavad Gita Ch.2 V.47 — "You have a right to your actions, not the fruits..."
Guidance: 3 actionable wellness tips
```

---

## 🚀 Quick Start (3 Steps)

### Option A — Automated (Linux/Mac)
```bash
bash setup.sh
```

### Option B — Manual
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Kaggle dataset + build training data
#    (works with OR without Kaggle credentials — see setup below)
python dataset/download_dataset.py

# 3. Train models + generate visualizations
python backend/train_model.py

# 4. Start API
python backend/app.py          # → http://localhost:5000

# 5. Open frontend
open frontend/index.html       # or double-click it
```

---

## 🔑 Kaggle Setup (Optional but Recommended)

The project works **without** Kaggle (uses 31 curated verses).  
With Kaggle credentials, it downloads 3 richer datasets automatically.

```
1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" → Click "Create New Token"
3. This downloads kaggle.json

Linux/Mac:
   mv kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json

Windows:
   Move kaggle.json → C:\Users\<YourName>\.kaggle\kaggle.json

Then run: python dataset/download_dataset.py
```

**Kaggle Datasets Downloaded:**
| Dataset | Link |
|---------|------|
| Bhagavad Gita | kaggle.com/datasets/dumanmesut/bhagavad-gita |
| Bhagavad Gita All Languages | kaggle.com/datasets/rajupalepu/bhagavad-gita-all-languages |
| Hindu Scriptures | kaggle.com/datasets/drewgjerstad/hindu-scriptures |

---

## 📁 Project Structure

```
nlp-wellness-assistant/
├── dataset/
│   ├── download_dataset.py   ← Kaggle downloader + dataset builder
│   ├── create_dataset.py     ← Standalone fallback dataset builder
│   ├── verses_db.csv         ← Generated verse database
│   └── wellness_dataset.csv  ← Generated training data
│
├── backend/
│   ├── app.py                ← Flask REST API
│   ├── train_model.py        ← ML training pipeline
│   ├── utils.py              ← Preprocessing + verse retrieval
│   ├── model.pkl             ← Trained model (after training)
│   ├── vectorizer.pkl        ← TF-IDF vectorizer (after training)
│   └── model_meta.json       ← Metrics & model info
│
├── frontend/
│   └── index.html            ← Complete single-file UI
│
├── notebooks/
│   └── INTERVIEW_QA.md       ← 20 Interview Q&A
│
├── screenshots/              ← Auto-generated visualizations
│   ├── emotion_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_lr.png
│   ├── word_frequency.png
│   └── word_cloud_viz.png
│
├── setup.sh                  ← Linux/Mac one-command setup
├── setup.bat                 ← Windows one-command setup
├── requirements.txt
└── README.md
```

---

## 🔌 API Reference

### `POST /predict`
```json
// Request
{ "text": "I am feeling stressed" }

// Response
{
  "emotion": "Stress",
  "confidence": "86%",
  "emotion_icon": "😣",
  "top_verse": {
    "source": "Bhagavad Gita",
    "chapter": 2, "verse_num": 47,
    "sanskrit": "karmaṇy evādhikāras te...",
    "english": "You have a right to perform your prescribed duties...",
    "meaning": "Focus only on your actions, not on results...",
    "relevance": "78%"
  },
  "related_verses": [...],
  "guidance": ["tip 1", "tip 2", "tip 3"]
}
```

### Other Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check + model info |
| `GET /model-info` | Accuracy, F1, Precision, Recall |
| `GET /emotions` | List of all 6 emotion categories |
| `GET /plot/<name>` | Visualization PNGs |

---

## 🤖 ML Pipeline

| Step | Component | Detail |
|------|-----------|--------|
| Features | TF-IDF | Unigrams + bigrams, sublinear_tf, 3000 features |
| Model 1 | Logistic Regression | C=1.0, lbfgs solver |
| Model 2 | Multinomial Naive Bayes | alpha=0.5 |
| Best Model | **Naive Bayes** | Higher F1 on small dataset |
| Retrieval | TF-IDF Cosine Similarity | + 1.3× emotion-matching boost |

---

## 📊 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 58.1% | 58.9% |
| **Naive Bayes ✅** | **67.7%** | **67.6%** |

---

## 🎤 Interview One-Liner

> "I built an NLP Wellness Assistant that classifies user emotions into 6 categories using TF-IDF + Naive Bayes, then retrieves contextually relevant Bhagavad Gita verses using cosine similarity with an emotion-aware boosting heuristic — all without LLMs, vector databases, or GPU."

---

*ॐ — Built with classical NLP, Kaggle datasets, and ancient wisdom*
"# nlp-wellness-assistant" 
"# nlp-wellness-assistant" 
"# nlp-wellness-assistant" 
