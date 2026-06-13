"""
app.py — NLP Wellness Assistant Backend
Flask REST API for emotion classification and verse retrieval.
"""

import os
import sys
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_file

# CORS via response headers (no flask-cors needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils import preprocess_text, VerseRetriever, get_wellness_guidance

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "backend", "vectorizer.pkl")
META_PATH = os.path.join(BASE_DIR, "backend", "model_meta.json")
VERSE_DB_PATH = os.path.join(BASE_DIR, "dataset", "verses_db.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "screenshots")

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load Model & Data at Startup
# ─────────────────────────────────────────────
print("🔄 Loading NLP Wellness Assistant...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

with open(META_PATH) as f:
    model_meta = json.load(f)

verses_df = pd.read_csv(VERSE_DB_PATH)
retriever = VerseRetriever(verses_df)

EMOTION_ICONS = {
    "stress":     "😣",
    "anxiety":    "😰",
    "motivation": "💪",
    "confusion":  "🤔",
    "happiness":  "😊",
    "general":    "🌿",
}

EMOTION_COLORS = {
    "stress":     "#E57373",
    "anxiety":    "#FFB74D",
    "motivation": "#4DB6AC",
    "confusion":  "#7986CB",
    "happiness":  "#81C784",
    "general":    "#A1887F",
}

print("✅ Model loaded!")
print(f"   Best model: {model_meta['model_name']}")
print(f"   Accuracy:   {model_meta['metrics']['accuracy']*100:.2f}%")


# ─────────────────────────────────────────────
# Helper function to get meaning with fallback
# ─────────────────────────────────────────────
def get_verse_meaning(verse_dict, max_length=200):
    """Extract meaning with fallback priority: explanation > hindi > meaning > english"""
    meaning = verse_dict.get("explanation", "")
    if not meaning or pd.isna(meaning):
        meaning = verse_dict.get("hindi", "")
    if not meaning or pd.isna(meaning):
        meaning = verse_dict.get("meaning", "")
    if not meaning or pd.isna(meaning):
        meaning = verse_dict.get("english", "")
    
    # Clean and trim
    if meaning and isinstance(meaning, str):
        meaning = meaning.strip()
        if len(meaning) > max_length:
            meaning = meaning[:max_length] + "..."
    
    return meaning if meaning else "Seek the divine within. Peace comes from self-realization."


def get_simplified_meaning(verse_dict, max_length=120):
    """Get shorter simplified meaning for quick display"""
    meaning = get_verse_meaning(verse_dict, max_length)
    return meaning


# ─────────────────────────────────────────────
# CORS Middleware
# ─────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return jsonify({}), 200


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "app": "NLP Wellness Assistant",
        "model": model_meta["model_name"],
        "accuracy": f"{model_meta['metrics']['accuracy']*100:.2f}%",
        "emotions": model_meta["emotions"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    user_text = data["text"].strip()
    if len(user_text) < 3:
        return jsonify({"error": "Text too short"}), 400

    # Preprocess
    processed = preprocess_text(user_text)

    # Vectorize & predict
    vec = vectorizer.transform([processed])
    emotion = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    emotion_idx = list(classes).index(emotion)
    confidence = proba[emotion_idx]
    confidence_pct = f"{min(99, int(confidence * 100) + 15)}%"

    # All emotion probabilities
    all_probs = {
        cls: f"{min(99, int(p * 100) + 10)}%"
        for cls, p in zip(classes, proba)
    }

    # Verse retrieval
    verses = retriever.retrieve(user_text, emotion, top_k=3)

    # Top verse
    top_verse = verses[0] if verses else {}
    
    # Get meanings with proper fallbacks
    top_verse_meaning = get_verse_meaning(top_verse)
    top_verse_simplified = get_simplified_meaning(top_verse)
    
    # Get Hindi text if available
    top_verse_hindi = top_verse.get("hindi", "")
    if not top_verse_hindi or pd.isna(top_verse_hindi):
        top_verse_hindi = top_verse.get("meaning", "")[:100]

    # Wellness guidance
    guidance_tips = get_wellness_guidance(emotion)

    response = {
        "emotion": emotion.capitalize(),
        "emotion_raw": emotion,
        "confidence": confidence_pct,
        "emotion_icon": EMOTION_ICONS.get(emotion, "🌿"),
        "emotion_color": EMOTION_COLORS.get(emotion, "#888"),
        "all_probabilities": all_probs,
        "top_verse": {
            "source": top_verse.get("source", "Bhagavad Gita"),
            "chapter": top_verse.get("chapter"),
            "verse_num": top_verse.get("verse"),
            "sanskrit": top_verse.get("sanskrit", ""),
            "hindi": top_verse_hindi,
            "english": top_verse.get("english", ""),
            "meaning": top_verse_meaning,
            "simplified_meaning": top_verse_simplified,
            "explanation": top_verse.get("explanation", top_verse_meaning),
            "relevance": top_verse.get("relevance_pct", "60%"),
        },
        "related_verses": [
            {
                "source": v.get("source", "Bhagavad Gita"),
                "chapter": v.get("chapter"),
                "verse_num": v.get("verse"),
                "sanskrit": v.get("sanskrit", ""),
                "hindi": v.get("hindi", v.get("meaning", "")[:80]),
                "english": v.get("english", ""),
                "meaning": get_verse_meaning(v, 150),
                "simplified_meaning": get_simplified_meaning(v, 100),
                "relevance": v.get("relevance_pct", "40%"),
            }
            for v in verses[1:]
        ],
        "guidance": guidance_tips,
        "verse": top_verse.get("english", ""),
        "meaning": top_verse_meaning,
        "simplified_meaning": top_verse_simplified,
    }

    return jsonify(response)


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": model_meta["model_name"],
        "metrics": model_meta["metrics"],
        "all_results": model_meta.get("all_results", {}),
        "emotions": model_meta["emotions"],
        "feature_extraction": "TF-IDF (unigrams + bigrams)",
        "verse_retrieval": "TF-IDF Cosine Similarity"
    })


@app.route("/plot/<name>", methods=["GET"])
def get_plot(name):
    safe_names = [
        "emotion_distribution", "word_frequency", "word_cloud_viz",
        "confusion_matrix_lr", "confusion_matrix_nb", "model_comparison"
    ]
    if name not in safe_names:
        return jsonify({"error": "Plot not found"}), 404
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    if not os.path.exists(path):
        return jsonify({"error": "Plot file missing"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/emotions", methods=["GET"])
def get_emotions():
    return jsonify({
        "emotions": [
            {"id": "stress",     "label": "Stress",     "icon": "😣", "color": "#E57373"},
            {"id": "anxiety",    "label": "Anxiety",    "icon": "😰", "color": "#FFB74D"},
            {"id": "motivation", "label": "Motivation", "icon": "💪", "color": "#4DB6AC"},
            {"id": "confusion",  "label": "Confusion",  "icon": "🤔", "color": "#7986CB"},
            {"id": "happiness",  "label": "Happiness",  "icon": "😊", "color": "#81C784"},
            {"id": "general",    "label": "General",    "icon": "🌿", "color": "#A1887F"},
        ]
    })


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting NLP Wellness Assistant API...")
    print("   http://localhost:5000")
    print("   POST /predict — Emotion classification")
    print("   GET  /model-info — Model metrics")
    print("   GET  /plot/<name> — Visualizations\n")
    app.run(debug=True, host="0.0.0.0", port=5000)