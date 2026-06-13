"""
utils.py — NLP Wellness Assistant (Updated for Hindi+Sanskrit dataset)
Preprocessing, verse retrieval, and guidance generation utilities.
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# Simple stopwords (no NLTK dependency)
# ─────────────────────────────────────────────
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "not", "no", "nor",
    "and", "but", "or", "so", "yet", "both", "either", "neither",
    "for", "of", "to", "in", "on", "at", "by", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "a", "an", "the", "as", "if", "than",
    "because", "while", "although", "since", "unless", "until", "when",
    "where", "how", "all", "each", "every", "any", "just", "very",
    "too", "also", "more", "most", "other", "some", "such", "own", "same",
    "so", "now", "then", "here", "there", "few", "much", "many",
}

# Basic lemmatization map
LEMMA_MAP = {
    "stressed": "stress", "stressing": "stress", "stressful": "stress",
    "anxious": "anxiety", "anxiously": "anxiety", "worrying": "worry",
    "worried": "worry", "worries": "worry", "fearful": "fear",
    "fearfully": "fear", "fears": "fear", "feared": "fear",
    "motivated": "motivation", "motivate": "motivation", "motivating": "motivation",
    "demotivated": "demotivation", "uninspired": "uninspired",
    "confused": "confusion", "confusing": "confusion", "confuse": "confusion",
    "happy": "happiness", "happily": "happiness", "joyful": "joy",
    "joyfully": "joy", "enjoying": "enjoy", "enjoyed": "enjoy",
    "feeling": "feel", "feels": "feel", "felt": "feel",
    "losing": "lose", "lost": "lose",
    "overwhelming": "overwhelm", "overwhelmed": "overwhelm",
    "burned": "burnout", "exhausted": "exhaustion", "drained": "drain",
    "scared": "fear", "frightened": "fear", "nervous": "nervousness",
    "panicking": "panic", "panicked": "panic",
    "achieving": "achieve", "achieved": "achieve",
    "struggling": "struggle", "struggled": "struggle",
    "needs": "need", "needed": "need", "wanting": "want",
    "finding": "find", "seeks": "seek", "seeking": "seek",
    "तनाव": "stress", "चिंता": "anxiety", "प्रेरणा": "motivation",
    "भ्रम": "confusion", "खुशी": "happiness", "सामान्य": "general",
}


def preprocess_text(text: str) -> str:
    """Full NLP preprocessing pipeline."""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMA_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# Wellness Guidance per emotion
# ─────────────────────────────────────────────
WELLNESS_GUIDANCE = {
    "stress": [
        "🧘 **5 Deep Breaths**: Inhale 4 counts, hold 4, exhale 6. Activates relaxation response.",
        "📋 **Priority Matrix**: Write down everything. Handle only top 2 today.",
        "🚶 **15-min Walk**: Reduces cortisol. Nature exposure amplifies the effect.",
        "📖 **Gita Wisdom**: Pain and pleasure are temporary. This stress too shall pass.",
    ],
    "anxiety": [
        "🌱 **5-4-3-2-1**: Name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
        "📝 **Write It Out**: Journal your worries for 10 minutes. Externalize to neutralize.",
        "🎯 **Focus on Control**: Make two columns - 'In My Control' vs 'Not In My Control'.",
        "📖 **Gita Wisdom**: Fear arises from over-identification with temporary circumstances.",
    ],
    "motivation": [
        "🔥 **2-Minute Rule**: Commit to just 2 minutes. Starting is the hardest part.",
        "🎯 **Reconnect with Your Why**: Write why you started. Read it daily.",
        "🏆 **Celebrate Small Wins**: One task completed? That counts!",
        "📖 **Gita Wisdom**: Arise! Do not yield to weakness. You carry strength within you.",
    ],
    "confusion": [
        "🗺️ **Write 'What I Know'**: Create solid ground from what you do know.",
        "🧭 **Values Compass**: What are your top 3 values? Clarity on values reveals path.",
        "🗣️ **Seek One Mentor**: One 30-min conversation can provide months of clarity.",
        "📖 **Gita Wisdom**: Seek a wise teacher with humility. Their wisdom is available to you.",
    ],
    "happiness": [
        "🙏 **Practice Gratitude**: Write 3 specific things you're grateful for today.",
        "🌊 **Share Your Joy**: Happiness compounds when shared with others.",
        "🌿 **Cultivate Inner Joy**: True happiness comes from within, not external conditions.",
        "📖 **Gita Wisdom**: The highest happiness is found in a peaceful, self-aware mind.",
    ],
    "general": [
        "🌅 **Morning Ritual**: 5 min silence + 5 min movement + 5 min intention-setting.",
        "🍽️ **Eat for Your Mind**: Fresh, nutritious, pure food promotes mental clarity.",
        "🧘 **Daily Meditation**: 10 minutes daily rewires your stress response.",
        "📖 **Gita Wisdom**: Your mind is your best friend or worst enemy. Make it your ally.",
    ],
}


def get_wellness_guidance(emotion: str) -> list:
    """Return wellness tips for the detected emotion."""
    tips = WELLNESS_GUIDANCE.get(emotion, WELLNESS_GUIDANCE["general"])
    indices = np.random.choice(len(tips), min(3, len(tips)), replace=False)
    return [tips[i] for i in indices]


# ─────────────────────────────────────────────
# TF-IDF Verse Retrieval (UPDATED for new dataset)
# ─────────────────────────────────────────────
class VerseRetriever:
    def __init__(self, verses_df: pd.DataFrame):
        self.verses_df = verses_df.copy()
        
        # Check which columns exist and use them
        text_columns = []
        for col in ['explanation', 'hindi', 'english', 'meaning', 'sanskrit']:
            if col in self.verses_df.columns:
                text_columns.append(col)
        
        # Create combined text from available columns
        if text_columns:
            self.verses_df["combined_text"] = ""
            for col in text_columns:
                self.verses_df["combined_text"] += self.verses_df[col].fillna("") + " "
        else:
            # Fallback: use any string column
            str_cols = self.verses_df.select_dtypes(include=['object']).columns
            if len(str_cols) > 0:
                self.verses_df["combined_text"] = self.verses_df[str_cols[0]].fillna("")
            else:
                self.verses_df["combined_text"] = ""

        self.verses_df["processed"] = self.verses_df["combined_text"].apply(preprocess_text)

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.verses_df["processed"])
        print(f"✅ VerseRetriever ready: {len(verses_df)} verses, {len(text_columns)} text columns")

    def retrieve(self, query: str, emotion: str, top_k: int = 3):
        """Retrieve top-k most relevant verses for the query."""
        processed_query = preprocess_text(query)
        if not processed_query:
            processed_query = emotion
        
        query_vec = self.tfidf.transform([processed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Boost scores for matching emotion
        if "emotion" in self.verses_df.columns:
            emotion_mask = self.verses_df["emotion"] == emotion
            similarities[emotion_mask] *= 1.3

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.verses_df.iloc[idx]
            result = {
                "source": row.get("source", "Bhagavad Gita"),
                "chapter": int(row["chapter"]) if pd.notna(row.get("chapter")) else None,
                "verse": int(row["verse"]) if pd.notna(row.get("verse")) else None,
                "sanskrit": row.get("sanskrit", ""),
                "hindi": row.get("hindi", ""),
                "english": row.get("english", ""),
                "explanation": row.get("explanation", row.get("meaning", "")),
                "emotion": row.get("emotion", "general"),
                "relevance_score": round(float(similarities[idx]), 4),
                "relevance_pct": f"{min(99, int(float(similarities[idx]) * 100 + 40))}%"
            }
            results.append(result)

        return results