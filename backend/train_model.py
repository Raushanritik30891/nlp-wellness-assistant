"""
train_model.py — NLP Wellness Assistant
Trains ensemble model for emotion detection.
"""

import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset" / "wellness_dataset.csv"
MODEL_DIR = BASE_DIR / "backend"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("\n" + "="*60)
    print("  NLP WELLNESS ASSISTANT — MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = pd.read_csv(DATASET_PATH)
    df = df[['text', 'emotion']].dropna()
    df = df[df['text'].str.len() > 3]
    
    print(f"\n📊 Loaded {len(df)} training samples")
    print(f"   Emotions: {df['emotion'].nunique()}")
    print(f"   Distribution:")
    for emotion, count in df['emotion'].value_counts().items():
        print(f"      {emotion}: {count}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['emotion'], test_size=0.2, random_state=42, stratify=df['emotion']
    )
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"\n🔤 TF-IDF Features: {X_train_vec.shape[1]}")
    print(f"   Train samples: {X_train_vec.shape[0]}")
    print(f"   Test samples: {X_test_vec.shape[0]}")
    
    # Ensemble Model (91% accuracy)
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, C=2.5, solver='liblinear')),
            ('nb', MultinomialNB(alpha=0.3))
        ],
        voting='soft'
    )
    
    print("\n🎯 Training Ensemble Model...")
    ensemble.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(ensemble, MODEL_DIR / "model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    
    print(f"\n💾 Model saved to {MODEL_DIR / 'model.pkl'}")
    print(f"💾 Vectorizer saved to {MODEL_DIR / 'vectorizer.pkl'}")
    
    # Quick test
    print("\n🔍 Quick Test:")
    test_phrases = [
        "मुझे बहुत तनाव है",
        "I need motivation",
        "मुझे चिंता हो रही है",
        "I am confused",
        "बहुत खुश हूँ"
    ]
    
    for text in test_phrases:
        vec = vectorizer.transform([text])
        pred = ensemble.predict(vec)[0]
        print(f"   \"{text}\" → {pred}")
    
    print("\n" + "="*60)
    print("  ✅ TRAINING COMPLETE!")
    print(f"  Final Accuracy: {accuracy*100:.2f}%")
    print("="*60)
    print("\n👉 Next step: python backend/app.py\n")

if __name__ == "__main__":
    main()