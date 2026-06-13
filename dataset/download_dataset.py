"""
download_dataset.py — NLP Wellness Assistant
Downloads Bhagavad Gita dataset, prepares verses and training data.
"""

import os
import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

print("\n" + "="*60)
print("  NLP WELLNESS ASSISTANT — DATASET PREPARATION")
print("="*60)

# ─────────────────────────────────────────────────────────────
# Load Bhagavad Gita verses from HuggingFace
# ─────────────────────────────────────────────────────────────
def download_gita_verses():
    try:
        from datasets import load_dataset
        print("\n📥 Downloading Bhagavad Gita from HuggingFace...")
        ds = load_dataset('OEvortex/Bhagavad_Gita', split='train')
        df = ds.to_pandas()
        
        # Rename columns
        df = df.rename(columns={
            'Sanskrit Anuvad': 'sanskrit',
            'Hindi Anuvad': 'hindi',
            'Enlgish Translation': 'english',
            'Chapter': 'chapter',
            'Verse': 'verse'
        })
        
        # Clean chapter and verse
        df['chapter'] = df['chapter'].str.extract('(\d+)').astype(int)
        df['verse'] = df['verse'].str.extract('(\d+\.?\d*)').astype(str)
        
        # Assign emotions based on keywords
        def assign_emotion(text):
            text = str(text).lower()
            if any(w in text for w in ['stress', 'work', 'duty', 'action', 'karma', 'perform']):
                return 'stress'
            if any(w in text for w in ['fear', 'worry', 'anxiety', 'doubt', 'afraid', 'scared']):
                return 'anxiety'
            if any(w in text for w in ['arise', 'strength', 'victory', 'courage', 'stand', 'fight', 'glory']):
                return 'motivation'
            if any(w in text for w in ['confused', 'path', 'direction', 'lost', 'confusion', 'bewildered']):
                return 'confusion'
            if any(w in text for w in ['joy', 'bliss', 'happy', 'happiness', 'peace', 'pleasure']):
                return 'happiness'
            return 'general'
        
        df['emotion'] = df['english'].apply(assign_emotion)
        df['source'] = 'Bhagavad Gita'
        df['explanation'] = df['english']
        
        # Select final columns
        final_df = df[['chapter', 'verse', 'sanskrit', 'hindi', 'english', 'explanation', 'emotion', 'source']]
        final_df.to_csv(DATASET_DIR / 'verses_db.csv', index=False)
        
        print(f"   ✅ Downloaded {len(final_df)} verses")
        print(f"   📊 Emotion distribution:")
        for emotion, count in final_df['emotion'].value_counts().items():
            print(f"      {emotion}: {count}")
        
        return final_df
    except Exception as e:
        print(f"   ⚠️ Could not download from HuggingFace: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# Training phrases for each emotion
# ─────────────────────────────────────────────────────────────
TRAINING_PHRASES = {
    "stress": [
        "I am feeling stressed", "work pressure is too much", "मुझे तनाव है",
        "too many responsibilities", "overwhelmed with work", "तनाव से परेशान हूँ",
        "can't handle this", "burned out", "काम का दबाव बहुत है",
        "stressed out", "feeling overwhelmed", "tension ho rahi hai",
        "deadline pressure", "too many tasks", "feeling exhausted"
    ],
    "anxiety": [
        "I feel anxious", "worried about future", "मुझे चिंता हो रही है",
        "fear is taking over", "can't stop worrying", "भविष्य की चिंता है",
        "overthinking", "panic", "डर लग रहा है",
        "nervous", "restless", "anxiety ho rahi hai",
        "I am overthinking everything", "feeling nervous", "heart is racing"
    ],
    "motivation": [
        "I need motivation", "feeling like giving up", "प्रेरणा चाहिए",
        "lost my drive", "can't find energy", "हिम्मत हार रहा हूँ",
        "want to succeed", "need inspiration", "उठना है आगे बढ़ना है",
        "demotivated", "no will power", "give me strength",
        "need a push", "stay motivated", "keep going"
    ],
    "confusion": [
        "confused", "don't know what to do", "समझ नहीं आ रहा",
        "feeling lost", "can't decide", "सही रास्ता क्या है",
        "uncertain", "need clarity", "निर्णय नहीं ले पा रहा",
        "directionless", "foggy mind", "which path to choose",
        "decision paralysis", "unsure about future"
    ],
    "happiness": [
        "feeling happy", "life is wonderful", "बहुत खुश हूँ",
        "joyful", "grateful", "आज बहुत अच्छा दिन है",
        "great mood", "blessed", "खुशी से भर गया हूँ",
        "content", "peaceful", "मन प्रसन्न है"
    ],
    "general": [
        "how to live well", "seeking wisdom", "जीवन कैसे जियूँ",
        "guide me", "need advice", "आध्यात्मिक मार्गदर्शन चाहिए",
        "meaning of life", "inner peace", "मानसिक शांति कैसे पाऊँ",
        "spiritual guidance", "life advice", "wellness tips"
    ],
}

# ─────────────────────────────────────────────────────────────
# Build training dataset
# ─────────────────────────────────────────────────────────────
def build_training_dataset(verses_df):
    training_data = []
    
    # Add training phrases
    for emotion, phrases in TRAINING_PHRASES.items():
        for phrase in phrases:
            training_data.append({'text': phrase, 'emotion': emotion})
            # Add variations
            training_data.append({'text': phrase + '!', 'emotion': emotion})
            training_data.append({'text': 'very ' + phrase.lower(), 'emotion': emotion})
    
    # Add verses as training examples
    for _, verse in verses_df.iterrows():
        if len(str(verse.get('hindi', ''))) > 20:
            training_data.append({'text': verse['hindi'][:200], 'emotion': verse['emotion']})
        if len(str(verse.get('english', ''))) > 30:
            training_data.append({'text': verse['english'][:200], 'emotion': verse['emotion']})
        if len(str(verse.get('explanation', ''))) > 30:
            training_data.append({'text': verse['explanation'][:200], 'emotion': verse['emotion']})
    
    # Create dataframe
    train_df = pd.DataFrame(training_data)
    train_df = train_df.drop_duplicates(subset=['text'])
    train_df = train_df[train_df['text'].str.len() > 3]
    
    train_df.to_csv(DATASET_DIR / 'wellness_dataset.csv', index=False)
    
    print(f"\n📊 Training dataset created: {len(train_df)} samples")
    print(f"   Distribution:")
    for emotion in ['stress', 'anxiety', 'motivation', 'confusion', 'happiness', 'general']:
        count = len(train_df[train_df['emotion'] == emotion])
        print(f"      {emotion}: {count}")
    
    return train_df

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # Download verses
    verses_df = download_gita_verses()
    
    if verses_df is None:
        print("\n❌ Failed to download dataset. Using fallback...")
        return
    
    # Build training dataset
    train_df = build_training_dataset(verses_df)
    
    print("\n" + "="*60)
    print("  ✅ DATASET PREPARATION COMPLETE!")
    print(f"  Verses: {len(verses_df)}")
    print(f"  Training samples: {len(train_df)}")
    print("="*60)
    print("\n👉 Next step: python train_model.py\n")

if __name__ == "__main__":
    main()