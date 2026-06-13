# NLP Wellness Assistant — Interview Questions & Answers
## 20 AI/ML/NLP Interview Questions for This Project

---

### 1. What is the overall architecture of your NLP Wellness Assistant?

**Answer:**
The system follows a 5-stage pipeline:
1. **Input** — User submits free-text describing their emotional state
2. **Preprocessing** — Lowercasing, punctuation removal, stopword filtering, lemmatization
3. **Emotion Classification** — TF-IDF vectorized features fed to a trained classifier (Naive Bayes selected as best)
4. **Verse Retrieval** — TF-IDF cosine similarity search over a curated dataset of 31 Bhagavad Gita + Purana verses
5. **Response Generation** — Structured JSON response with emotion, confidence, verse, meaning, and 3 wellness tips

No LLMs, no transformers, no vector databases — pure classical NLP.

---

### 2. Why did you choose TF-IDF over Word2Vec or BERT embeddings?

**Answer:**
TF-IDF was chosen for several reasons:
- **Interpretability**: You can inspect which words drive each classification — important in an interview context
- **Efficiency**: Trains in milliseconds, no GPU needed, suitable for production on a simple server
- **Sufficient complexity**: For 6 emotion categories with ~150 labeled samples, TF-IDF captures the vocabulary signal adequately
- **Transparency**: In wellness/healthcare contexts, explainability matters — you need to justify recommendations

BERT would require 110M parameters for a dataset of ~150 samples — clear overfitting risk and unjustified complexity.

---

### 3. Walk me through the TF-IDF formula and why it's useful for text classification.

**Answer:**
TF-IDF = TF × IDF

- **TF (Term Frequency)** = count of word in document / total words in document. Rewards words that appear often in a specific text.
- **IDF (Inverse Document Frequency)** = log(N / df), where N = total docs, df = docs containing the word. Penalizes common words (like "the", "is") that appear everywhere and carry no discriminative value.

Together: a word scores high if it's **frequent in this document** but **rare across all documents** — making it distinctive and discriminative. Perfect for emotion classification where "stressed", "overwhelmed", "burnout" are highly discriminative.

We also used **sublinear_tf=True** which applies log(1 + tf) to reduce the dominance of very frequent terms.

---

### 4. Why did Naive Bayes outperform Logistic Regression on this dataset?

**Answer:**
Naive Bayes won because:
1. **Small dataset advantage**: NB works well with limited data (~124 training samples after split). It estimates class-conditional probabilities from word counts — very data-efficient.
2. **High-dimensional sparse features**: TF-IDF creates sparse vectors. NB handles sparsity naturally through its probabilistic formulation.
3. **Independence assumption**: The Naive Bayes conditional independence assumption happens to be a reasonable approximation for short emotional text where individual key words ("stress", "anxiety") are strong independent signals.
4. **Regularization-free**: Logistic Regression can overfit on sparse high-dimensional small data without careful hyperparameter tuning.

On larger datasets (1000+ samples), Logistic Regression typically surpasses NB.

---

### 5. What preprocessing steps did you apply and why?

**Answer:**
```
Raw Text → Lowercase → Remove Punctuation → Remove Numbers → 
Tokenize → Remove Stopwords → Lemmatization → Rejoin
```

- **Lowercase**: "Stressed" and "stressed" are the same token
- **Remove punctuation**: "feeling," vs "feeling" — same meaning
- **Stopwords**: "I", "am", "the", "a" carry no emotion signal
- **Lemmatization**: "stressed" → "stress", "anxious" → "anxiety" — reduces vocabulary size and improves generalization
- We used a custom lemmatizer instead of NLTK because NLTK wasn't available in the deployment environment — demonstrating adaptability

---

### 6. How does your verse retrieval work? What algorithm do you use?

**Answer:**
We use **TF-IDF Cosine Similarity** — classic Information Retrieval:

1. Build a TF-IDF matrix for all 31 verses (combining keywords + English translation + meaning)
2. Transform the user's query into the same TF-IDF space
3. Compute cosine similarity between the query vector and each verse vector
4. Apply an **emotion boost multiplier (1.3×)** to verses matching the classified emotion
5. Return top-3 verses by score

**Why cosine similarity?** It measures the angle between vectors regardless of document length — a long verse and a short verse can still score equally if they contain the same key terms.

**Why not FAISS or vector DBs?** For 31 verses, sklearn's `cosine_similarity` is O(31) — no latency difference. FAISS has value at 100,000+ documents.

---

### 7. Explain the evaluation metrics you used and what they mean.

**Answer:**
| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | Correct / Total | Overall correctness — misleading with class imbalance |
| **Precision** | TP / (TP + FP) | Of all samples labeled as emotion X, how many were actually X? |
| **Recall** | TP / (TP + FN) | Of all actual emotion X samples, how many did we correctly find? |
| **F1 Score** | 2×P×R / (P+R) | Harmonic mean of precision and recall — best single metric |

We used **weighted average** for all metrics to account for class imbalance. F1 was the primary selection criterion for choosing the best model.

---

### 8. What is the emotion boost in verse retrieval and why did you add it?

**Answer:**
After computing TF-IDF cosine similarity, verses whose `emotion` label matches the classified emotion get their score multiplied by **1.3 (30% boost)**.

**Why?** Pure keyword similarity can match a "motivation" verse to a "stress" query if they share general wellness vocabulary. The emotion boost ensures that the retrieved verse is contextually appropriate — if someone is anxious, they should receive verses about managing anxiety, not verses about happiness.

This is a **domain-specific heuristic** that improves practical utility without complicating the core retrieval algorithm.

---

### 9. How would you scale this system to handle 10,000 users per day?

**Answer:**
1. **Model serving**: Replace joblib.load at request time with a pre-loaded model in memory (already done — loaded at app startup)
2. **Caching**: Redis cache for identical/similar queries (stress queries converge to similar verses)
3. **Gunicorn/uWSGI**: Replace Flask dev server with a production WSGI server (4 workers)
4. **Containerize**: Docker image for consistent deployment
5. **CDN for plots**: Serve static visualization assets from CDN, not Flask
6. **If scaling further**: Move to FastAPI for async support and better throughput

At 10K users/day (~7 requests/minute), even a single Gunicorn server handles this trivially.

---

### 10. How did you handle the lack of a large labeled dataset?

**Answer:**
Several strategies:
1. **Curated seed data**: 25-26 training phrases per emotion, carefully written to cover linguistic variation
2. **Verse keyword augmentation**: Each verse's keywords also serve as training signal
3. **TF-IDF with bigrams**: Captures phrases like "giving up", "no motivation", "feel stressed" — more robust than unigrams alone
4. **Class-balanced training**: Ensured equal representation across all 6 emotions
5. **Cross-validation awareness**: With only ~124 training samples, we acknowledge this in evaluation

For production with budget, we would collect real user data through the chat interface and iteratively improve.

---

### 11. What are the limitations of your current approach?

**Answer (honest self-assessment — interviewers love this)**:
1. **Small dataset**: ~25 phrases per emotion may not cover all linguistic expressions (sarcasm, indirect expression)
2. **Keyword dependency**: TF-IDF fails on paraphrases — "I feel overwhelmed" and "everything is too much" may score differently despite same meaning
3. **No context memory**: Each query is classified independently — no conversation history
4. **English only**: No Hindi/multilingual support
5. **No negation handling**: "I don't feel stressed" and "I feel stressed" may both score high for stress
6. **Static verse mapping**: The 31-verse database covers common emotions but may not have ideal verses for niche inputs

---

### 12. How would you improve the model if you had more time?

**Answer:**
Short term:
- Expand training data with user feedback (active learning)
- Add negation detection preprocessing
- Use cross-validation for better metric estimation

Medium term:
- Use pre-trained sentence embeddings (sentence-transformers) for semantic similarity instead of TF-IDF
- Add multilingual support (Hindi/English)
- Integrate larger verse database (700+ Gita verses + Puranas)

Long term:
- Build a retrieval-augmented generation (RAG) layer on top of the classifier
- Fine-tune a small language model (Phi-2, Gemma-2B) for more natural responses

---

### 13. Explain the difference between Multinomial Naive Bayes and Bernoulli Naive Bayes.

**Answer:**
| | Multinomial NB | Bernoulli NB |
|--|--|--|
| **Feature type** | Word counts / TF-IDF weights | Binary (word present/absent) |
| **Best for** | Longer texts, when frequency matters | Short texts, document classification |
| **Formula** | P(word\|class) = (count + α) / (class_total + α*V) | Considers absence of words too |

We use **Multinomial NB with TF-IDF** — TF-IDF values are non-negative and interpretable as term weights, fitting the multinomial assumption. The `alpha=0.5` Laplace smoothing prevents zero-probability for unseen words.

---

### 14. What is the role of the `sublinear_tf` parameter in TF-IDF?

**Answer:**
`sublinear_tf=True` applies `tf = 1 + log(tf)` instead of raw term frequency.

**Problem it solves**: A word appearing 100 times should not be 100x more important than a word appearing once. The logarithm compresses this scale — 100 occurrences becomes ~5.6, and 1 occurrence stays at 1. This prevents high-frequency words from dominating the feature space and creates more balanced feature weights.

---

### 15. How do you handle the cold-start problem for new emotion categories?

**Answer:**
Adding a new emotion category (e.g., "grief") requires:
1. Add 20-25 labeled training phrases for "grief" to the dataset
2. Add relevant verses for "grief" to the verse database
3. Retrain the model (`python backend/train_model.py`)
4. The API automatically picks up new emotion classes from model.classes_

The system is designed for easy extensibility — no hardcoded emotion list in the training code. The verse retrieval system also automatically indexes new verses.

---

### 16. What is the significance of ngram_range=(1,2) in your TF-IDF vectorizer?

**Answer:**
`ngram_range=(1,2)` includes both **unigrams** (single words) and **bigrams** (word pairs):
- Unigram: "stress", "anxiety", "give", "up"
- Bigram: "give up", "lost motivation", "feel anxious", "no motivation"

Bigrams capture important phrases that lose meaning when split. "Give up" as a bigram is strongly discriminative for motivation loss, but "give" and "up" separately are nearly meaningless. This improves classification accuracy for phrasal emotional expressions.

---

### 17. How does your API handle errors and edge cases?

**Answer:**
The Flask API handles:
- **Missing field**: Returns `{"error": "Missing 'text' field"}` with 400 status
- **Too short input**: Returns `{"error": "Text too short"}` for inputs < 3 characters
- **CORS**: Handled via after_request headers — works with any frontend origin
- **OPTIONS preflight**: Handled explicitly for browser compatibility
- **Model not found**: Would raise FileNotFoundError at startup — fail-fast is better than silently serving stale results

Production improvements would add request rate limiting and input sanitization.

---

### 18. How would you evaluate the verse retrieval quality, not just classification?

**Answer:**
Verse retrieval evaluation requires different metrics from classification:
1. **Relevance Rate**: Human evaluation — what % of top-1 retrieved verses are contextually appropriate?
2. **NDCG (Normalized Discounted Cumulative Gain)**: Standard IR metric for ranked result quality
3. **User satisfaction survey**: In production, add a thumbs up/down on each verse response
4. **A/B testing**: Test emotion-boosted retrieval vs pure similarity — measure user engagement

Currently we lack ground truth for verse relevance. Building this dataset through user feedback is the correct production approach.

---

### 19. What are the ethical considerations in an AI wellness application?

**Answer:**
Critical considerations:
1. **Not a medical device**: The system must clearly state it provides spiritual guidance, not clinical mental health treatment
2. **Crisis detection**: Should detect keywords like "suicide", "self-harm" and redirect to professional help
3. **Data privacy**: User emotional inputs are sensitive — must not be stored without explicit consent
4. **Bias in dataset**: If training data underrepresents certain demographic expressions of emotion, the model may fail those groups
5. **Religious sensitivity**: Presenting Hindu scriptures — must be culturally respectful and not preachy
6. **Dependency risk**: Should encourage professional help, not replace it

---

### 20. Compare your approach to building this with a RAG-based LLM system.

**Answer:**
| Aspect | Our NLP Approach | RAG + LLM Approach |
|---|---|---|
| **Cost** | ₹0 inference cost | ₹X per API call |
| **Latency** | <50ms | 2-8 seconds |
| **Interpretability** | High — inspect TF-IDF weights | Low — black box |
| **Control** | Full — you own the model | Dependent on LLM provider |
| **Quality** | Good for defined categories | Better for open-ended responses |
| **Privacy** | All local | Data sent to third-party API |
| **Complexity** | Low — easy to explain in interviews | High — requires prompt engineering |

For a portfolio project targeting interviews at non-LLM-focused companies (data science, analytics, ML roles), our approach demonstrates more fundamental ML knowledge. RAG would show API integration skills but less core NLP understanding.

---

## Architecture Diagram

```
USER INPUT (free text)
        │
        ▼
┌─────────────────────┐
│   TEXT PREPROCESSING │
│  • Lowercase         │
│  • Remove punctuation│
│  • Remove stopwords  │
│  • Lemmatization     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF VECTORIZER  │
│  • Unigrams+bigrams │
│  • sublinear_tf     │
│  • 3000 max features│
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────┐  ┌──────┐
│  LR  │  │  NB  │ ← Trained models
└──────┘  └──────┘
    │         │
    └────┬────┘
         │  Model comparison → Best model selected (NB)
         ▼
┌─────────────────────┐
│  EMOTION LABEL      │
│  + CONFIDENCE SCORE │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  VERSE RETRIEVAL    │
│  TF-IDF Cosine Sim  │
│  + Emotion Boost    │
│  → Top 3 Verses     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  WELLNESS GUIDANCE  │
│  Rule-based tips    │
│  per emotion class  │
└────────┬────────────┘
         │
         ▼
   JSON RESPONSE
  {emotion, confidence,
   verse, meaning,
   guidance}
```

---

## Quick Setup Commands

```bash
# 1. Clone / navigate to project
cd nlp-wellness-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python dataset/create_dataset.py

# 4. Train models (also generates visualizations)
python backend/train_model.py

# 5. Start Flask API
python backend/app.py

# 6. Open frontend
open frontend/index.html
# or serve it:
python -m http.server 3000 --directory frontend
```
