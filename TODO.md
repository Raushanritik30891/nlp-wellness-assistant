# TODO - Kaggle token support

- [x] Confirm how Kaggle CLI in this environment authenticates (kaggle.json vs env var)
- [ ] Update dataset/download_dataset.py to support KAGGLE_API_TOKEN by auto-writing kaggle.json if token env var exists
- [x] Run: python dataset/download_dataset.py and verify it builds datasets (falls back cleanly when Kaggle API returns 403)
- [x] Verify backend/train_model.py regenerates screenshots


