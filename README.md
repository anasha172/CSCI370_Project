# Fast Fashion Intelligence Engine
## CSCI370 — YouTube Comment Analysis Dashboard

An NLP-powered dashboard analyzing 30,000+ YouTube comments about fast fashion brands (Shein, Temu, Zara, H&M).

### Features
- **Sentiment Analysis** — VADER + RoBERTa ensemble
- **Topic Modeling** — BERTopic (general + per-sentiment)
- **Named Entity Recognition** — spaCy with custom brand rules
- **RAG System** — ChromaDB + BM25 + Groq LLaMA 3
- **Interactive Dashboard** — 5-tab Streamlit app

### Setup

#### Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

#### Streamlit Cloud
1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set GROQ_API_KEY in App Settings -> Secrets

### Project Structure
```
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── .streamlit/
│   ├── config.toml                 # Theme config
│   └── secrets.toml                # API keys (local only, not on GitHub)
├── youtube_comments_topics.csv     # Final processed dataset
├── notebooks/
│   ├── step1_preprocess.ipynb
│   ├── step2_sentiment.ipynb
│   ├── step2b_language_filter.ipynb
│   ├── step3_ner_keywords.ipynb
│   ├── step3b_ner_audit.ipynb
│   ├── step4_topic_modeling.ipynb
│   ├── step5a_rag_indexing.ipynb
│   ├── step5b_rag_retrieval_llm.ipynb
│   └── step6_evaluation_mlflow.ipynb
└── rag_pipeline.py                 # RAG module imported by dashboard
```

### Dataset
- **Source:** YouTube Data API v3
- **Topics:** Fast fashion brands (Shein, Temu, Zara, H&M, etc.)
- **Size:** 40,937 raw comments → 30,609 after cleaning + language filtering
- **Date range:** 2018 – 2026

### Tech Stack
| Component | Tool |
|---|---|
| Sentiment | VADER + cardiffnlp/twitter-roberta-base-sentiment |
| NER | spaCy en_core_web_sm + EntityRuler |
| Keywords | KeyBERT |
| Topics | BERTopic |
| Vector DB | ChromaDB |
| Lexical Search | BM25 (rank_bm25) |
| LLM | Groq LLaMA 3.3 70B |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
