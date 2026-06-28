
import numpy as np
import chromadb
from chromadb import EmbeddingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from groq import Groq

# Must be defined here so pickle can load embedding_function.pkl correctly
class TFIDFEmbeddingFunction(EmbeddingFunction):
    def __init__(self, fit_docs, n_components=128):
        self.tfidf = TfidfVectorizer(max_features=8000, stop_words="english",
                                     ngram_range=(1, 2), min_df=2)
        self.svd   = TruncatedSVD(n_components=n_components, random_state=42)
        X = self.tfidf.fit_transform(fit_docs)
        self.svd.fit(X)
    def __call__(self, input):
        X = self.tfidf.transform(input)
        return self.svd.transform(X).tolist()

def load_rag_pipeline(groq_api_key, db_path="./chroma_db",
                      embedding_pkl="embedding_function.pkl",
                      bm25_pkl="bm25_index.pkl"):
    with open(embedding_pkl, "rb") as f:
        embedding_fn = pickle.load(f)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("youtube_comments", embedding_function=embedding_fn)
    with open(bm25_pkl, "rb") as f:
        bm25_data = pickle.load(f)
    groq_client = Groq(api_key=groq_api_key)
    return collection, bm25_data, groq_client

def rag_query(query, collection, bm25_data, groq_client, n_results=8):
    bm25_index   = bm25_data["index"]
    bm25_texts   = bm25_data["texts"]
    bm25_indices = bm25_data["indices"]
    q = query.lower()

    sentiment_filter = None
    if any(w in q for w in ["negative","complain","hate","bad","angry"]):
        sentiment_filter = "Negative"
    elif any(w in q for w in ["positive","love","great","good","praise"]):
        sentiment_filter = "Positive"

    task = "summarize" if any(w in q for w in ["summarize","summary","overview"]) else "qa"

    brands = ["shein","temu","zara","h&m","primark","asos","boohoo"]
    if any(b in q for b in brands) and len(q.split()) <= 5:
        strategy = "lexical"
    elif any(w in q for w in ["feel","think","concern","impact","environment"]):
        strategy = "semantic"
    else:
        strategy = "hybrid"

    if strategy == "lexical":
        scores   = bm25_index.get_scores(q.split())
        top_idxs = np.argsort(scores)[::-1][:n_results]
        docs     = [{"text": bm25_texts[i], "sentiment_label": "", "like_count": 0}
                    for i in top_idxs if scores[i] > 0]
    elif strategy == "semantic":
        kwargs = {"query_texts": [query], "n_results": n_results}
        if sentiment_filter:
            kwargs["where"] = {"sentiment_label": sentiment_filter}
        res  = collection.query(**kwargs)
        docs = [{"text": d, **m} for d, m in zip(res["documents"][0], res["metadatas"][0])]
    else:
        kwargs = {"query_texts": [query], "n_results": n_results * 2}
        if sentiment_filter:
            kwargs["where"] = {"sentiment_label": sentiment_filter}
        sem      = collection.query(**kwargs)
        sem_docs = [{"text": d, **m} for d, m in zip(sem["documents"][0], sem["metadatas"][0])]
        lex_scrs = bm25_index.get_scores(q.split())
        lex_top  = np.argsort(lex_scrs)[::-1][:n_results * 2]
        lex_docs = [{"text": bm25_texts[i], "sentiment_label": "", "like_count": 0} for i in lex_top]
        seen, docs = set(), []
        for doc in sem_docs + lex_docs:
            key = doc["text"][:80]
            if key not in seen:
                seen.add(key)
                docs.append(doc)
            if len(docs) >= n_results:
                break

    context = "".join([
        f"[Comment {i} | {d.get('sentiment_label','?')}]: {d['text']}\n\n"
        for i, d in enumerate(docs[:8], 1)
    ])

    if task == "summarize":
        prompt = f"Summarize these YouTube comments about fast fashion in 5 sentences:\n\n{context}\nSUMMARY:"
    else:
        prompt = (
            f"Answer based ONLY on these comments. Be concise (3-5 sentences).\n\n"
            f"Question: {query}\n\nComments:\n{context}\nAnswer:"
        )

    resp = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3
    )
    return {
        "answer"  : resp.choices[0].message.content.strip(),
        "sources" : docs[:3],
        "strategy": strategy,
        "task"    : task
    }
