import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fast Fashion Intelligence Engine",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #1E1E1E; border-radius: 12px;
        padding: 1.2rem 1.5rem; border: 1px solid #333;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4ECDC4; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 0.2rem; }
    .answer-box {
        background: #1a2332; border-left: 4px solid #4ECDC4;
        border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1rem 0;
        font-size: 0.95rem; line-height: 1.7;
    }
    .source-box {
        background: #1E1E1E; border-radius: 8px;
        padding: 0.8rem 1rem; margin: 0.5rem 0;
        border: 1px solid #333; font-size: 0.85rem;
    }
    .tag-positive { background:#1a3a2a; color:#4ade80; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
    .tag-negative { background:#3a1a1a; color:#f87171; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
    .tag-neutral  { background:#2a2a1a; color:#facc15; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('youtube_comments_topics.csv', on_bad_lines='skip')
    except FileNotFoundError:
        df = pd.read_csv('youtube_comments_english.csv', on_bad_lines='skip')
    df = df.dropna(subset=['text_clean']).reset_index(drop=True)
    df['updated_at'] = pd.to_datetime(df['updated_at'], utc=True, errors='coerce')
    df['month'] = df['updated_at'].dt.to_period('M').astype(str)
    df['year']  = df['updated_at'].dt.year
    return df

df = load_data()

# ── Load RAG pipeline ─────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    try:
        from chromadb import EmbeddingFunction
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from rank_bm25 import BM25Okapi
        import chromadb
        from groq import Groq

        class TFIDFEmbeddingFunction(EmbeddingFunction):
            def __init__(self, fit_docs, n_components=128):
                self.tfidf = TfidfVectorizer(max_features=8000, stop_words='english',
                                             ngram_range=(1,2), min_df=2)
                self.svd   = TruncatedSVD(n_components=n_components, random_state=42)
                X = self.tfidf.fit_transform(fit_docs)
                self.svd.fit(X)
            def __call__(self, input):
                X = self.tfidf.transform(input)
                return self.svd.transform(X).tolist()

        all_docs    = df['text_clean'].tolist()
        embedding_fn = TFIDFEmbeddingFunction(all_docs)

        # Build in-memory ChromaDB
        client     = chromadb.Client()
        collection = client.get_or_create_collection('yt_comments', embedding_function=embedding_fn)

        if collection.count() == 0:
            batch_size = 500
            for start in range(0, len(df), batch_size):
                batch = df.iloc[start:start+batch_size]
                collection.add(
                    documents=batch['text_clean'].tolist(),
                    ids=[str(i) for i in batch.index],
                    metadatas=[{
                        'sentiment_label': str(row.get('sentiment_label','')),
                        'video_id'       : str(row.get('video_id','')),
                        'like_count'     : int(row.get('like_count', 0)),
                        'topic_general'  : int(row.get('topic_general', -1)) if 'topic_general' in df.columns else -1,
                    } for _, row in batch.iterrows()]
                )

        # BM25
        tokenized  = [str(t).lower().split() for t in df['text_clean'].tolist()]
        bm25_index = BM25Okapi(tokenized)
        bm25_texts = df['text_clean'].tolist()

        groq_key    = st.secrets.get("GROQ_API_KEY", "")
        groq_client = Groq(api_key=groq_key) if groq_key else None

        return collection, bm25_index, bm25_texts, groq_client, embedding_fn
    except Exception as e:
        return None, None, None, None, None

# ── RAG query function ────────────────────────────────────────────────────────
def run_rag(query, collection, bm25_index, bm25_texts, groq_client, n=8):
    q = query.lower()

    sentiment_filter = None
    if any(w in q for w in ['negative','bad','angry','hate','complain']):
        sentiment_filter = 'Negative'
    elif any(w in q for w in ['positive','good','love','great','praise']):
        sentiment_filter = 'Positive'

    task   = 'summarize' if any(w in q for w in ['summarize','summary','overview']) else 'qa'
    brands = ['shein','temu','zara','h&m','primark','asos','boohoo']

    if any(b in q for b in brands) and len(q.split()) <= 5:
        strategy = 'lexical'
    elif any(w in q for w in ['feel','think','concern','impact','environment']):
        strategy = 'semantic'
    else:
        strategy = 'hybrid'

    if strategy == 'lexical':
        scores   = bm25_index.get_scores(q.split())
        top_idxs = np.argsort(scores)[::-1][:n]
        docs     = [{'text': bm25_texts[i], 'sentiment_label': '', 'like_count': 0}
                    for i in top_idxs if scores[i] > 0]
    elif strategy == 'semantic':
        kwargs = {'query_texts': [query], 'n_results': n}
        if sentiment_filter:
            kwargs['where'] = {'sentiment_label': sentiment_filter}
        res  = collection.query(**kwargs)
        docs = [{'text': d, **m} for d, m in zip(res['documents'][0], res['metadatas'][0])]
    else:
        kwargs = {'query_texts': [query], 'n_results': n*2}
        if sentiment_filter:
            kwargs['where'] = {'sentiment_label': sentiment_filter}
        sem      = collection.query(**kwargs)
        sem_docs = [{'text': d, **m} for d, m in zip(sem['documents'][0], sem['metadatas'][0])]
        lex_scrs = bm25_index.get_scores(q.split())
        lex_top  = np.argsort(lex_scrs)[::-1][:n*2]
        lex_docs = [{'text': bm25_texts[i], 'sentiment_label': '', 'like_count': 0} for i in lex_top]
        seen, docs = set(), []
        for doc in sem_docs + lex_docs:
            key = doc['text'][:80]
            if key not in seen:
                seen.add(key)
                docs.append(doc)
            if len(docs) >= n:
                break

    context = ''.join([
        f"[Comment {i} | {d.get('sentiment_label','?')}]: {d['text']}\n\n"
        for i, d in enumerate(docs[:8], 1)
    ])

    if not groq_client:
        return "Groq API key not configured. Add GROQ_API_KEY to Streamlit secrets.", docs[:3], strategy

    if task == 'summarize':
        prompt = f"Summarize these YouTube comments about fast fashion in 5 sentences:\n\n{context}\nSUMMARY:"
    else:
        prompt = (f"Answer based ONLY on these comments. Be concise (3-5 sentences).\n\n"
                  f"Question: {query}\n\nComments:\n{context}\nAnswer:")

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.3
    )
    return resp.choices[0].message.content.strip(), docs[:3], strategy

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👗 Fast Fashion Engine")
    st.markdown("*YouTube Comment Intelligence*")
    st.divider()

    sentiment_filter = st.multiselect(
        "Filter by Sentiment",
        ['Positive', 'Negative', 'Neutral'],
        default=['Positive', 'Negative', 'Neutral']
    )

    video_options = ['All Videos'] + sorted(df['video_id'].unique().tolist())
    selected_video = st.selectbox("Filter by Video", video_options)

    st.divider()
    st.markdown(f"**Total comments:** {len(df):,}")
    st.markdown(f"**Videos:** {df['video_id'].nunique()}")
    st.markdown(f"**Date range:** {df['updated_at'].min().strftime('%b %Y')} – {df['updated_at'].max().strftime('%b %Y')}")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[df['sentiment_label'].isin(sentiment_filter)]
if selected_video != 'All Videos':
    filtered = filtered[filtered['video_id'] == selected_video]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎬 By Video",
    "🔍 Topic Explorer",
    "💬 Ask a Question",
    "📈 Trends Over Time"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="main-header">Fast Fashion Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyzing YouTube public opinion on fast fashion brands</div>', unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(filtered):,}</div><div class="metric-label">Total Comments</div></div>', unsafe_allow_html=True)
    with col2:
        neg_pct = round((filtered['sentiment_label']=='Negative').mean()*100, 1)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{neg_pct}%</div><div class="metric-label">Negative Sentiment</div></div>', unsafe_allow_html=True)
    with col3:
        pos_pct = round((filtered['sentiment_label']=='Positive').mean()*100, 1)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{pos_pct}%</div><div class="metric-label">Positive Sentiment</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{filtered["video_id"].nunique()}</div><div class="metric-label">Videos Analyzed</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Sentiment Distribution")
        sent_counts = filtered['sentiment_label'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        colors = {'Positive': '#4ade80', 'Negative': '#f87171', 'Neutral': '#facc15'}
        fig = px.pie(sent_counts, values='Count', names='Sentiment',
                     color='Sentiment', color_discrete_map=colors,
                     hole=0.45)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', showlegend=True, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top Keywords — Word Cloud")
        if 'keywords' in filtered.columns:
            all_kws = []
            for kw_list in filtered['keywords'].dropna():
                try:
                    kws = ast.literal_eval(kw_list) if isinstance(kw_list, str) else kw_list
                    all_kws.extend(kws)
                except:
                    all_kws.extend(str(kw_list).split(','))
            if all_kws:
                kw_freq = Counter(all_kws)
                wc = WordCloud(width=600, height=300, background_color='#0e1117',
                               colormap='cool', max_words=60).generate_from_frequencies(kw_freq)
                fig_wc, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                fig_wc.patch.set_facecolor('#0e1117')
                st.pyplot(fig_wc)
        else:
            st.info("Run Step 3 (keyword extraction) and reload to see word cloud.")

    # Most liked comments
    st.subheader("Most Liked Comments")
    top_liked = filtered.nlargest(5, 'like_count')[['text_clean','sentiment_label','like_count','video_id']]
    for _, row in top_liked.iterrows():
        tag_class = f"tag-{row['sentiment_label'].lower()}"
        st.markdown(f"""
        <div class="source-box">
            <span class="{tag_class}">{row['sentiment_label']}</span>
            <span style="color:#888; font-size:0.8rem; margin-left:8px">👍 {row['like_count']:,} likes</span>
            <p style="margin-top:0.5rem; margin-bottom:0">{row['text_clean'][:300]}</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BY VIDEO
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Sentiment by Video")
    st.markdown("Compare how sentiment differs across the 24 videos in the dataset.")

    # Sentiment breakdown per video
    video_sent = df.groupby(['video_id','sentiment_label']).size().unstack(fill_value=0).reset_index()
    for col in ['Positive','Negative','Neutral']:
        if col not in video_sent.columns:
            video_sent[col] = 0
    video_sent['total'] = video_sent[['Positive','Negative','Neutral']].sum(axis=1)
    video_sent['neg_pct'] = (video_sent['Negative'] / video_sent['total'] * 100).round(1)
    video_sent = video_sent.sort_values('total', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Positive', x=video_sent['video_id'], y=video_sent['Positive'],
                         marker_color='#4ade80'))
    fig.add_trace(go.Bar(name='Negative', x=video_sent['video_id'], y=video_sent['Negative'],
                         marker_color='#f87171'))
    fig.add_trace(go.Bar(name='Neutral',  x=video_sent['video_id'], y=video_sent['Neutral'],
                         marker_color='#facc15'))
    fig.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                      xaxis_title='Video ID', yaxis_title='Comment Count',
                      height=420, legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # Video detail
    st.subheader("Video Deep Dive")
    sel_vid = st.selectbox("Select a video to inspect", df['video_id'].unique())
    vid_df  = df[df['video_id'] == sel_vid]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Comments", f"{len(vid_df):,}")
    c2.metric("Negative %", f"{(vid_df['sentiment_label']=='Negative').mean()*100:.1f}%")
    c3.metric("Avg Likes", f"{vid_df['like_count'].mean():.1f}")

    st.markdown("**Sample comments from this video:**")
    for _, row in vid_df.sample(min(5, len(vid_df)), random_state=42).iterrows():
        tag_class = f"tag-{row['sentiment_label'].lower()}"
        st.markdown(f"""
        <div class="source-box">
            <span class="{tag_class}">{row['sentiment_label']}</span>
            <p style="margin-top:0.4rem; margin-bottom:0">{row['text_clean'][:250]}</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOPIC EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Topic Explorer")
    st.markdown("Browse the themes discovered by BERTopic across all comments.")

    if 'topic_general' in df.columns and 'topic_general_label' in df.columns:
        topic_counts = df[df['topic_general'] != -1].groupby(
            ['topic_general','topic_general_label']).size().reset_index(name='count')
        topic_counts = topic_counts.sort_values('count', ascending=False)

        # Topic size bar chart
        fig = px.bar(topic_counts, x='topic_general_label', y='count',
                     color='count', color_continuous_scale='teal',
                     labels={'topic_general_label':'Topic','count':'Comments'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', height=380,
                          xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Topic drill-down
        st.subheader("Explore a Topic")
        topic_labels = topic_counts['topic_general_label'].tolist()
        sel_topic_label = st.selectbox("Select topic", topic_labels)
        sel_topic_id    = topic_counts[topic_counts['topic_general_label']==sel_topic_label]['topic_general'].values[0]
        topic_df        = df[df['topic_general'] == sel_topic_id]

        c1, c2, c3 = st.columns(3)
        c1.metric("Comments in topic", f"{len(topic_df):,}")
        c2.metric("Negative %", f"{(topic_df['sentiment_label']=='Negative').mean()*100:.1f}%")
        c3.metric("Positive %", f"{(topic_df['sentiment_label']=='Positive').mean()*100:.1f}%")

        st.markdown("**Sentiment split for this topic:**")
        topic_sent = topic_df['sentiment_label'].value_counts().reset_index()
        topic_sent.columns = ['Sentiment','Count']
        fig2 = px.pie(topic_sent, values='Count', names='Sentiment',
                      color='Sentiment', color_discrete_map={'Positive':'#4ade80','Negative':'#f87171','Neutral':'#facc15'},
                      hole=0.4)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=260)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Sample comments from this topic:**")
        for _, row in topic_df.sample(min(5, len(topic_df)), random_state=1).iterrows():
            tag_class = f"tag-{row['sentiment_label'].lower()}"
            st.markdown(f"""
            <div class="source-box">
                <span class="{tag_class}">{row['sentiment_label']}</span>
                <p style="margin-top:0.4rem; margin-bottom:0">{row['text_clean'][:280]}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Topic columns not found. Run Step 4 (BERTopic) and reload with youtube_comments_topics.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ASK A QUESTION (RAG)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Ask a Question")
    st.markdown("Ask anything about fast fashion opinions and get an answer grounded in real YouTube comments.")

    collection, bm25_index, bm25_texts, groq_client, embedding_fn = load_rag()

    example_queries = [
        "What do people say about Shein's labor practices?",
        "Summarize the negative comments about environmental damage",
        "How do people justify buying from Temu?",
        "What are the main concerns about fast fashion?",
        "What do positive commenters appreciate about Shein?",
    ]

    st.markdown("**Try an example:**")
    cols = st.columns(len(example_queries))
    selected_example = None
    for i, (col, ex) in enumerate(zip(cols, example_queries)):
        if col.button(f"💬 {ex[:30]}...", key=f"ex_{i}"):
            selected_example = ex

    query = st.text_input(
        "Or type your own question:",
        value=selected_example or "",
        placeholder="e.g. What do people think about fast fashion workers?"
    )

    if st.button("🔍 Search & Answer", type="primary") and query:
        if collection is None:
            st.error("RAG pipeline failed to load. Check that all required libraries are installed.")
        else:
            with st.spinner("Retrieving relevant comments and generating answer..."):
                try:
                    answer, sources, strategy = run_rag(
                        query, collection, bm25_index, bm25_texts, groq_client
                    )
                    st.markdown(f"**Retrieval strategy used:** `{strategy}`")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    st.markdown("**Top source comments used:**")
                    for src in sources:
                        tag_class = f"tag-{src.get('sentiment_label','neutral').lower()}"
                        st.markdown(f"""
                        <div class="source-box">
                            <span class="{tag_class}">{src.get('sentiment_label','?')}</span>
                            <p style="margin-top:0.4rem; margin-bottom:0">{src.get('text','')[:280]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRENDS OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Trends Over Time")
    st.markdown("How has the sentiment and volume of fast fashion discussion changed over time?")

    time_df = df.dropna(subset=['month'])

    # Comment volume over time
    st.subheader("Comment Volume by Month")
    vol = time_df.groupby('month').size().reset_index(name='count')
    vol = vol.sort_values('month')
    fig = px.area(vol, x='month', y='count', color_discrete_sequence=['#4ECDC4'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='white', height=320,
                      xaxis_title='Month', yaxis_title='Comments')
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment trend over time
    st.subheader("Sentiment Trend Over Time")
    sent_time = time_df.groupby(['month','sentiment_label']).size().unstack(fill_value=0).reset_index()
    sent_time = sent_time.sort_values('month')

    fig2 = go.Figure()
    colors = {'Positive':'#4ade80','Negative':'#f87171','Neutral':'#facc15'}
    for sentiment in ['Positive','Negative','Neutral']:
        if sentiment in sent_time.columns:
            fig2.add_trace(go.Scatter(
                x=sent_time['month'], y=sent_time[sentiment],
                name=sentiment, mode='lines',
                line=dict(color=colors[sentiment], width=2),
                fill='tozeroy', fillcolor=colors[sentiment].replace(')',',0.1)').replace('rgb','rgba')
            ))
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font_color='white', height=360,
                       xaxis_title='Month', yaxis_title='Comments',
                       legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

    # Yearly summary
    st.subheader("Yearly Sentiment Summary")
    yearly = time_df.groupby(['year','sentiment_label']).size().unstack(fill_value=0).reset_index()
    yearly = yearly[yearly['year'] >= 2018]
    fig3 = px.bar(yearly, x='year', y=['Positive','Negative','Neutral'] if 'Neutral' in yearly.columns else ['Positive','Negative'],
                  barmode='group',
                  color_discrete_map={'Positive':'#4ade80','Negative':'#f87171','Neutral':'#facc15'})
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font_color='white', height=340,
                       xaxis_title='Year', yaxis_title='Comments')
    st.plotly_chart(fig3, use_container_width=True)

