"""
Dhurandhar 2 — Review Intelligence Studio
An end-to-end NLP dashboard (Sentiment, Translation, Q&A, Summarization)
built on Hugging Face Transformers, wrapped in a polished Streamlit UI.
"""
 
import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
 
# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Dhurandhar 2 | Review Intelligence Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# --------------------------------------------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
 
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
 
    .stApp {
        background: radial-gradient(circle at 10% 0%, #1a0a12 0%, #0b0710 45%, #050308 100%);
        color: #f2eef5;
    }
 
    #MainMenu, footer, header {visibility: hidden;}
 
    .hero {
        padding: 2.4rem 2.6rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #3b0a2e 0%, #6b0f2a 45%, #b5182c 100%);
        box-shadow: 0 20px 60px rgba(181, 24, 44, 0.35);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after{
        content:"";
        position:absolute; top:-40%; right:-10%;
        width:420px; height:420px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.05rem;
        color: #f4d9df;
        max-width: 720px;
        margin: 0;
    }
    .badge-row { margin-top: 1.1rem; display:flex; gap:0.6rem; flex-wrap:wrap; }
    .badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        backdrop-filter: blur(6px);
    }
 
    .metric-card {
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        text-align: left;
        transition: all 0.25s ease;
    }
    .metric-card:hover { border-color: rgba(255,255,255,0.28); transform: translateY(-3px); }
    .metric-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #d79bb0; font-weight: 600;}
    .metric-value { font-family:'Poppins', sans-serif; font-size: 2rem; font-weight: 700; margin-top: 2px;}
    .metric-sub { font-size: 0.78rem; color: #b8adbf; margin-top: 2px;}
 
    .review-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid #b5182c;
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .review-card.pos { border-left-color: #34c98e; }
    .review-card.neg { border-left-color: #ff5470; }
    .rc-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;}
    .rc-tag { font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:999px; letter-spacing:0.5px;}
    .rc-tag.pos { background: rgba(52,201,142,0.15); color:#34c98e; border:1px solid rgba(52,201,142,0.4);}
    .rc-tag.neg { background: rgba(255,84,112,0.15); color:#ff5470; border:1px solid rgba(255,84,112,0.4);}
    .rc-conf { font-size:0.75rem; color:#b8adbf; }
    .rc-text { font-size:0.88rem; color:#e6dfe9; line-height:1.5; }
    .match-yes { color:#34c98e; font-weight:700;}
    .match-no { color:#ff5470; font-weight:700;}
 
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12060d 0%, #0b0710 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
 
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: #cfc3d3;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #b5182c, #6b0f2a);
        color: white !important;
    }
 
    .stButton>button {
        background: linear-gradient(120deg, #b5182c, #8a1224);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        box-shadow: 0 8px 20px rgba(181,24,44,0.3);
    }
    .stButton>button:hover { filter: brightness(1.12); }
 
    .section-title {
        font-family:'Poppins', sans-serif;
        font-weight:700;
        font-size:1.35rem;
        margin: 0.4rem 0 1rem 0;
        color: #f2eef5;
    }
    .footer-note { color:#7d7284; font-size:0.78rem; text-align:center; margin-top:2.5rem; }
</style>
""", unsafe_allow_html=True)
 
# --------------------------------------------------------------------------------------
# DATA LOADING (bundled dataset — no upload needed)
# --------------------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "netflix_movie_dhurandhar_2.csv")
 
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH, delimiter=";")
    df.columns = [c.strip() for c in df.columns]
    return df
 
df = load_data()
reviews = df["Review"].tolist()
real_labels = df["Class"].tolist()
 
# --------------------------------------------------------------------------------------
# MODEL LOADING (cached)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
 
@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    from transformers import pipeline
    return pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
 
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    from transformers import pipeline
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
 
@st.cache_resource(show_spinner=False)
def load_summarization_pipeline():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
 
# --------------------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 Dhurandhar 2 — Review Intelligence Studio</h1>
    <p>An end-to-end NLP pipeline powered by Hugging Face Transformers — sentiment classification,
    machine translation, question answering, and summarization applied to real audience reviews.</p>
    <div class="badge-row">
        <span class="badge">🤗 Transformers</span>
        <span class="badge">DistilBERT · SST-2</span>
        <span class="badge">Helsinki-NLP · EN→ES</span>
        <span class="badge">SQuAD Q&A</span>
        <span class="badge">DistilBART Summarizer</span>
    </div>
</div>
""", unsafe_allow_html=True)
 
# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Studio Controls")
    st.caption("Dataset is preloaded — sit back and explore.")
    st.markdown("---")
    st.markdown(f"**Total reviews:** {len(df)}")
    pos_count = real_labels.count("POSITIVE")
    neg_count = real_labels.count("NEGATIVE")
    st.markdown(f"**Positive:** {pos_count}  |  **Negative:** {neg_count}")
    st.markdown("---")
    st.markdown("### 📚 Pipeline Stack")
    st.markdown("""
    - **Sentiment:** `distilbert-base-uncased-finetuned-sst-2-english`
    - **Translation:** `Helsinki-NLP/opus-mt-en-es`
    - **Q&A:** `distilbert-base-cased-distilled-squad`
    - **Summarization:** `sshleifer/distilbart-cnn-12-6`
    """)
    st.markdown("---")
    st.caption("Built with 🩷 using Streamlit + 🤗 Transformers")
 
# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "😊 Sentiment Analysis", "🌍 Translation", "❓ Question Answering", "📝 Summarization"
])
 
# ========================================================================================
# TAB 1 — OVERVIEW
# ========================================================================================
with tab1:
    st.markdown('<div class="section-title">Dataset at a Glance</div>', unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    avg_len = int(df["Review"].str.split().apply(len).mean())
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Reviews</div>
        <div class="metric-value">{len(df)}</div><div class="metric-sub">Curated critic & audience reviews</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Positive Reviews</div>
        <div class="metric-value" style="color:#34c98e;">{pos_count}</div><div class="metric-sub">{pos_count/len(df)*100:.0f}% of dataset</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Negative Reviews</div>
        <div class="metric-value" style="color:#ff5470;">{neg_count}</div><div class="metric-sub">{neg_count/len(df)*100:.0f}% of dataset</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Review Length</div>
        <div class="metric-value">{avg_len}</div><div class="metric-sub">words per review</div></div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.4])
 
    with left:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Sentiment Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=["Positive", "Negative"],
            values=[pos_count, neg_count],
            hole=0.62,
            marker=dict(colors=["#34c98e", "#ff5470"]),
            textinfo="label+percent",
            textfont=dict(color="white", size=13),
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300,
            font=dict(color="#f2eef5"),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with right:
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Review Length Distribution</div>', unsafe_allow_html=True)
        lengths = df["Review"].str.split().apply(len)
        fig2 = px.histogram(x=lengths, nbins=10, color_discrete_sequence=["#b5182c"])
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), height=300,
            xaxis_title="Words per review", yaxis_title="Count",
            font=dict(color="#f2eef5"),
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown('<div class="section-title">Browse Reviews</div>', unsafe_allow_html=True)
    for i, (rev, lab) in enumerate(zip(reviews, real_labels)):
        cls = "pos" if lab == "POSITIVE" else "neg"
        snippet = rev if len(rev) < 380 else rev[:380].rsplit(" ", 1)[0] + " …"
        st.markdown(f"""
        <div class="review-card {cls}">
            <div class="rc-head">
                <span class="rc-tag {cls}">{lab}</span>
                <span class="rc-conf">Review #{i+1}</span>
            </div>
            <div class="rc-text">{snippet}</div>
        </div>
        """, unsafe_allow_html=True)
 
# ========================================================================================
# TAB 2 — SENTIMENT ANALYSIS
# ========================================================================================
with tab2:
    st.markdown('<div class="section-title">Batch Sentiment Evaluation</div>', unsafe_allow_html=True)
    st.caption("Runs DistilBERT (SST-2 fine-tuned) across the full review set and benchmarks it against ground-truth labels.")
 
    run_batch = st.button("▶ Run Sentiment Analysis on Dataset", key="run_batch")
 
    if run_batch or "batch_results" in st.session_state:
        if run_batch:
            with st.spinner("Loading model & classifying reviews..."):
                classifier = load_sentiment_pipeline()
                predicted_labels = classifier(reviews)
                st.session_state["batch_results"] = predicted_labels
        predicted_labels = st.session_state["batch_results"]
 
        import evaluate
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        references = [1 if label == "POSITIVE" else 0 for label in real_labels]
        predictions = [1 if p["label"] == "POSITIVE" else 0 for p in predicted_labels]
        acc = accuracy.compute(references=references, predictions=predictions)["accuracy"]
        f1_score = f1.compute(references=references, predictions=predictions)["f1"]
 
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy</div>
            <div class="metric-value">{acc*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">F1 Score</div>
            <div class="metric-value">{f1_score:.3f}</div></div>""", unsafe_allow_html=True)
        with m3:
            correct = sum(1 for r, p in zip(references, predictions) if r == p)
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Correct Predictions</div>
            <div class="metric-value">{correct}/{len(references)}</div></div>""", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.1rem;">Prediction Detail</div>', unsafe_allow_html=True)
        for i, (rev, pred, actual) in enumerate(zip(reviews, predicted_labels, real_labels)):
            cls = "pos" if pred["label"] == "POSITIVE" else "neg"
            match = pred["label"] == actual
            snippet = rev if len(rev) < 260 else rev[:260].rsplit(" ", 1)[0] + " …"
            st.markdown(f"""
            <div class="review-card {cls}">
                <div class="rc-head">
                    <span class="rc-tag {cls}">Predicted: {pred['label']}</span>
                    <span class="rc-conf">Confidence: {pred['score']:.2%} &nbsp;|&nbsp; Actual: {actual} &nbsp;
                    <span class="{'match-yes' if match else 'match-no'}">{'✓ match' if match else '✗ mismatch'}</span></span>
                </div>
                <div class="rc-text">{snippet}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Click the button above to classify all reviews and compute accuracy / F1 score.")
 
    st.markdown("---")
    st.markdown('<div class="section-title" style="font-size:1.15rem;">Try Your Own Review</div>', unsafe_allow_html=True)
    custom_text = st.text_area("Enter a movie review:", "KGF 2 is an amazing movie with powerful action and excellent performance.", height=100)
    if st.button("Analyze Sentiment", key="custom_sentiment"):
        with st.spinner("Analyzing..."):
            classifier = load_sentiment_pipeline()
            result = classifier(custom_text)[0]
        cls = "pos" if result["label"] == "POSITIVE" else "neg"
        st.markdown(f"""
        <div class="review-card {cls}">
            <div class="rc-head">
                <span class="rc-tag {cls}">{result['label']}</span>
                <span class="rc-conf">Confidence: {result['score']:.2%}</span>
            </div>
            <div class="rc-text">{custom_text}</div>
        </div>
        """, unsafe_allow_html=True)
 
# ========================================================================================
# TAB 3 — TRANSLATION
# ========================================================================================
with tab3:
    st.markdown('<div class="section-title">English → Spanish Translation</div>', unsafe_allow_html=True)
    st.caption("Powered by Helsinki-NLP's MarianMT (opus-mt-en-es).")
 
    review_idx = st.selectbox(
        "Choose a review to translate, or write your own below:",
        options=list(range(len(reviews))),
        format_func=lambda i: f"Review #{i+1} ({real_labels[i]}) — {reviews[i][:70]}...",
    )
    default_text = reviews[review_idx]
    text_to_translate = st.text_area("Text to translate:", default_text, height=150)
 
    if st.button("🌍 Translate to Spanish"):
        with st.spinner("Translating..."):
            translator = load_translation_pipeline()
            chunk = text_to_translate[:1000]
            translated = translator(chunk)[0]["translation_text"]
        col_en, col_es = st.columns(2)
        with col_en:
            st.markdown("**🇬🇧 Original (English)**")
            st.markdown(f'<div class="review-card">{chunk}</div>', unsafe_allow_html=True)
        with col_es:
            st.markdown("**🇪🇸 Translated (Spanish)**")
            st.markdown(f'<div class="review-card pos">{translated}</div>', unsafe_allow_html=True)
 
# ========================================================================================
# TAB 4 — QUESTION ANSWERING
# ========================================================================================
with tab4:
    st.markdown('<div class="section-title">Ask Questions About a Review</div>', unsafe_allow_html=True)
    st.caption("Extractive Q&A using DistilBERT fine-tuned on SQuAD — the model pulls the answer directly from the review text.")
 
    qa_idx = st.selectbox(
        "Select context review:",
        options=list(range(len(reviews))),
        format_func=lambda i: f"Review #{i+1} ({real_labels[i]}) — {reviews[i][:70]}...",
        key="qa_select",
    )
    context = reviews[qa_idx]
    st.markdown(f'<div class="review-card">{context}</div>', unsafe_allow_html=True)
 
    sample_questions = [
        "Who directed the movie?",
        "Who plays the lead role?",
        "What is the movie about?",
        "Custom question...",
    ]
    q_choice = st.selectbox("Pick a sample question or write your own:", sample_questions)
    question = st.text_input("Your question:", "" if q_choice == "Custom question..." else q_choice)
 
    if st.button("❓ Get Answer") and question.strip():
        with st.spinner("Finding the answer..."):
            qa_pipe = load_qa_pipeline()
            result = qa_pipe(question=question, context=context)
        st.markdown(f"""
        <div class="review-card pos">
            <div class="rc-head">
                <span class="rc-tag pos">Answer</span>
                <span class="rc-conf">Confidence: {result['score']:.2%}</span>
            </div>
            <div class="rc-text" style="font-size:1.05rem; font-weight:600;">{result['answer']}</div>
        </div>
        """, unsafe_allow_html=True)
 
# ========================================================================================
# TAB 5 — SUMMARIZATION
# ========================================================================================
with tab5:
    st.markdown('<div class="section-title">Review Summarization</div>', unsafe_allow_html=True)
    st.caption("Condenses long reviews into concise summaries using DistilBART (CNN/DailyMail fine-tuned).")
 
    sum_idx = st.selectbox(
        "Select a review to summarize:",
        options=list(range(len(reviews))),
        format_func=lambda i: f"Review #{i+1} ({real_labels[i]}) — {reviews[i][:70]}...",
        key="sum_select",
    )
    full_review = reviews[sum_idx]
    st.markdown(f'<div class="review-card">{full_review}</div>', unsafe_allow_html=True)
 
    max_len = st.slider("Max summary length (tokens):", 30, 150, 80)
    min_len = st.slider("Min summary length (tokens):", 10, 60, 25)
 
    if st.button("📝 Summarize"):
        with st.spinner("Generating summary..."):
            summarizer = load_summarization_pipeline()
            summary = summarizer(full_review, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        orig_words = len(full_review.split())
        summ_words = len(summary.split())
        st.markdown(f"""
        <div class="review-card pos">
            <div class="rc-head">
                <span class="rc-tag pos">Summary</span>
                <span class="rc-conf">{orig_words} words → {summ_words} words ({(1 - summ_words/orig_words)*100:.0f}% shorter)</span>
            </div>
            <div class="rc-text">{summary}</div>
        </div>
        """, unsafe_allow_html=True)
 
# --------------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------------
st.markdown('<div class="footer-note">Dhurandhar 2 Review Intelligence Studio · Streamlit + Hugging Face Transformers</div>', unsafe_allow_html=True)
 



