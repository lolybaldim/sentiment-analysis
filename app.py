"""
CineMood: Movie Review Sentiment Studio
A Streamlit dashboard for the IMDB sentiment model (TF-IDF + Logistic Regression).
"""
import json
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────── Page setup ───────────────────────────
st.set_page_config(page_title="CineMood · Sentiment Studio", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

GOLD, POS, NEG, MUTED = "#F5C451", "#3DDC97", "#FF5C7A", "#8B93A7"
CARD, BG = "#161B26", "#0D1017"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, input, textarea {{ font-family: 'Inter', sans-serif; }}
.stApp {{
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(245,196,81,.10), transparent 60%),
    radial-gradient(900px 400px at -10% 110%, rgba(61,220,151,.06), transparent 60%),
    {BG};
}}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }}
#MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: #0A0D13; border-right: 1px solid #1E2431; }}
section[data-testid="stSidebar"] .stRadio label p {{ font-size: 15px; }}

/* Hero */
.hero {{
  position: relative; overflow: hidden;
  border: 1px solid #262D3D; border-radius: 22px; padding: 30px 34px; margin-bottom: 22px;
  background: linear-gradient(135deg, #1A2030 0%, #12161F 60%);
}}
.hero:before {{   /* film-strip edge */
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 14px;
  background: repeating-linear-gradient(#0D1017 0 8px, {GOLD} 8px 16px); opacity: .55;
}}
.hero h1 {{ font-family: 'Fraunces', serif; font-weight: 800; font-size: 40px; margin: 0;
           color: #F4F1EA; letter-spacing: -.5px; }}
.hero h1 span {{ color: {GOLD}; }}
.hero p {{ color: {MUTED}; font-size: 15.5px; margin: 8px 0 0; max-width: 720px; }}
.tag {{ display: inline-block; font-size: 11.5px; font-weight: 600; letter-spacing: 1.4px; text-transform: uppercase;
        color: {GOLD}; border: 1px solid rgba(245,196,81,.35); border-radius: 999px; padding: 4px 11px; margin-bottom: 12px; }}

/* KPI cards */
.kpi {{ background: {CARD}; border: 1px solid #232A39; border-radius: 16px; padding: 18px 20px; height: 100%; }}
.kpi .label {{ color: {MUTED}; font-size: 12.5px; font-weight: 600; letter-spacing: .8px; text-transform: uppercase; }}
.kpi .value {{ font-family: 'Fraunces', serif; font-size: 32px; font-weight: 800; color: #F4F1EA; margin-top: 4px; }}
.kpi .sub {{ color: {MUTED}; font-size: 13px; margin-top: 2px; }}

/* Verdict */
.verdict {{ border-radius: 18px; padding: 22px 24px; border: 1px solid; }}
.verdict.pos {{ background: rgba(61,220,151,.08); border-color: rgba(61,220,151,.4); }}
.verdict.neg {{ background: rgba(255,92,122,.08); border-color: rgba(255,92,122,.4); }}
.verdict .big {{ font-family: 'Fraunces', serif; font-size: 34px; font-weight: 800; }}
.verdict.pos .big {{ color: {POS}; }}  .verdict.neg .big {{ color: {NEG}; }}
.verdict .small {{ color: #C9CEDA; font-size: 14.5px; margin-top: 4px; }}

/* Highlighted review */
.hl {{ background: {CARD}; border: 1px solid #232A39; border-radius: 16px; padding: 18px 20px;
       line-height: 2.05; font-size: 15.5px; color: #D7DBE5; }}
.hl .w {{ padding: 2px 5px; border-radius: 6px; }}

.section-title {{ font-family: 'Fraunces', serif; font-size: 22px; font-weight: 700; color: #F4F1EA; margin: 8px 0 6px; }}
.caption {{ color: {MUTED}; font-size: 13.5px; margin-bottom: 10px; }}

/* Inputs & buttons */
textarea {{ background: {CARD} !important; border-radius: 14px !important; font-size: 15px !important; }}
.stButton > button[kind="primary"], .stDownloadButton > button {{
  background: {GOLD}; color: #14161C; border: none; font-weight: 700; border-radius: 12px; padding: .55rem 1.4rem;
}}
.stButton > button[kind="primary"]:hover {{ background: #FFD772; color: #14161C; }}
div[data-testid="stTabs"] button p {{ font-size: 15px; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Model & helpers ───────────────────────────
def clean_text(text: str) -> str:
    """Same cleaning as in the notebook."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    vocab = tfidf.get_feature_names_out()
    weights = pd.Series(clf.coef_[0], index=vocab)
    return model, tfidf, weights


@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return None


model, tfidf, WEIGHTS = load_model()
METRICS = load_metrics()


def explain(cleaned: str, top_n: int = 10) -> pd.DataFrame:
    """How much each word/phrase in the review pushed the prediction (TF-IDF value × weight)."""
    vec = tfidf.transform([cleaned])
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return pd.DataFrame(columns=["term", "impact"])
    terms = tfidf.get_feature_names_out()[idx]
    impact = vec.toarray()[0, idx] * WEIGHTS.values[idx]
    df = pd.DataFrame({"term": terms, "impact": impact})
    df = df.reindex(df["impact"].abs().sort_values(ascending=False).index).head(top_n)
    return df.sort_values("impact")


def highlight(raw: str) -> str:
    """Colour each word of the review by how positive/negative the model thinks it is."""
    scale = max(WEIGHTS.abs().quantile(0.995), 1e-6)
    out = []
    for word in raw.split():
        key = clean_text(word)
        w = WEIGHTS.get(key, 0.0) if key else 0.0
        if abs(w) < scale * 0.08:
            out.append(word)
            continue
        alpha = min(abs(w) / scale, 1) * 0.75 + 0.15
        rgb = "61,220,151" if w > 0 else "255,92,122"
        out.append(f"<span class='w' style='background:rgba({rgb},{alpha:.2f})'>{word}</span>")
    return " ".join(out)


def gauge(pos_prob: float) -> go.Figure:
    color = POS if pos_prob >= 0.5 else NEG
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pos_prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": "#F4F1EA", "family": "Fraunces"}},
        title={"text": "Positivity score", "font": {"size": 14, "color": MUTED}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
               "bar": {"color": color, "thickness": 0.28},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 50], "color": "rgba(255,92,122,.12)"},
                         {"range": [50, 100], "color": "rgba(61,220,151,.12)"}],
               "threshold": {"line": {"color": GOLD, "width": 3}, "thickness": 0.8, "value": 50}}))
    return style(fig, 260)


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color="#C9CEDA", size=13),
                      legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(gridcolor="#222838", zerolinecolor="#2D3446")
    fig.update_yaxes(gridcolor="#222838", zerolinecolor="#2D3446")
    return fig


def impact_chart(df: pd.DataFrame, height: int = 340) -> go.Figure:
    colors = [POS if v > 0 else NEG for v in df["impact"]]
    fig = go.Figure(go.Bar(x=df["impact"], y=df["term"], orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="← pushes negative · pushes positive →")
    return style(fig, height)


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='label'>{label}</div>"
                 f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>",
                 unsafe_allow_html=True)


def predict_many(texts: pd.Series) -> np.ndarray:
    return model.predict_proba(texts.astype(str).apply(clean_text))[:, 1]


SAMPLE_REVIEWS = [
    "An absolute masterpiece. The acting was brilliant and the story kept me hooked until the very end.",
    "Boring, predictable and way too long. I want my two hours back.",
    "I thought it would be bad, but it was not bad at all. Actually quite enjoyable!",
    "The visuals were stunning but the plot made no sense whatsoever.",
    "One of the worst films I have ever seen. Terrible script and awful acting.",
    "A beautiful, heartfelt story. I cried and laughed. Highly recommend it to everyone.",
    "Mediocre at best. A few funny moments, but mostly forgettable.",
    "The soundtrack is incredible and the lead actress gives a career-best performance.",
    "Waste of time. The dialogue was cringe and the ending was lazy.",
    "Smart, funny and surprisingly emotional. A perfect weekend movie.",
    "I fell asleep halfway through. Nothing happens for an hour.",
    "Great cast, great direction, great everything. Instant classic.",
    "The first half was promising but the second half completely fell apart.",
    "Loved the characters and the world-building. Can't wait for the sequel!",
    "Poorly edited, confusing and dull. Avoid.",
]

if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-family:Fraunces,serif;font-size:28px;font-weight:800;color:#F4F1EA'>"
                f"🎬 Cine<span style='color:{GOLD}'>Mood</span></div>"
                f"<div style='color:{MUTED};font-size:13px;margin-bottom:18px'>Sentiment Studio</div>",
                unsafe_allow_html=True)
    page = st.radio("Navigate", ["🔍  Analyze a review", "📊  Batch dashboard", "🧠  Model insights", "ℹ️  About"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"<div style='color:{MUTED};font-size:12.5px;line-height:1.7'>"
                f"<b style='color:#C9CEDA'>Model</b><br>TF-IDF (1–2 grams) + Logistic Regression<br><br>"
                f"<b style='color:#C9CEDA'>Trained on</b><br>IMDB · 50,000 movie reviews<br><br>"
                f"<b style='color:#C9CEDA'>Vocabulary</b><br>{len(WEIGHTS):,} words & phrases"
                + (f"<br><br><b style='color:#C9CEDA'>Test accuracy</b><br>{METRICS['accuracy']:.1%}" if METRICS else "")
                + "</div>", unsafe_allow_html=True)


def hero(tag, title_html, text):
    st.markdown(f"<div class='hero'><div class='tag'>{tag}</div><h1>{title_html}</h1><p>{text}</p></div>",
                unsafe_allow_html=True)


# ─────────────────────────── Page 1: Analyze ───────────────────────────
if page.endswith("Analyze a review"):
    hero("Live analysis", "What's the <span>mood</span> of this review?",
         "Paste any movie review. The model reads it, scores how positive it is, and shows you exactly "
         "which words made it decide.")

    examples = {"✍️  Write my own": "", "🌟  Glowing": SAMPLE_REVIEWS[0], "💀  Harsh": SAMPLE_REVIEWS[1],
                "🔄  Tricky negation": SAMPLE_REVIEWS[2], "⚖️  Mixed feelings": SAMPLE_REVIEWS[3]}
    pick = st.pills("Try an example", list(examples), default="✍️  Write my own")
    text = st.text_area("Your review", value=examples.get(pick or "✍️  Write my own", ""), height=140,
                        placeholder="e.g. The plot was clever but the ending felt rushed…",
                        label_visibility="collapsed")
    go_btn = st.button("Analyze sentiment  →", type="primary")

    if go_btn and not text.strip():
        st.warning("Type or pick a review first.")
    elif go_btn:
        cleaned = clean_text(text)
        p = float(model.predict_proba([cleaned])[0][1])
        label = "Positive" if p >= 0.5 else "Negative"
        conf = p if p >= 0.5 else 1 - p
        strength = "Strongly" if conf >= 0.85 else "Fairly" if conf >= 0.65 else "Slightly"
        st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"),
                                            "review": text[:90] + ("…" if len(text) > 90 else ""),
                                            "sentiment": label, "confidence": round(conf, 3)})

        st.markdown("")
        c1, c2 = st.columns([1, 1.15], gap="large")
        with c1:
            css = "pos" if label == "Positive" else "neg"
            emoji = "😀" if label == "Positive" else "😞"
            st.markdown(f"<div class='verdict {css}'><div class='small'>The model says</div>"
                        f"<div class='big'>{emoji} {label}</div>"
                        f"<div class='small'>{strength} {label.lower()} · {conf:.0%} confident</div></div>",
                        unsafe_allow_html=True)
            st.plotly_chart(gauge(p), width="stretch", config={"displayModeBar": False})
        with c2:
            st.markdown("<div class='section-title'>Why it decided this</div>"
                        "<div class='caption'>Words and phrases with the biggest effect on this prediction</div>",
                        unsafe_allow_html=True)
            exp = explain(cleaned)
            if exp.empty:
                st.info("None of these words are in the model's vocabulary.")
            else:
                st.plotly_chart(impact_chart(exp, 330), width="stretch", config={"displayModeBar": False})

        st.markdown("<div class='section-title'>Your review, colour-coded</div>"
                    f"<div class='caption'><span style='color:{POS}'>■</span> positive words &nbsp; "
                    f"<span style='color:{NEG}'>■</span> negative words &nbsp; (darker = stronger)</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='hl'>{highlight(text)}</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("<br><div class='section-title'>Session history</div>", unsafe_allow_html=True)
        hist = pd.DataFrame(st.session_state.history)
        h1, h2, h3 = st.columns(3)
        kpi(h1, "Reviews analyzed", len(hist))
        kpi(h2, "Positive", f"{(hist.sentiment == 'Positive').mean():.0%}")
        kpi(h3, "Avg confidence", f"{hist.confidence.mean():.0%}")
        st.markdown("")
        st.dataframe(hist, width="stretch", hide_index=True,
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", min_value=0, max_value=1, format="percent")})

# ─────────────────────────── Page 2: Batch dashboard ───────────────────────────
elif page.endswith("Batch dashboard"):
    hero("Bulk analysis", "Analyze <span>hundreds</span> of reviews at once",
         "Upload a CSV with a column of reviews and get a full sentiment dashboard in seconds. "
         "No file? Load the sample set to see how it works.")

    u1, u2 = st.columns([3, 1], vertical_alignment="bottom")
    file = u1.file_uploader("Upload a CSV file", type="csv")
    if u2.button("Load sample reviews", width="stretch"):
        st.session_state.batch = pd.DataFrame({"review": SAMPLE_REVIEWS})
    if file is not None:
        st.session_state.batch = pd.read_csv(file)

    data = st.session_state.get("batch")
    if data is None:
        st.info("⬆️ Upload a CSV or click **Load sample reviews** to build the dashboard.")
    else:
        text_cols = [c for c in data.columns if data[c].dtype == object]
        if not text_cols:
            st.error("This CSV has no text columns.")
            st.stop()
        default = text_cols.index("review") if "review" in text_cols else 0
        col = st.selectbox("Which column contains the reviews?", text_cols, index=default)

        df = data.dropna(subset=[col]).copy()
        with st.spinner(f"Analyzing {len(df):,} reviews…"):
            df["positive_prob"] = predict_many(df[col])
        df["predicted"] = np.where(df["positive_prob"] >= 0.5, "Positive", "Negative")
        df["confidence"] = np.where(df["positive_prob"] >= 0.5, df["positive_prob"], 1 - df["positive_prob"])
        df["words"] = df[col].astype(str).str.split().str.len()

        n, pos_share = len(df), (df.predicted == "Positive").mean()
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Total reviews", f"{n:,}")
        kpi(k2, "Positive", f"{pos_share:.0%}", f"{(df.predicted == 'Positive').sum():,} reviews")
        kpi(k3, "Negative", f"{1 - pos_share:.0%}", f"{(df.predicted == 'Negative').sum():,} reviews")
        kpi(k4, "Avg confidence", f"{df.confidence.mean():.0%}",
            f"{(df.confidence < 0.65).sum():,} uncertain")

        # If the CSV has true labels (like the IMDB file), show accuracy too
        label_col = next((c for c in df.columns if c.lower() in ("sentiment", "label")), None)
        if label_col is not None:
            truth = df[label_col].astype(str).str.lower().map(
                {"positive": "Positive", "negative": "Negative", "1": "Positive", "0": "Negative"})
            if truth.notna().any():
                acc = (truth == df.predicted)[truth.notna()].mean()
                st.success(f"✅ This file has true labels in **{label_col}**: the model got "
                           f"**{acc:.1%}** of them right.")

        st.markdown("")
        g1, g2 = st.columns([1, 1.4], gap="large")
        with g1:
            st.markdown("<div class='section-title'>Sentiment split</div>", unsafe_allow_html=True)
            counts = df.predicted.value_counts().reindex(["Positive", "Negative"]).fillna(0)
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.68,
                                   marker=dict(colors=[POS, NEG], line=dict(color=BG, width=3)),
                                   textinfo="percent", sort=False))
            fig.add_annotation(text=f"<b>{n:,}</b><br>reviews", showarrow=False,
                               font=dict(size=18, color="#F4F1EA"))
            st.plotly_chart(style(fig, 320), width="stretch", config={"displayModeBar": False})
        with g2:
            st.markdown("<div class='section-title'>How sure is the model?</div>", unsafe_allow_html=True)
            fig = px.histogram(df, x="positive_prob", color="predicted", nbins=25,
                               color_discrete_map={"Positive": POS, "Negative": NEG})
            fig.update_layout(xaxis_title="Positivity score (0 = very negative, 1 = very positive)",
                              yaxis_title="Reviews", bargap=0.06, legend_title=None)
            fig.add_vline(x=0.5, line_dash="dash", line_color=GOLD)
            st.plotly_chart(style(fig, 320), width="stretch", config={"displayModeBar": False})

        g3, g4 = st.columns(2, gap="large")
        with g3:
            st.markdown("<div class='section-title'>Words driving the mood</div>"
                        "<div class='caption'>Total impact of each term across all reviews</div>",
                        unsafe_allow_html=True)
            X = tfidf.transform(df[col].astype(str).apply(clean_text))
            total = pd.Series(np.asarray(X.multiply(WEIGHTS.values).sum(axis=0)).ravel(),
                              index=WEIGHTS.index)
            top = pd.concat([total.nsmallest(8), total.nlargest(8)]).drop_duplicates()
            top = top[top != 0].sort_values()
            st.plotly_chart(impact_chart(pd.DataFrame({"term": top.index, "impact": top.values}), 380),
                            width="stretch", config={"displayModeBar": False})
        with g4:
            st.markdown("<div class='section-title'>Review length vs sentiment</div>"
                        "<div class='caption'>Do longer reviews lean one way?</div>", unsafe_allow_html=True)
            fig = px.box(df, x="predicted", y="words", color="predicted", points="outliers",
                         color_discrete_map={"Positive": POS, "Negative": NEG})
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Words per review")
            st.plotly_chart(style(fig, 380), width="stretch", config={"displayModeBar": False})

        st.markdown("<div class='section-title'>All results</div>", unsafe_allow_html=True)
        f1, f2 = st.columns([1, 2])
        show = f1.segmented_control("Show", ["All", "Positive", "Negative", "Uncertain"], default="All")
        query = f2.text_input("Search reviews", placeholder="Search for a word…", label_visibility="collapsed")
        view = df
        if show in ("Positive", "Negative"):
            view = view[view.predicted == show]
        elif show == "Uncertain":
            view = view[view.confidence < 0.65]
        if query:
            view = view[view[col].astype(str).str.contains(query, case=False, na=False)]
        st.dataframe(view[[col, "predicted", "confidence"]], width="stretch", hide_index=True, height=360,
                     column_config={col: st.column_config.TextColumn("Review", width="large"),
                                    "confidence": st.column_config.ProgressColumn(
                                        "Confidence", min_value=0, max_value=1, format="percent")})
        st.download_button("⬇  Download results (CSV)", df.drop(columns=["words"]).to_csv(index=False),
                           "cinemood_results.csv", "text/csv")

# ─────────────────────────── Page 3: Model insights ───────────────────────────
elif page.endswith("Model insights"):
    hero("Under the hood", "What did the model <span>learn</span>?",
         "Every word and phrase gets a weight during training. Positive weights push towards a good review, "
         "negative weights towards a bad one. Here's what the model picked up from 50,000 reviews.")

    if METRICS:
        m1, m2, m3, m4 = st.columns(4)
        kpi(m1, "Accuracy", f"{METRICS['accuracy']:.1%}", "on unseen test reviews")
        kpi(m2, "Precision", f"{METRICS['precision']:.1%}", "when it says positive, it's right")
        kpi(m3, "Recall", f"{METRICS['recall']:.1%}", "positive reviews it catches")
        kpi(m4, "F1-score", f"{METRICS['f1']:.1%}", "balance of both")
        st.markdown("")
    else:
        st.info("💡 Tip: run the **save metrics** cell in the notebook to show accuracy and the confusion matrix here.")

    i1, i2 = st.columns(2, gap="large")
    for col_, title, series, color in [
        (i1, "Most positive words", WEIGHTS.nlargest(15).sort_values(), POS),
        (i2, "Most negative words", WEIGHTS.nsmallest(15).sort_values(ascending=False), NEG)]:
        with col_:
            st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=series.values, y=series.index, orientation="h", marker_color=color,
                                   hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
            fig.update_layout(xaxis_title="Weight")
            st.plotly_chart(style(fig, 460), width="stretch", config={"displayModeBar": False})

    j1, j2 = st.columns([1, 1.2], gap="large")
    with j1:
        st.markdown("<div class='section-title'>Look up any word</div>"
                    "<div class='caption'>See how the model feels about a word or 2-word phrase</div>",
                    unsafe_allow_html=True)
        word = st.text_input("Word", value="not good", label_visibility="collapsed")
        key = clean_text(word)
        if key in WEIGHTS.index:
            w = WEIGHTS[key]
            rank = (WEIGHTS.abs() > abs(w)).sum() + 1
            tone = "positive" if w > 0 else "negative"
            colr = POS if w > 0 else NEG
            st.markdown(f"<div class='kpi'><div class='label'>“{key}”</div>"
                        f"<div class='value' style='color:{colr}'>{w:+.2f}</div>"
                        f"<div class='sub'>Leans {tone} · #{rank:,} most influential of {len(WEIGHTS):,}</div></div>",
                        unsafe_allow_html=True)
        elif key:
            st.warning(f"“{key}” isn't in the vocabulary. It may be too rare, or it's a stop word.")
    with j2:
        if METRICS and "confusion_matrix" in METRICS:
            st.markdown("<div class='section-title'>Confusion matrix</div>"
                        "<div class='caption'>Test-set predictions vs the real answers</div>",
                        unsafe_allow_html=True)
            cm = np.array(METRICS["confusion_matrix"])
            fig = px.imshow(cm, text_auto=",", x=["Pred. negative", "Pred. positive"],
                            y=["Actually negative", "Actually positive"],
                            color_continuous_scale=[[0, "#1A2030"], [1, GOLD]])
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(textfont_size=18)
            st.plotly_chart(style(fig, 320), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown("<div class='section-title'>Weight distribution</div>"
                        "<div class='caption'>Most words are neutral (near 0); a few carry strong feeling</div>",
                        unsafe_allow_html=True)
            fig = px.histogram(WEIGHTS.clip(-6, 6), nbins=80, color_discrete_sequence=[GOLD])
            fig.update_layout(showlegend=False, xaxis_title="Weight", yaxis_title="Words")
            st.plotly_chart(style(fig, 320), width="stretch", config={"displayModeBar": False})

# ─────────────────────────── Page 4: About ───────────────────────────
else:
    hero("About the project", "How <span>CineMood</span> works",
         "A sentiment analysis project built with Python, scikit-learn and Streamlit.")
    steps = [("01", "Data", "50,000 IMDB movie reviews from Kaggle, half positive and half negative."),
             ("02", "Cleaning", "Lowercase, remove HTML tags, links, punctuation and numbers. Keep negations like “not”."),
             ("03", "TF-IDF", "Turn each review into numbers: words common in one review but rare overall score high."),
             ("04", "Model", "Logistic Regression learns a weight for every word and 2-word phrase."),
             ("05", "Evaluation", "Tested on 10,000 reviews the model never saw during training."),
             ("06", "Deployment", "This Streamlit app loads the saved model and predicts in real time.")]
    for row in (steps[:3], steps[3:]):
        cols = st.columns(3)
        for c, (num, title, body) in zip(cols, row):
            c.markdown(f"<div class='kpi'><div class='label' style='color:{GOLD}'>Step {num}</div>"
                       f"<div class='value' style='font-size:24px'>{title}</div>"
                       f"<div class='sub' style='font-size:14px;line-height:1.55'>{body}</div></div>",
                       unsafe_allow_html=True)
        st.markdown("")
    st.markdown("<div class='section-title'>Limitations</div>", unsafe_allow_html=True)
    st.markdown("- **Sarcasm** (“Oh great, another two hours wasted”) can fool it.\n"
                "- **Mixed reviews** sit near 50% because they contain both kinds of words.\n"
                "- It was trained on **movie reviews**, so other topics may be less accurate.")
