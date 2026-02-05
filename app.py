import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# =============================
# Config
# =============================
st.set_page_config(
    page_title="Elmosyar Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "final.csv"

NUM_COLS = [
    'coherence', 'knowledge', 'teaching',
    'management', 'responsiveness', 'behavior'
]

# =============================
# Load Data
# =============================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data(DATA_PATH)

# =============================
# Aggregations
# =============================
def professor_aggregate(df):
    grp = df.groupby("professor")

    scores = grp[NUM_COLS].mean()
    overall = scores.mean(axis=1)

    agg = pd.DataFrame({
        "professor": scores.index,
        "overall_score": overall.values,
        "mean_sentiment": grp["sentiment"].mean().values,
        "n_comments": grp.size().values,
        "grading_mode": grp["grading_status"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        "attendance_mode": grp["attendance_status"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    })

    return agg.reset_index(drop=True)

# =============================
# Sidebar Filters
# =============================
st.sidebar.title("Filters")

terms = sorted(df["term"].dropna().unique())
depts = sorted(df["dept"].dropna().unique())

sel_terms = st.sidebar.multiselect("Term", terms)
sel_depts = st.sidebar.multiselect("Department", depts)

df_f = df.copy()
if sel_terms:
    df_f = df_f[df_f["term"].isin(sel_terms)]
if sel_depts:
    df_f = df_f[df_f["dept"].isin(sel_depts)]

prof_agg = professor_aggregate(df_f)

# =============================
# Navigation
# =============================
st.title("📊 بازار داشبورد اساتید")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Search & Filter", "Professor Profile", "Compare", "Recommender"]
)

# =============================
# Overview
# =============================
if page == "Overview":
    st.header("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total comments", len(df_f))
    c2.metric("Professors", prof_agg["professor"].nunique())
    c3.metric("Avg overall score", f"{prof_agg['overall_score'].mean():.2f}")

    st.subheader("Overall score distribution")
    fig = px.histogram(prof_agg, x="overall_score", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Grading status distribution")
    fig2 = px.histogram(df_f, x="grading_status")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Professor map (PCA)")
    X = StandardScaler().fit_transform(
        prof_agg[["overall_score", "mean_sentiment"]].fillna(0)
    )
    pca = PCA(n_components=2)
    p = pca.fit_transform(X)

    prof_agg["pca1"], prof_agg["pca2"] = p[:, 0], p[:, 1]

    fig3 = px.scatter(
        prof_agg,
        x="pca1",
        y="pca2",
        size="n_comments",
        color="overall_score",
        hover_name="professor"
    )
    st.plotly_chart(fig3, use_container_width=True)

# =============================
# Search & Filter
# =============================
elif page == "Search & Filter":
    st.header("Search & Filter")

    q = st.text_input("Search professor or course")

    subset = df_f.copy()
    if q:
        subset = subset[
            subset["professor"].str.contains(q, case=False, na=False) |
            subset["course"].str.contains(q, case=False, na=False)
        ]

    profs = professor_aggregate(subset).sort_values("overall_score", ascending=False)

    for _, r in profs.iterrows():
        with st.expander(f"{r['professor']} — ⭐ {r['overall_score']:.2f}"):
            st.write("Comments:", int(r["n_comments"]))
            st.write("Mean sentiment:", round(r["mean_sentiment"], 2))
            st.write("Grading:", r["grading_mode"])
            st.write("Attendance:", r["attendance_mode"])

            courses = subset[subset["professor"] == r["professor"]]["course"].unique()
            st.write("Courses:", "، ".join(courses))

# =============================
# Professor Profile
# =============================
elif page == "Professor Profile":
    st.header("Professor Profile")

    prof_name = st.selectbox(
        "Select professor",
        sorted(df_f["professor"].unique())
    )

    rows = df_f[df_f["professor"] == prof_name]
    agg = professor_aggregate(rows).iloc[0]

    c1, c2 = st.columns([2, 1])
    c1.metric("Overall score", f"{agg['overall_score']:.2f}")
    c1.metric("Mean sentiment", f"{agg['mean_sentiment']:.2f}")
    c1.metric("Comments", int(agg["n_comments"]))

    radar_vals = rows[NUM_COLS].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_vals.values,
        theta=NUM_COLS,
        fill="toself"
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])))
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Courses")
    st.write("، ".join(rows["course"].unique()))

    st.subheader("Sentiment over terms")
    ts = rows.groupby("term")["sentiment"].mean()
    st.plotly_chart(px.line(ts, markers=True), use_container_width=True)

    st.subheader("Sample comments")
    st.dataframe(
        rows[["term", "course", "comment_text", "sentiment"]]
        .sort_values("sentiment", ascending=False)
        .head(50)
    )

# =============================
# Compare
# =============================
elif page == "Compare":
    st.header("Compare Professors")

    selected = st.multiselect(
        "Select professors",
        prof_agg["professor"].tolist()
    )

    if len(selected) >= 2:
        comp = prof_agg[prof_agg["professor"].isin(selected)]

        st.plotly_chart(
            px.bar(comp, x="professor", y="overall_score"),
            use_container_width=True
        )

        radar = df_f[df_f["professor"].isin(selected)].groupby("professor")[NUM_COLS].mean()

        fig = go.Figure()
        for p in radar.index:
            fig.add_trace(go.Scatterpolar(
                r=radar.loc[p].values,
                theta=NUM_COLS,
                fill="toself",
                name=p
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp)

# =============================
# Recommender
# =============================
elif page == "Recommender":
    st.header("Recommender")

    w_overall = st.slider("Overall score importance", 0.0, 1.0, 0.5)
    w_teaching = st.slider("Teaching importance", 0.0, 1.0, 0.5)

    prof_agg["teaching_mean"] = (
        df_f.groupby("professor")["teaching"].mean()
        .reindex(prof_agg["professor"])
        .values
    )

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(
        prof_agg[["overall_score", "teaching_mean"]].fillna(0)
    )

    prof_agg["rec_score"] = (
        norm[:, 0] * w_overall +
        norm[:, 1] * w_teaching
    )

    top = prof_agg.sort_values("rec_score", ascending=False).head(10)

    for _, r in top.iterrows():
        st.markdown(f"### {r['professor']}")
        st.write("Overall:", round(r["overall_score"], 2))
        st.write("Teaching:", round(r["teaching_mean"], 2))

        courses = (
            df_f[df_f["professor"] == r["professor"]]["course"]
            .value_counts()
            .head(3)
            .index
            .tolist()
        )
        st.write("Top courses:", "، ".join(courses))
        st.markdown("---")