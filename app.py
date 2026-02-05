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
safe_label = lambda x: "نامشخص" if pd.isna(x) else str(x)


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

    q = st.text_input("Search professor or course (partial match)")

    sort_by = st.selectbox("Sort results by", ["Overall score", "Number of distinct courses"], index=0)
    min_courses = st.number_input("Minimum distinct courses to show", min_value=0, value=0, step=1)
    top_k = st.number_input("Show top K courses in the detail", min_value=1, value=3, step=1)

    subset = df_f.copy()

    if q:
        mask_q = (
            subset["professor"].fillna("").str.contains(q, case=False, na=False) |
            subset["course"].fillna("").str.contains(q, case=False, na=False)
        )
        subset = subset[mask_q]

    profs = professor_aggregate(subset)

    course_counts = subset.groupby("professor")["course"].nunique().rename("n_courses")
    profs = profs.merge(course_counts, how="left", left_on="professor", right_index=True)
    profs["n_courses"] = profs["n_courses"].fillna(0).astype(int)

    if min_courses > 0:
        profs = profs[profs["n_courses"] >= int(min_courses)]

    if sort_by == "Number of distinct courses":
        profs = profs.sort_values(["n_courses", "overall_score"], ascending=[False, False])
    else:
        profs = profs.sort_values(["overall_score", "n_courses"], ascending=[False, False])

    st.write(f"Found {len(profs)} professors matching filters")

    for _, r in profs.iterrows():
        header = f"{r['professor']} — ⭐ {r['overall_score']:.2f} — courses: {int(r['n_courses'])} — comments: {int(r['n_comments'])}"
        with st.expander(header):
            st.write("Mean sentiment:", round(r["mean_sentiment"], 2))
            st.write("Grading:", r["grading_mode"])
            st.write("Attendance:", r["attendance_mode"])

            courses_series = (
                subset[subset["professor"] == r["professor"]]["course"]
                .fillna("")
                .value_counts()
            )
            top_courses = courses_series.head(top_k).index.tolist()
            st.write("Top courses:", "، ".join([c for c in top_courses if c]))

            if not courses_series.empty:
                cs_df = courses_series.reset_index()
                cs_df.columns = ["course", "count"]
                st.table(cs_df.head(20))


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

    st.markdown("Choose preferences and optionally a course. If a course is selected, the recommender ranks professors who taught that course.")

    w_overall = st.slider("Overall score importance", 0.0, 1.0, 0.5)
    w_teaching = st.slider("Teaching importance", 0.0, 1.0, 0.5)
    w_sentiment = st.slider("Student sentiment importance", 0.0, 1.0, 0.0)

    courses_list = sorted(df_f['course'].dropna().unique().tolist())
    course_sel = st.selectbox("Filter by course (optional)", options=["All courses"] + courses_list, index=0)

    if course_sel == "All courses":
        base_df = df_f.copy()
        note = "Recommendations across all courses."
    else:
        base_df = df_f[df_f['course'] == course_sel].copy()
        note = f"Recommendations restricted to course: {course_sel}"

    st.markdown(f"**Note:** {note}")

    if base_df.shape[0] == 0:
        st.warning("No data available for the selected course.")
    else:
        prof_df = professor_aggregate(base_df).set_index('professor')

        teaching_mean = base_df.groupby('professor')['teaching'].mean()
        prof_df['teaching_mean'] = teaching_mean.reindex(prof_df.index).fillna(0).values

        prof_df['mean_sentiment'] = prof_df['mean_sentiment'].fillna(0)

        features_for_norm = prof_df[['overall_score', 'teaching_mean', 'mean_sentiment']].fillna(0).values
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(features_for_norm)

        prof_df['rec_score'] = (
            norm[:, 0] * w_overall +
            norm[:, 1] * w_teaching +
            norm[:, 2] * w_sentiment
        )

        counts = base_df.groupby('professor').size().rename('n_comments_course')
        prof_df['n_comments_course'] = counts.reindex(prof_df.index).fillna(0).astype(int).values

        top_k = st.number_input("How many top professors to show", min_value=1, max_value=50, value=10, step=1)
        top = prof_df.sort_values('rec_score', ascending=False).head(top_k)

        st.subheader("Top recommendations")
        for prof_name, row in top.iterrows():
            st.markdown(f"### {prof_name}")
            st.write("Recommendation score:", round(row['rec_score'], 3))
            st.write("Overall score (on selected scope):", round(row['overall_score'], 2))
            st.write("Teaching mean (on selected scope):", round(row['teaching_mean'], 2))
            st.write("Mean sentiment (on selected scope):", round(row['mean_sentiment'], 2))
            st.write("Comments for this scope:", int(row['n_comments_course']))
            st.write("Most frequent grading:", safe_label(row.get('grading_mode', None)))

            top_courses = (
                df_f[df_f['professor'] == prof_name]['course']
                .fillna('')
                .value_counts()
                .head(5)
                .index
                .tolist()
            )
            if top_courses:
                st.write("Top courses (global):", "، ".join([c for c in top_courses if c]))
            st.markdown("---")
