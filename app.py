import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Netflix Movie Dashboard", layout="wide")

st.title("Netflix Movie Dashboard")
st.markdown("Explore Netflix movie data interactively.")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Try converting common date/year fields
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    return df

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Column Names")
    st.write(list(df.columns))

    st.sidebar.header("Filters")

    filtered_df = df.copy()

    # Title filter
    if "title" in df.columns:
        title_search = st.sidebar.text_input("Search by title")
        if title_search:
            filtered_df = filtered_df[
                filtered_df["title"].astype(str).str.contains(title_search, case=False, na=False)
            ]

    # Type filter
    if "type" in df.columns:
        type_options = filtered_df["type"].dropna().unique().tolist()
        selected_type = st.sidebar.multiselect("Select type", type_options, default=type_options)
        if selected_type:
            filtered_df = filtered_df[filtered_df["type"].isin(selected_type)]

    # Genre filter
    genre_col = None
    for col in ["listed_in", "genre", "genres", "category"]:
        if col in filtered_df.columns:
            genre_col = col
            break

    if genre_col:
        all_genres = []
        for item in filtered_df[genre_col].dropna():
            all_genres.extend([g.strip() for g in str(item).split(",")])
        all_genres = sorted(list(set(all_genres)))

        selected_genres = st.sidebar.multiselect("Select genre", all_genres)
        if selected_genres:
            filtered_df = filtered_df[
                filtered_df[genre_col].astype(str).apply(
                    lambda x: any(g in [i.strip() for i in x.split(",")] for g in selected_genres)
                )
            ]

    # Release year filter
    if "release_year" in filtered_df.columns:
        min_year = int(filtered_df["release_year"].dropna().min()) if filtered_df["release_year"].dropna().shape[0] > 0 else 2000
        max_year = int(filtered_df["release_year"].dropna().max()) if filtered_df["release_year"].dropna().shape[0] > 0 else 2024

        year_range = st.sidebar.slider(
            "Select release year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        filtered_df = filtered_df[
            (filtered_df["release_year"] >= year_range[0]) &
            (filtered_df["release_year"] <= year_range[1])
        ]

    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

    st.subheader("Visualizations")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if "release_year" in filtered_df.columns:
            year_count = (
                filtered_df["release_year"]
                .dropna()
                .value_counts()
                .sort_index()
                .reset_index()
            )
            year_count.columns = ["release_year", "count"]

            fig_year = px.line(
                year_count,
                x="release_year",
                y="count",
                title="Content Released by Year",
                markers=True
            )
            st.plotly_chart(fig_year, use_container_width=True)

    with chart_col2:
        if genre_col:
            genre_list = []
            for item in filtered_df[genre_col].dropna():
                genre_list.extend([g.strip() for g in str(item).split(",")])

            if genre_list:
                genre_df = pd.DataFrame(pd.Series(genre_list).value_counts()).reset_index()
                genre_df.columns = ["genre", "count"]

                fig_genre = px.bar(
                    genre_df.head(10),
                    x="genre",
                    y="count",
                    title="Top 10 Genres"
                )
                st.plotly_chart(fig_genre, use_container_width=True)

    if "rating" in filtered_df.columns:
        st.subheader("Rating Distribution")
        rating_count = filtered_df["rating"].dropna().value_counts().reset_index()
        rating_count.columns = ["rating", "count"]

        fig_rating = px.pie(
            rating_count,
            names="rating",
            values="count",
            title="Rating Distribution"
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    if "director" in filtered_df.columns:
        st.subheader("Top Directors")
        director_count = (
            filtered_df["director"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )
        director_count.columns = ["director", "count"]

        fig_director = px.bar(
            director_count,
            x="director",
            y="count",
            title="Top 10 Directors"
        )
        st.plotly_chart(fig_director, use_container_width=True)

    st.subheader("Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered CSV",
        data=csv,
        file_name="filtered_netflix_data.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to begin.")
