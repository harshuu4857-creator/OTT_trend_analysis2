import streamlit as st
import pandas as pd
import numpy as np
import requests
import ast
import matplotlib.pyplot as plt
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="OTT Trend Analysis",
    layout="wide"
)

# =====================================================
# CACHE DATA
# =====================================================

@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    return movies

movies = load_data()

# =====================================================
# LOAD MODEL + VECTORIZER
# =====================================================

model = joblib.load("movie_model.pkl")
cv = joblib.load("vectorizer.pkl")

# =====================================================
# DATA CLEANING
# =====================================================

movies = movies[
    [
        'title',
        'genres',
        'overview',
        'release_date',
        'vote_average',
        'popularity',
        'budget',
        'runtime',
        'original_language',
        'cast',
        'crew'
    ]
]

movies.dropna(inplace=True)

movies['year'] = movies['release_date'].apply(
    lambda x: int(x.split("-")[0])
)

# =====================================================
# GENRE CONVERT
# =====================================================

def convert(obj):

    L = []

    for i in ast.literal_eval(obj):
        L.append(i['name'])

    return L

movies['genres'] = movies['genres'].apply(convert)

movies['genres_text'] = movies['genres'].apply(
    lambda x: " ".join(x)
)

# =====================================================
# DIRECTOR
# =====================================================

def fetch_director(obj):

    L = []

    for i in ast.literal_eval(obj):

        if i['job'] == 'Director':
            L.append(i['name'])

    return ", ".join(L)

movies['director'] = movies['crew'].apply(fetch_director)

# =====================================================
# CAST
# =====================================================

def fetch_cast(obj):

    L = []

    counter = 0

    for i in ast.literal_eval(obj):

        if counter != 3:
            L.append(i['name'])
            counter += 1

        else:
            break

    return ", ".join(L)

movies['cast_names'] = movies['cast'].apply(fetch_cast)

# =====================================================
# TMDB API
# =====================================================

TMDB_API_KEY = "5609ab5a9c50d7e2e03b53ff1e36401a"

# =====================================================
# POSTER CACHE
# =====================================================

@st.cache_data
def fetch_poster(movie_name):

    try:

        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"

        data = requests.get(url).json()

        if data['results']:

            poster_path = data['results'][0]['poster_path']

            if poster_path:

                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

        return "https://via.placeholder.com/500x750?text=No+Image"

    except:

        return "https://via.placeholder.com/500x750?text=Error"

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: sans-serif;
}

.block-container {
    padding-top: 1rem;
}

.title {
    font-size: 60px;
    font-weight: 900;
    color: #ff1e2d;
}

.subtitle {
    color: #bbbbbb;
    font-size: 18px;
}

.metric-box {
    background: #111111;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
}

.movie-card {
    background: #111111;
    padding: 12px;
    border-radius: 20px;
    transition: 0.3s;
    margin-bottom: 20px;
}

.movie-card:hover {
    transform: scale(1.03);
}

.movie-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 10px;
}

.movie-info {
    color: #cccccc;
    margin-top: 5px;
    font-size: 14px;
}

.stButton > button {
    background: #ff1e2d;
    color: white;
    border: none;
    padding: 14px 30px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎬 OTT AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "All Movies"
    ]
)

# =====================================================
# ALL MOVIES PAGE
# =====================================================

if page == "All Movies":

    st.title("📊 OTT Analytics Dashboard")

    st.dataframe(
        movies[
            [
                'title',
                'vote_average',
                'popularity',
                'original_language',
                'budget',
                'year'
            ]
        ]
    )

    st.markdown("---")

    st.subheader("📈 OTT Trend Insights")

    col1, col2, col3 = st.columns(3)

    genre_count = movies['genres'].explode().value_counts()

    with col1:

        st.metric(
            "🔥 Most Popular Genre",
            genre_count.index[0]
        )

    with col2:

        st.metric(
            "🌍 Top Language",
            movies['original_language'].mode()[0]
        )

    best_year = movies.groupby(
        'year'
    )['vote_average'].mean().idxmax()

    with col3:

        st.metric(
            "🏆 Highest Rated Year",
            int(best_year)
        )

    st.markdown("---")

    # =================================================
    # GRAPH 1
    # =================================================

    st.subheader("⭐ Average Rating Trend")

    rating_trend = movies.groupby(
        'year'
    )['vote_average'].mean()

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        rating_trend.index,
        rating_trend.values
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Rating")

    st.pyplot(fig)

    # =================================================
    # GRAPH 2
    # =================================================

    st.subheader("💰 Budget vs Popularity")

    fig2, ax2 = plt.subplots(figsize=(10,4))

    ax2.scatter(
        movies['budget'],
        movies['popularity']
    )

    ax2.set_xlabel("Budget")
    ax2.set_ylabel("Popularity")

    st.pyplot(fig2)

# =====================================================
# PREDICTION PAGE
# =====================================================

if page == "Prediction":

    # =================================================
    # HEADER
    # =================================================

    col1, col2, col3 = st.columns([6,1,1])

    with col1:

        st.markdown(
            "<div class='title'>OTT Trend Analysis</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='subtitle'>Created by Harsh Patel</div>",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class='metric-box'>
            <h1>{movies.shape[0]}</h1>
            Movies
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        total_genres = len(
            set(
                " ".join(
                    movies['genres_text']
                ).split()
            )
        )

        st.markdown(
            f"""
            <div class='metric-box'>
            <h1>{total_genres}</h1>
            Genres
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =================================================
    # FILTERS
    # =================================================

    c1, c2, c3, c4 = st.columns([3,1,1,2])

    all_genres = sorted(
        set(
            " ".join(
                movies['genres_text']
            ).split()
        )
    )

    with c1:

        selected_genres = st.multiselect(
            "🎭 Genres",
            all_genres,
            placeholder="Choose genres..."
        )

    with c2:

        year = st.slider(
            "📅 Year",
            1980,
            2020,
            2015
        )

    with c3:

        rating = st.slider(
            "⭐ Rating",
            1.0,
            10.0,
            7.0
        )

    with c4:

        search = st.text_input(
            "🔍 Search",
            placeholder="Search movie"
        )

    show = st.button("🚀 Discover Movies")

    # =================================================
    # PREDICT
    # =================================================

    if show:

        filtered = movies.copy()

        # =============================================
        # GENRE FILTER
        # =============================================

        if selected_genres:

            for genre in selected_genres:

                filtered = filtered[
                    filtered['genres_text'].str.contains(
                        genre,
                        case=False
                    )
                ]

        # =============================================
        # YEAR FILTER
        # =============================================

        filtered = filtered[
            abs(filtered['year'] - year) <= 5
        ]

        # =============================================
        # RATING FILTER
        # =============================================

        filtered = filtered[
            filtered['vote_average'] >= rating
        ]

        # =============================================
        # SEARCH FILTER
        # =============================================

        if search:

            filtered = filtered[
                filtered['title'].str.contains(
                    search,
                    case=False
                )
            ]

        # =============================================
        # PREDICTION
        # =============================================

        genre_vec_all = cv.transform(
            filtered['genres_text']
        ).toarray()

        year_vec_all = filtered[
            'year'
        ].values.reshape(-1,1)

        X_all = np.concatenate(
            (
                genre_vec_all,
                year_vec_all
            ),
            axis=1
        )

        filtered['predicted_rating'] = model.predict(X_all)

        top_movies = filtered.sort_values(
            by='predicted_rating',
            ascending=False
        ).head(8)

        # =============================================
        # RESULTS
        # =============================================

        st.markdown("---")

        st.subheader("🔥 Recommended Movies")

        cols = st.columns(4)

        for i, row in top_movies.iterrows():

            with cols[i % 4]:

                poster = fetch_poster(row['title'])

                st.markdown(
                    "<div class='movie-card'>",
                    unsafe_allow_html=True
                )

                st.image(
                    poster,
                    use_container_width=True
                )

                st.markdown(
                    f"<div class='movie-title'>{row['title']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>⭐ {round(row['vote_average'],1)}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>🎯 {row['movie_status']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>🌍 {row['original_language']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>🔥 Popularity: {round(row['popularity'],1)}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>💰 Budget: ${row['budget']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>🎬 Director: {row['director']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>🎭 Cast: {row['cast_names']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>📅 {row['year']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>⏱ {row['runtime']} min</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>{row['overview'][:180]}...</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )