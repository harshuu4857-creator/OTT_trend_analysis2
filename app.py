import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Netflix AI",
    layout="wide",
    page_icon="🎬"
)

# =====================================================
# LOAD MODEL + VECTORIZER
# =====================================================
model = joblib.load("movie_model.pkl")
cv = joblib.load("vectorizer.pkl")

# =====================================================
# TMDB API
# =====================================================
TMDB_API_KEY = "5609ab5a9c50d7e2e03b53ff1e36401a"

# =====================================================
# FETCH POSTER
# =====================================================
@st.cache_data
def fetch_poster(movie_name):

    try:

        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"

        data = requests.get(url).json()

        if data['results']:

            poster_path = data['results'][0]['poster_path']

            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return "https://via.placeholder.com/500x750?text=No+Image"

    except:
        return "https://via.placeholder.com/500x750?text=Error"

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on='title')

    movies = movies[
        [
            'title',
            'genres',
            'cast',
            'crew',
            'release_date',
            'vote_average',
            'budget',
            'original_language',
            'popularity'
        ]
    ]

    movies.dropna(inplace=True)

    # YEAR
    movies['year'] = movies['release_date'].apply(
        lambda x: int(x.split("-")[0])
    )

    # GENRES
    def convert(obj):

        L = []

        for i in ast.literal_eval(obj):
            L.append(i['name'])

        return L

    movies['genres'] = movies['genres'].apply(convert)

    # CAST
    def get_cast(obj):

        L = []
        counter = 0

        for i in ast.literal_eval(obj):

            if counter != 3:
                L.append(i['name'])
                counter += 1

            else:
                break

        return L

    movies['cast'] = movies['cast'].apply(get_cast)

    # DIRECTOR
    def fetch_director(obj):

        L = []

        for i in ast.literal_eval(obj):

            if i['job'] == 'Director':
                L.append(i['name'])
                break

        return L

    movies['crew'] = movies['crew'].apply(fetch_director)

    # REMOVE SPACES
    movies['genres'] = movies['genres'].apply(
        lambda x:[i.replace(" ","") for i in x]
    )

    movies['cast'] = movies['cast'].apply(
        lambda x:[i.replace(" ","") for i in x]
    )

    movies['crew'] = movies['crew'].apply(
        lambda x:[i.replace(" ","") for i in x]
    )

    # TAGS
    movies['tags'] = (
        movies['genres'] +
        movies['cast'] +
        movies['crew']
    )

    movies['tags'] = movies['tags'].apply(
        lambda x:" ".join(x)
    )

    return movies

movies = load_data()

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0b0f1a;
    color: white;
    font-family: sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* NAVBAR */

.navbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:18px 25px;
    background:#111111;
    border-radius:18px;
    margin-bottom:25px;
}

.logo {
    font-size:36px;
    font-weight:900;
    color:#E50914;
}

.stats {
    display:flex;
    gap:18px;
}

.stat-box {
    background:#181818;
    padding:10px 18px;
    border-radius:12px;
    text-align:center;
}

.stat-number {
    font-size:24px;
    font-weight:bold;
}

.stat-label {
    color:#aaaaaa;
    font-size:13px;
}

/* HERO */

.hero-container {
    position: relative;
    height: 600px;
    border-radius: 25px;
    overflow: hidden;
    margin-top: 20px;
    margin-bottom: 40px;
    background-size: cover;
    background-position: center;
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to right,
        rgba(0,0,0,0.95),
        rgba(0,0,0,0.2)
    );
}

.hero-content {
    position: absolute;
    bottom: 60px;
    left: 50px;
    width: 40%;
    z-index: 2;
}

.hero-title {
    font-size: 65px;
    font-weight: 900;
    line-height:1;
    margin-bottom:15px;
}

.hero-rating {
    color: #e50914;
    font-size: 28px;
    font-weight: bold;
    margin-bottom:10px;
}

.hero-desc {
    color: #dddddd;
    font-size: 18px;
    line-height:1.6;
}

/* SECTION */

.section-title {
    font-size: 34px;
    font-weight: 800;
    margin-top:20px;
    margin-bottom: 20px;
}

/* CARDS */

.movie-card {
    background: #141414;
    padding: 12px;
    border-radius: 18px;
    transition: 0.4s;
    margin-bottom: 20px;
}

.movie-card:hover {
    transform: scale(1.05);
}

.movie-name {
    font-size: 18px;
    font-weight: 700;
    margin-top: 10px;
}

.movie-info {
    color: #d1d1d1;
    font-size: 14px;
    margin-top: 4px;
}

/* BUTTON */

.stButton > button {
    background: linear-gradient(90deg,#e50914,#ff4d4d);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    width: 220px;
    font-size: 18px;
    font-weight: bold;
}

/* INPUTS */

.stTextInput input {
    background-color:#181818;
    color:white;
    border-radius:12px;
}

.stMultiSelect div {
    background-color:#181818;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# NAVBAR
# =====================================================

total_movies = len(movies)

all_genres_count = len(
    set(
        " ".join(
            movies['genres'].apply(
                lambda x:" ".join(x)
            )
        ).split()
    )
)

st.markdown(f"""
<div class="navbar">

<div class="logo">
NETFLIX AI
</div>

<div class="stats">

<div class="stat-box">
<div class="stat-number">{total_movies}</div>
<div class="stat-label">Movies</div>
</div>

<div class="stat-box">
<div class="stat-number">{all_genres_count}</div>
<div class="stat-label">Genres</div>
</div>

</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# FILTERS
# =====================================================

all_genres = sorted(
    set(
        " ".join(
            movies['genres'].apply(
                lambda x:" ".join(x)
            )
        ).split()
    )
)

col1, col2, col3 = st.columns([2,1,1])

with col1:

    selected_genres = st.multiselect(
        "",
        all_genres,
        placeholder="🎭 Select Genres"
    )

with col2:

    year = st.slider(
        "📅 Year",
        1980,
        2020,
        2015
    )

with col3:

    search = st.text_input(
        "",
        placeholder="🔍 Search Movie"
    )

# =====================================================
# BUTTON
# =====================================================

discover = st.button("🎬 Discover Movies")

# =====================================================
# DEFAULT HERO
# =====================================================

default_movie = movies.sort_values(
    by='vote_average',
    ascending=False
).iloc[0]

# =====================================================
# RESULTS
# =====================================================

if discover:

    if not selected_genres:

        st.warning("Please select at least one genre")

    else:

        filtered = movies[
            (
                movies['genres'].astype(str).str.contains(
                    selected_genres[0],
                    case=False
                )
            )
            &
            (
                abs(movies['year'] - year) <= 5
            )
        ].copy()

        if search:

            filtered = filtered[
                filtered['title'].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        vectors_filtered = cv.transform(
            filtered['tags']
        ).toarray()

        years_filtered = filtered['year'].values.reshape(-1,1)

        X_filtered = np.concatenate(
            (vectors_filtered, years_filtered),
            axis=1
        )

        filtered['predicted_rating'] = model.predict(
            X_filtered
        )

        top_movies = filtered.sort_values(
            by='predicted_rating',
            ascending=False
        ).head(10)

        hero_movie = top_movies.iloc[0]

else:

    top_movies = movies.sort_values(
        by='vote_average',
        ascending=False
    ).head(10)

    hero_movie = default_movie

# =====================================================
# HERO SECTION
# =====================================================

hero_poster = fetch_poster(hero_movie['title'])

hero_html = f"""
<div class="hero-container"
style="
background-image:
linear-gradient(
to right,
rgba(0,0,0,0.95),
rgba(0,0,0,0.3)),
url('{hero_poster}');
">

<div class="hero-overlay"></div>

<div class="hero-content">

<div class="hero-title">
{hero_movie['title']}
</div>

<div class="hero-rating">
⭐ {round(hero_movie['vote_average'],2)}
</div>

<div class="hero-desc">
AI powered OTT trend analysis and movie recommendation system based on genres, popularity, ratings and audience trends.
</div>

</div>

</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

# =====================================================
# SECTION TITLE
# =====================================================

st.markdown(
    "<div class='section-title'>🔥 Trending & Recommended</div>",
    unsafe_allow_html=True
)

# =====================================================
# MOVIE GRID
# =====================================================

cols = st.columns(5)

for i, row in enumerate(top_movies.itertuples()):

    with cols[i % 5]:

        poster = fetch_poster(row.title)

        st.markdown(
            "<div class='movie-card'>",
            unsafe_allow_html=True
        )

        st.image(
            poster,
            use_container_width=True
        )

        st.markdown(
            f"<div class='movie-name'>{row.title}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='movie-info'>⭐ Rating: {round(row.vote_average,2)}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='movie-info'>🔥 Popularity: {round(row.popularity,2)}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='movie-info'>🌍 Language: {row.original_language.upper()}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='movie-info'>💰 Budget: ${int(row.budget):,}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='movie-info'>📅 Year: {row.year}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )