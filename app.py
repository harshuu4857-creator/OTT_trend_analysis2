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
    page_title="Netflix AI Movie Predictor",
    layout="wide",
    page_icon="🎬"
)

# =====================================================
# LOAD SAVED MODEL + VECTORIZER
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
        'vote_average'
    ]
]

movies.dropna(inplace=True)

# =====================================================
# YEAR
# =====================================================
movies['year'] = movies['release_date'].apply(
    lambda x: int(x.split("-")[0])
)

# =====================================================
# GENRES
# =====================================================
def convert(obj):

    L = []

    for i in ast.literal_eval(obj):
        L.append(i['name'])

    return L

movies['genres'] = movies['genres'].apply(convert)

# =====================================================
# CAST
# =====================================================
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

# =====================================================
# DIRECTOR
# =====================================================
def fetch_director(obj):

    L = []

    for i in ast.literal_eval(obj):

        if i['job'] == 'Director':

            L.append(i['name'])

            break

    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# =====================================================
# CLEAN SPACES
# =====================================================
movies['genres'] = movies['genres'].apply(
    lambda x:[i.replace(" ","") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x:[i.replace(" ","") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x:[i.replace(" ","") for i in x]
)

# =====================================================
# TAGS
# =====================================================
movies['tags'] = (
    movies['genres'] +
    movies['cast'] +
    movies['crew']
)

movies['tags'] = movies['tags'].apply(
    lambda x:" ".join(x)
)

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #0b0f1a;
    color: white;
    font-family: 'Poppins', sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.hero {
    position: relative;
    height: 500px;
    border-radius: 25px;
    overflow: hidden;
    margin-bottom: 40px;
}

.big-title {
    font-size: 70px;
    font-weight: 800;
    letter-spacing: -2px;
}

.subtitle {
    color: #b3b3b3;
    font-size: 20px;
    margin-top: -10px;
}

.section-title {
    font-size: 30px;
    font-weight: 700;
    margin-top: 40px;
    margin-bottom: 20px;
}

.movie-card {
    background: #141414;
    border-radius: 20px;
    overflow: hidden;
    transition: 0.4s;
    cursor: pointer;
    margin-bottom: 20px;
}

.movie-card:hover {
    transform: scale(1.06);
}

.movie-name {
    font-size: 16px;
    font-weight: 600;
    margin-top: 10px;
}

.movie-info {
    color: #b3b3b3;
    font-size: 14px;
}

.stButton>button {
    background: linear-gradient(90deg,#e50914,#ff4d4d);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    width: 220px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div>
    <div class='big-title'>NETFLIX AI</div>
    <div class='subtitle'>
        AI Powered Movie Recommendation System
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# INPUTS
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

col1, col2 = st.columns([2,1])

with col1:

    selected_genres = st.multiselect(
        "🎭 Select Genre(s)",
        all_genres
    )

with col2:

    year = st.slider(
        "📅 Select Year",
        1980,
        2020,
        2015
    )

# =====================================================
# SEARCH
# =====================================================
search = st.text_input(
    "🔍 Search Movie",
    placeholder="Search movies like Interstellar..."
)

# =====================================================
# BUTTON
# =====================================================
if st.button("🎬 Discover Movies"):

    if not selected_genres:

        st.warning("Please select at least one genre")

    else:

        # FILTER
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

        # VECTORIZE
        vectors_filtered = cv.transform(
            filtered['tags']
        ).toarray()

        years_filtered = filtered['year'].values.reshape(-1,1)

        X_filtered = np.concatenate(
            (vectors_filtered, years_filtered),
            axis=1
        )

        # PREDICT
        filtered['predicted_rating'] = model.predict(
            X_filtered
        )

        # SORT
        top_movies = filtered.sort_values(
            by='predicted_rating',
            ascending=False
        ).head(10)

        # HERO
        hero_movie = top_movies.iloc[0]

        hero_poster = fetch_poster(
            hero_movie['title']
        )

        st.markdown(
            f"""
            <div class="hero"
            style="
            background-image:
            linear-gradient(
            to right,
            rgba(0,0,0,0.95),
            rgba(0,0,0,0.2)),
            url('{hero_poster}');

            background-size: cover;
            background-position: center;
            ">

            <div style="
            position:absolute;
            bottom:60px;
            left:50px;
            width:50%;
            ">

            <h1 style="
            font-size:60px;
            ">
            {hero_movie['title']}
            </h1>

            <h3 style="color:#e50914;">
            ⭐ {round(hero_movie['predicted_rating'],2)}
            </h3>

            <p style="
            color:#d1d1d1;
            font-size:18px;
            ">
            AI-selected premium recommendation.
            </p>

            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # RECOMMENDED
        st.markdown(
            "<div class='section-title'>🔥 Recommended For You</div>",
            unsafe_allow_html=True
        )

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
                    f"<div class='movie-info'>⭐ {round(row.predicted_rating,2)}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='movie-info'>📅 {row.year}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )