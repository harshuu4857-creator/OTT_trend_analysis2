import streamlit as st
import pandas as pd
import ast
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor

# =========================
# 🔑 TMDB API KEY
# =========================
TMDB_API_KEY = "5609ab5a9c50d7e2e03b53ff1e36401a"

# =========================
# 🎬 Fetch Poster
# =========================
def fetch_poster(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
        data = requests.get(url).json()

        if data["results"]:
            poster_path = data["results"][0]["poster_path"]
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return "https://via.placeholder.com/300x450?text=No+Image"
    except:
        return "https://via.placeholder.com/300x450?text=Error"

# =========================
# 📊 LOAD DATA
# =========================
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['title', 'genres', 'release_date', 'vote_average']]
movies.dropna(inplace=True)

movies['year'] = movies['release_date'].apply(lambda x: int(x.split("-")[0]))

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))

# =========================
# 🤖 MODEL
# =========================
cv = CountVectorizer(max_features=1000)
genre_matrix = cv.fit_transform(movies['genres']).toarray()

year_feature = movies['year'].values.reshape(-1,1)
X = np.concatenate((genre_matrix, year_feature), axis=1)
y = movies['vote_average']

model = RandomForestRegressor()
model.fit(X, y)

# =========================
# 🎨 PREMIUM CSS
# =========================
st.set_page_config(layout="wide")

st.markdown("""
<style>
body {background-color: #0b0c10; color: white;}
.block-container {padding: 2rem 4rem;}

img {
    border-radius: 10px;
    transition: transform 0.3s;
}
img:hover {
    transform: scale(1.08);
}

.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 30px;
}

.hero-title {
    font-size: 32px;
    font-weight: bold;
}

.subtle {
    color: #b3b3b3;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🎬 HEADER
# =========================
st.title("🎬 Netflix Style Movie Predictor")
st.markdown("<div class='subtle'>Discover movies using Genre & Year</div>", unsafe_allow_html=True)

# =========================
# 🎯 INPUT
# =========================
all_genres = sorted(set(" ".join(movies['genres']).split()))
selected_genres = st.multiselect("Select Genre(s)", all_genres)
year = st.slider("Select Year", 1980, 2020, 2015)

# =========================
# 🔮 RESULTS
# =========================
if st.button("Show Movies"):
    if not selected_genres:
        st.warning("Select at least one genre")
    else:
        input_genre = " ".join(selected_genres)
        genre_vec = cv.transform([input_genre]).toarray()
        year_vec = np.array([[year]])
        input_data = np.concatenate((genre_vec, year_vec), axis=1)

        # Filter
        filtered = movies[
            (movies['genres'].str.contains(selected_genres[0], case=False)) &
            (abs(movies['year'] - year) <= 3)
        ].copy()

        genre_vec_all = cv.transform(filtered['genres']).toarray()
        year_vec_all = filtered['year'].values.reshape(-1,1)
        X_all = np.concatenate((genre_vec_all, year_vec_all), axis=1)

        filtered['predicted_rating'] = model.predict(X_all)
        top_movies = filtered.sort_values(by='predicted_rating', ascending=False).head(10)

        # =========================
        # 🎬 HERO SECTION
        # =========================
        top = top_movies.iloc[0]
        st.markdown("<div class='section-title'>🎬 Featured</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1,2])

        with col1:
            st.image(fetch_poster(top['title']), use_container_width=True)

        with col2:
            st.markdown(f"<div class='hero-title'>{top['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"⭐ {round(top['predicted_rating'],2)}")
            st.markdown(f"📅 {top['year']}")
            st.markdown("<div class='subtle'>Top recommendation for you</div>", unsafe_allow_html=True)

        # =========================
        # 🔥 RECOMMENDED ROW
        # =========================
        st.markdown("<div class='section-title'>🔥 Recommended For You</div>", unsafe_allow_html=True)

        cols = st.columns(len(top_movies))

        for i, row in enumerate(top_movies.itertuples()):
            with cols[i]:
                st.image(fetch_poster(row.title), use_container_width=True)
                st.caption(row.title)

        # =========================
        # 🎯 TRENDING (same reused)
        # =========================
        st.markdown("<div class='section-title'>🎯 Trending</div>", unsafe_allow_html=True)

        trending = movies.sort_values(by='vote_average', ascending=False).head(10)

        cols = st.columns(len(trending))

        for i, row in enumerate(trending.itertuples()):
            with cols[i]:
                st.image(fetch_poster(row.title), use_container_width=True)
                st.caption(row.title)