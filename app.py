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

# =========================
# 🎯 YEAR
# =========================
movies['year'] = movies['release_date'].apply(lambda x: int(x.split("-")[0]))

# =========================
# 🔄 GENRE CONVERT
# =========================
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))

# =========================
# 🤖 VECTORIZE
# =========================
cv = CountVectorizer(max_features=1000)
genre_matrix = cv.fit_transform(movies['genres']).toarray()

year_feature = movies['year'].values.reshape(-1,1)

X = np.concatenate((genre_matrix, year_feature), axis=1)
y = movies['vote_average']

# =========================
# 🤖 MODEL
# =========================
model = RandomForestRegressor()
model.fit(X, y)

# =========================
# 🎨 NETFLIX STYLE CSS
# =========================
st.set_page_config(layout="wide")

st.markdown("""
<style>
body {background-color: #0e1117; color: white;}

.card {
    background-color: #141414;
    border-radius: 12px;
    padding: 10px;
    transition: transform 0.3s;
    text-align: center;
}
.card:hover {
    transform: scale(1.08);
}

.title {
    font-size: 14px;
    font-weight: bold;
    margin-top: 5px;
}

.rating {
    color: #f5c518;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🎬 HEADER
# =========================
st.title("🎬 Netflix Style Movie Predictor")
st.write("Find movies using Genre & Year")

# =========================
# 🎯 INPUT
# =========================
all_genres = sorted(set(" ".join(movies['genres']).split()))
selected_genres = st.multiselect("Select Genre(s)", all_genres)
year = st.slider("Select Year", 1980, 2020, 2015)

# =========================
# 🔮 PREDICTION + RESULTS
# =========================
if st.button("Show Movies"):
    if not selected_genres:
        st.warning("Select at least one genre")
    else:
        # Prepare input
        input_genre = " ".join(selected_genres)
        genre_vec = cv.transform([input_genre]).toarray()
        year_vec = np.array([[year]])
        input_data = np.concatenate((genre_vec, year_vec), axis=1)

        base_rating = model.predict(input_data)[0]

        # Filter movies
        filtered = movies[
            (movies['genres'].str.contains(selected_genres[0], case=False)) &
            (abs(movies['year'] - year) <= 3)
        ].copy()

        # Predict for all filtered
        genre_vec_all = cv.transform(filtered['genres']).toarray()
        year_vec_all = filtered['year'].values.reshape(-1,1)
        X_all = np.concatenate((genre_vec_all, year_vec_all), axis=1)

        filtered['predicted_rating'] = model.predict(X_all)

        top_movies = filtered.sort_values(by='predicted_rating', ascending=False).head(10)

        st.subheader("🔥 Recommended Movies")

        cols = st.columns(5)

        for i, row in top_movies.iterrows():
            with cols[i % 5]:
                poster = fetch_poster(row['title'])

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(poster, use_container_width=True)
                st.markdown(f"<div class='title'>{row['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rating'>⭐ {round(row['predicted_rating'],2)}</div>", unsafe_allow_html=True)
                st.markdown(f"📅 {row['year']}")
                st.markdown('</div>', unsafe_allow_html=True)