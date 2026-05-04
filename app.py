import streamlit as st
import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor

# =========================
# 📊 LOAD DATA
# =========================
movies = pd.read_csv("tmdb_5000_movies.csv")

movies = movies[['title', 'genres', 'release_date', 'vote_average']]
movies.dropna(inplace=True)

# =========================
# 🎯 EXTRACT YEAR
# =========================
movies['year'] = movies['release_date'].apply(lambda x: int(x.split("-")[0]))

# =========================
# 🔄 CONVERT GENRES
# =========================
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))

# =========================
# 🤖 VECTORIZE GENRES
# =========================
cv = CountVectorizer(max_features=1000)
genre_matrix = cv.fit_transform(movies['genres']).toarray()

# =========================
# 📊 FEATURES + TARGET
# =========================
year_feature = movies['year'].values.reshape(-1,1)

X = np.concatenate((genre_matrix, year_feature), axis=1)
y = movies['vote_average']

# =========================
# 🤖 TRAIN MODEL
# =========================
model = RandomForestRegressor()
model.fit(X, y)

# =========================
# 🌐 NETFLIX STYLE UI
# =========================
st.set_page_config(page_title="Movie Predictor", layout="wide")

st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.card {
    background-color: #141414;
    padding: 10px;
    border-radius: 10px;
    transition: transform 0.3s;
    text-align: center;
}
.card:hover {
    transform: scale(1.05);
}
.title {
    font-size: 16px;
    font-weight: bold;
}
.rating {
    color: #f5c518;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Netflix Style Movie Predictor")
st.write("Find movies based on Genre & Year")

# =========================
# 🎯 USER INPUT
# =========================
all_genres = sorted(set(" ".join(movies['genres']).split()))

selected_genres = st.multiselect("Select Genre(s)", all_genres)
year = st.slider("Select Year", 1980, 2020, 2015)

# =========================
# 🔮 PREDICTION + FILTER
# =========================
if st.button("Show Movies"):
    if not selected_genres:
        st.warning("Please select at least one genre")
    else:
        input_genre = " ".join(selected_genres)
        genre_vec = cv.transform([input_genre]).toarray()
        year_vec = np.array([[year]])

        input_data = np.concatenate((genre_vec, year_vec), axis=1)

        # Predict base rating
        predicted_rating = model.predict(input_data)[0]

        # Filter movies (same genre + nearby year)
        filtered = movies[
            (movies['genres'].str.contains(selected_genres[0], case=False)) &
            (abs(movies['year'] - year) <= 3)
        ]

        filtered = filtered.copy()

        # Predict rating for filtered movies
        genre_vec_all = cv.transform(filtered['genres']).toarray()
        year_vec_all = filtered['year'].values.reshape(-1,1)
        X_all = np.concatenate((genre_vec_all, year_vec_all), axis=1)

        filtered['predicted_rating'] = model.predict(X_all)

        # Top movies
        top_movies = filtered.sort_values(by='predicted_rating', ascending=False).head(10)

        st.subheader("🔥 Recommended Movies")

        cols = st.columns(5)

        for i, row in top_movies.iterrows():
            with cols[i % 5]:
                st.markdown(f"""
                <div class="card">
                    <div class="title">{row['title']}</div>
                    <div class="rating">⭐ {round(row['predicted_rating'],2)}</div>
                    <div>📅 {row['year']}</div>
                </div>
                """, unsafe_allow_html=True)