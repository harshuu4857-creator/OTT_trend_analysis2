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

movies = movies[['genres', 'release_date', 'vote_average']]
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
# 🌐 STREAMLIT UI
# =========================
st.set_page_config(page_title="Movie Rating Predictor", layout="centered")

st.title("🎬 Movie Rating Prediction App")
st.write("Predict movie rating using Genre + Year")

# =========================
# 🎯 USER INPUT
# =========================
all_genres = sorted(set(" ".join(movies['genres']).split()))

selected_genres = st.multiselect("Select Genre(s)", all_genres)

year = st.slider("Select Year", 1980, 2025, 2020)

# =========================
# 🔮 PREDICTION
# =========================
if st.button("Predict Rating"):
    if not selected_genres:
        st.warning("Please select at least one genre")
    else:
        input_genre = " ".join(selected_genres)
        genre_vec = cv.transform([input_genre]).toarray()

        year_vec = np.array([[year]])

        input_data = np.concatenate((genre_vec, year_vec), axis=1)

        prediction = model.predict(input_data)[0]

        st.success(f"⭐ Predicted Rating: {round(prediction, 2)}")