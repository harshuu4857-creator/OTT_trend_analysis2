import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import joblib
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="OTT Trend Analysis",
    layout="wide",
    page_icon="🎬"
)

# =====================================================
# LOAD MODEL
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
    background-color: #050816;
    color: white;
    font-family: sans-serif;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #0f0f0f;
    border-right: 1px solid #1f1f1f;
}

.sidebar-title {
    font-size:30px;
    font-weight:900;
    color:#E50914;
    margin-bottom:20px;
}

/* MAIN */

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
    background:#0f0f0f;
    padding:20px 28px;
    border-radius:20px;
    margin-bottom:30px;
}

.logo-section {
    display:flex;
    flex-direction:column;
}

.logo-title {
    font-size:42px;
    font-weight:900;
    color:#E50914;
}

.logo-sub {
    color:#bdbdbd;
    margin-top:8px;
    font-size:15px;
}

.stats {
    display:flex;
    gap:18px;
}

.stat-box {
    background:#1a1a1a;
    padding:14px 24px;
    border-radius:14px;
    text-align:center;
    min-width:120px;
}

.stat-number {
    font-size:28px;
    font-weight:bold;
}

.stat-label {
    color:#aaaaaa;
    font-size:14px;
}

/* FILTERS */

.filter-box {
    background:#111111;
    padding:20px;
    border-radius:20px;
    margin-bottom:25px;
}

/* HERO */

.hero-container {
    position: relative;
    height: 620px;
    border-radius: 30px;
    overflow: hidden;
    margin-top: 20px;
    margin-bottom: 40px;
    background-size: cover;
    background-position: center;
}

.hero-overlay {
    position:absolute;
    inset:0;
    background:linear-gradient(
    to right,
    rgba(0,0,0,0.96),
    rgba(0,0,0,0.2)
    );
}

.hero-content {
    position:absolute;
    bottom:60px;
    left:60px;
    width:40%;
    z-index:2;
}

.hero-title {
    font-size:72px;
    font-weight:900;
    line-height:1;
    margin-bottom:20px;
}

.hero-rating {
    color:#E50914;
    font-size:32px;
    font-weight:bold;
    margin-bottom:15px;
}

.hero-desc {
    color:#d9d9d9;
    font-size:18px;
    line-height:1.7;
}

/* SECTION */

.section-title {
    font-size:34px;
    font-weight:800;
    margin-top:20px;
    margin-bottom:20px;
}

/* MOVIE CARDS */

.movie-card {
    background:#141414;
    padding:12px;
    border-radius:18px;
    transition:0.4s;
    margin-bottom:20px;
}

.movie-card:hover {
    transform:scale(1.05);
}

.movie-name {
    font-size:18px;
    font-weight:700;
    margin-top:10px;
}

.movie-info {
    color:#d1d1d1;
    font-size:14px;
    margin-top:4px;
}

/* BUTTON */

.stButton > button {
    background:linear-gradient(90deg,#E50914,#ff4d4d);
    color:white;
    border:none;
    border-radius:14px;
    height:52px;
    width:220px;
    font-size:18px;
    font-weight:bold;
}

/* INPUTS */

.stTextInput input {
    background:#181818;
    color:white;
    border-radius:12px;
}

.stMultiSelect div {
    background:#181818;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        "<div class='sidebar-title'>OTT AI</div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🎯 Prediction",
            "🎬 All Movies"
        ]
    )

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

<div class="logo-section">

<div class="logo-title">
OTT Trend Analysis
</div>

<div class="logo-sub">
Created by Harsh Patel
</div>

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
# PREDICTION PAGE
# =====================================================

if page == "🎯 Prediction":

    all_genres = sorted(
        set(
            " ".join(
                movies['genres'].apply(
                    lambda x:" ".join(x)
                )
            ).split()
        )
    )

    st.markdown("<div class='filter-box'>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2.2,1,1,1])

    with col1:

        selected_genres = st.multiselect(
            "🎭 Select Genres",
            all_genres
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
            "🔍 Search",
            placeholder="Search Movie"
        )

    with col4:

        min_rating = st.slider(
            "⭐ Rating",
            0.0,
            10.0,
            7.0,
            0.1
        )

    discover = st.button("🎬 Discover Movies")

    st.markdown("</div>", unsafe_allow_html=True)

    default_movie = movies.sort_values(
        by='vote_average',
        ascending=False
    ).iloc[0]

    if discover and selected_genres:

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
            &
            (
                movies['vote_average'] >= min_rating
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

    # HERO
    hero_poster = fetch_poster(hero_movie['title'])

    hero_html = f"""
    <div class="hero-container"
    style="
    background-image:
    linear-gradient(
    to right,
    rgba(0,0,0,0.96),
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
    AI powered OTT trend analysis and intelligent movie recommendation system using machine learning and audience trends.
    </div>

    </div>

    </div>
    """

    st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>🔥 Trending & Recommended Movies</div>",
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

# =====================================================
# ALL MOVIES PAGE
# =====================================================

elif page == "🎬 All Movies":

    st.markdown(
        "<div class='section-title'>🎬 OTT Dataset Analytics Dashboard</div>",
        unsafe_allow_html=True
    )

    # KPI ROW

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "🎬 Total Movies",
            len(movies)
        )

    with k2:
        st.metric(
            "⭐ Average Rating",
            round(movies['vote_average'].mean(), 2)
        )

    with k3:
        st.metric(
            "🔥 Avg Popularity",
            round(movies['popularity'].mean(), 2)
        )

    with k4:
        st.metric(
            "🌍 Languages",
            movies['original_language'].nunique()
        )

    st.divider()

    # GRAPH 1

    lang_df = (
        movies['original_language']
        .value_counts()
        .head(10)
        .reset_index()
    )

    lang_df.columns = ['Language', 'Movies']

    fig1 = px.bar(
        lang_df,
        x='Language',
        y='Movies',
        title='Top Languages in Dataset'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # GRAPH 2

    year_df = (
        movies['year']
        .value_counts()
        .sort_index()
        .reset_index()
    )

    year_df.columns = ['Year', 'Movies']

    fig2 = px.line(
        year_df,
        x='Year',
        y='Movies',
        title='Movies Released Per Year'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # GRAPH 3

    fig3 = px.histogram(
        movies,
        x='vote_average',
        nbins=20,
        title='Movie Rating Distribution'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # GRAPH 4

    fig4 = px.scatter(
        movies,
        x='popularity',
        y='vote_average',
        hover_name='title',
        title='Popularity vs Rating'
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # GRAPH 5

    top_budget = movies.sort_values(
        by='budget',
        ascending=False
    ).head(10)

    fig5 = px.bar(
        top_budget,
        x='title',
        y='budget',
        title='Top 10 Highest Budget Movies'
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # DATAFRAME

    st.markdown(
        "<div class='section-title'>📊 Complete Movie Dataset</div>",
        unsafe_allow_html=True
    )

    display_movies = movies[
        [
            'title',
            'vote_average',
            'popularity',
            'original_language',
            'budget',
            'year'
        ]
    ].sort_values(
        by='vote_average',
        ascending=False
    )

    st.dataframe(
        display_movies,
        use_container_width=True,
        height=700
    )