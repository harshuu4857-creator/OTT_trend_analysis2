# =====================================================
# PROFESSIONAL FILTER SECTION
# REPLACE YOUR CURRENT FILTER SECTION WITH THIS
# =====================================================

st.markdown("""
<style>

/* FILTER CONTAINER */

.pro-filter-box{
    background: linear-gradient(
        145deg,
        rgba(18,18,18,0.95),
        rgba(10,10,10,0.92)
    );
    padding:28px;
    border-radius:24px;
    border:1px solid rgba(255,255,255,0.06);
    box-shadow:
        0 0 25px rgba(0,0,0,0.45);
    margin-bottom:35px;
}

/* SECTION LABEL */

.filter-heading{
    font-size:28px;
    font-weight:800;
    margin-bottom:8px;
    color:white;
}

.filter-sub{
    color:#b8b8b8;
    margin-bottom:28px;
    font-size:15px;
}

/* INPUT LABELS */

label{
    font-weight:600 !important;
    color:white !important;
    font-size:15px !important;
}

/* MULTISELECT */

.stMultiSelect div[data-baseweb="select"]{
    background:#171717 !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-radius:16px !important;
    min-height:58px !important;
}

/* TEXT INPUT */

.stTextInput input{
    background:#171717 !important;
    border-radius:16px !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    color:white !important;
    height:58px !important;
    font-size:16px !important;
}

/* SLIDER */

.stSlider{
    padding-top:12px;
}

/* BUTTON */

.stButton > button{

    width:100%;
    height:60px;

    border:none;
    border-radius:18px;

    background:linear-gradient(
        90deg,
        #E50914,
        #ff4d4d
    );

    color:white;
    font-size:18px;
    font-weight:700;

    transition:0.3s;
}

.stButton > button:hover{
    transform:scale(1.02);
    box-shadow:0 0 25px rgba(229,9,20,0.45);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# FILTER UI
# =====================================================

st.markdown("""
<div class="pro-filter-box">

<div class="filter-heading">
🎯 Smart Movie Discovery
</div>

<div class="filter-sub">
Discover premium movies using AI-powered OTT trend analysis
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# FILTER COLUMNS
# =====================================================

col1, col2, col3, col4 = st.columns([2.3,1,1,1])

# =====================================================
# GENRES
# =====================================================

with col1:

    selected_genres = st.multiselect(
        "🎭 Genres",
        all_genres,
        placeholder="Choose movie genres..."
    )

# =====================================================
# YEAR
# =====================================================

with col2:

    year = st.slider(
        "📅 Release Year",
        1980,
        2020,
        2015
    )

# =====================================================
# SEARCH
# =====================================================

with col3:

    search = st.text_input(
        "🔍 Movie Search",
        placeholder="Interstellar..."
    )

# =====================================================
# RATING
# =====================================================

with col4:

    min_rating = st.slider(
        "⭐ Minimum Rating",
        0.0,
        10.0,
        7.0,
        0.1
    )

# =====================================================
# BUTTON
# =====================================================

discover = st.button("🚀 Discover Premium Movies")