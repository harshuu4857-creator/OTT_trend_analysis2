# =====================================================
# ALL MOVIES PAGE
# =====================================================

if page == "🎬 All Movies":

    import plotly.express as px

    st.markdown(
        """
        <div class='section-title'>
        🎬 OTT Dataset Analytics Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

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

    # =====================================================
    # GRAPH 1
    # TOP LANGUAGES
    # =====================================================

    lang_df = (
        movies['original_language']
        .value_counts()
        .head(10)
        .reset_index()
    )

    lang_df.columns = [
        'Language',
        'Movies'
    ]

    fig1 = px.bar(
        lang_df,
        x='Language',
        y='Movies',
        title='Top Languages in OTT Dataset'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================================
    # GRAPH 2
    # MOVIES RELEASED BY YEAR
    # =====================================================

    year_df = (
        movies['year']
        .value_counts()
        .sort_index()
        .reset_index()
    )

    year_df.columns = [
        'Year',
        'Movies'
    ]

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

    # =====================================================
    # GRAPH 3
    # RATING DISTRIBUTION
    # =====================================================

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

    # =====================================================
    # GRAPH 4
    # POPULARITY VS RATING
    # =====================================================

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

    # =====================================================
    # GRAPH 5
    # TOP BUDGET MOVIES
    # =====================================================

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

    # =====================================================
    # DATAFRAME
    # =====================================================

    st.markdown(
        """
        <div class='section-title'>
        📊 Complete Movie Dataset
        </div>
        """,
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