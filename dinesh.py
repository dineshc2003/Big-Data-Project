# streamlit_app.py
# Group 20: The Rise and Fall of Genres – CS661 Visual Analytics (Streamlit Version)

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# 0. Configuration: dataset path
DATA_PATH = r'C:\Users\kinshuk dutt\Downloads\Big data\TMDB_movie_dataset_v11 (1).csv'


# Ensure the dataset exists
if not os.path.exists(DATA_PATH):
    st.error(f"CSV file not found at {DATA_PATH}")
    st.stop()

# 1. Data Preprocessing
@st.cache_data
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 1.a Drop rows missing key identifiers before we remove 'id'
    df = df.dropna(subset=['id', 'title'])

    # 1.b Drop unwanted columns
    to_drop = [
        'id',                # A
        'status',            # E
        'backdrop_path',     # J
        'homepage',          # L
        'imdb_id',           # M
        'original_language', # N
        'original_title',    # O
        'overview',          # P
        'poster_path',       # R
        'tagline'            # S
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    # 1.c Parse release_date
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df[df['release_date'].notna()]

    # 1.d Numeric columns – coerce & median impute
    num_cols = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())

    # 1.e Adult flag
    df['adult'] = df['adult'].astype(bool)

    # 1.f Split list-like columns
    list_cols = [
        'genres',
        'production_companies',
        'production_countries',
        'spoken_languages',
        'keywords'
    ]
    for c in list_cols:
        df[c] = df[c].fillna('').apply(
            lambda x: [i.strip() for i in x.split(',')] if x else []
        )

    # 1.g Extract year
    df['year'] = df['release_date'].dt.year

    return df

# 2. Extended Analyses & Metrics
@st.cache_data
def compute_numeric_corr(df: pd.DataFrame) -> pd.DataFrame:
    nums = ['vote_average', 'runtime', 'adult', 'popularity']
    return df[nums].corr()

@st.cache_data
def list_column_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    exploded = df.explode(col)
    counts = exploded[col].value_counts().rename_axis(col).reset_index(name='count')
    return counts

@st.cache_data
def genre_success_metrics(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df.explode('genres').dropna(subset=['genres'])
    grp = exploded.groupby('genres').agg(
        avg_revenue=('revenue', 'mean'),
        avg_rating=('vote_average', 'mean'),
        avg_popularity=('popularity', 'mean'),
        count=('title', 'count')
    ).sort_values('avg_revenue', ascending=False)
    return grp

@st.cache_data
def top1000_company_freq(df: pd.DataFrame) -> pd.DataFrame:
    top1000 = df.nlargest(1000, 'revenue')
    exploded = top1000.explode('production_companies').dropna(subset=['production_companies'])
    freq = (
        exploded['production_companies']
        .value_counts()
        .rename_axis('company')
        .reset_index(name='appearances')
    )
    return freq

@st.cache_data
def genre_popularity_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['decade'] = (df['year'] // 10) * 10
    exploded = df.explode('genres').dropna(subset=['genres'])
    grp = exploded.groupby(['decade', 'genres'])['popularity'].mean().reset_index()
    idx = grp.groupby('decade')['popularity'].idxmax()
    top_by_decade = grp.loc[idx].sort_values('decade')
    return top_by_decade

# 3. Original Visualization Helpers
def create_budget_revenue_scatter(df: pd.DataFrame):
    return px.scatter(
        df,
        x='budget',
        y='revenue',
        size='popularity',
        color='vote_average',
        hover_data=['title'],
        title='Budget vs Revenue (size=popularity, color=avg rating)'
    )

def create_genre_popularity_bar(df: pd.DataFrame, top_n: int = 10):
    exploded = df.explode('genres')
    grp = exploded.groupby('genres')['popularity'].mean().nlargest(top_n).reset_index()
    return px.bar(
        grp,
        x='genres',
        y='popularity',
        title=f'Top {top_n} Genres by Avg Popularity'
    )

def create_popularity_over_time(df: pd.DataFrame):
    yr = df.groupby('year')['popularity'].mean().reset_index()
    return px.line(
        yr,
        x='year',
        y='popularity',
        title='Avg Movie Popularity Over Years'
    )

def create_top_movies_table(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.nlargest(n, 'revenue')[[
        'title', 'year', 'revenue', 'vote_average', 'popularity'
    ]]

# -- Main App --
st.set_page_config(page_title='Cinema Trends Dashboard', layout='wide')
st.title('Cinema Trends Dashboard')

# Load & preprocess data
df = load_and_preprocess(DATA_PATH)

# Sidebar filters
st.sidebar.header('Filters')
all_genres = sorted({g for genres in df['genres'] for g in genres})
selected_genres = st.sidebar.multiselect('Select Genres', all_genres)
min_year, max_year = int(df['year'].min()), int(df['year'].max())
selected_years = st.sidebar.slider(
    'Select Year Range', min_year, max_year, (min_year, max_year)
)

# Apply filters
dff = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
if selected_genres:
    dff = dff[dff['genres'].apply(lambda lst: any(g in lst for g in selected_genres))]

# Create tabs for all visualizations
tabs = st.tabs([
    'Budget vs Revenue',
    'Genre Popularity',
    'Popularity Over Time',
    'Top Movies',
    'Numeric Correlation',
    'List-Column Counts',
    'Genre Success Metrics',
    'Top-1000 Company Freq',
    'Top Genre by Decade'
])

with tabs[0]:
    st.plotly_chart(create_budget_revenue_scatter(dff), use_container_width=True)

with tabs[1]:
    st.plotly_chart(create_genre_popularity_bar(dff), use_container_width=True)

with tabs[2]:
    st.plotly_chart(create_popularity_over_time(dff), use_container_width=True)

with tabs[3]:
    st.dataframe(create_top_movies_table(dff), use_container_width=True)

with tabs[4]:
    corr = compute_numeric_corr(dff)
    st.write("### Numeric Correlation Matrix")
    st.dataframe(corr.style.background_gradient(), use_container_width=True)

with tabs[5]:
    for col in [
        'genres',
        'production_companies',
        'production_countries',
        'spoken_languages',
        'keywords'
    ]:
        st.write(f"#### Top 10 {col.replace('_', ' ').title()}")
        counts = list_column_counts(dff, col).head(10)
        st.bar_chart(counts.set_index(col)['count'])

with tabs[6]:
    metrics = genre_success_metrics(dff)
    st.write("### Revenue, Rating & Popularity by Genre")
    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics['avg_revenue'])

with tabs[7]:
    freq = top1000_company_freq(dff)
    st.write("### Production Company Frequencies in Top 1,000 Revenue Movies")
    st.dataframe(freq.head(20), use_container_width=True)
    st.bar_chart(freq.set_index('company')['appearances'].head(20))

with tabs[8]:
    top_decade = genre_popularity_by_decade(dff)
    st.write("### Most Popular Genre by Decade (Avg. Popularity)")
    st.table(top_decade.set_index('decade'))

# Footer
st.markdown(
    '*Project:* The Rise and Fall of Genres – CS661 Visual Analytics | *Group 20*'
)