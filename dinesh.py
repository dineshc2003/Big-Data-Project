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

# 1. Data Preprocessing and Cleaning
@st.cache_data
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 1.a Drop rows missing key identifiers
    df = df.dropna(subset=['id', 'title'])

    # 1.b Parse release_date
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df[df['release_date'].notna()]

    # 1.c Numeric columns – coerce & median-impute
    num_cols = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())

    # 1.d Data cleaning: remove rows with zero votes/counts or zero budget/revenue
    df = df[(df['vote_average'] > 0) & (df['vote_count'] > 0) &
            (df['budget'] > 0) & (df['revenue'] > 0)]

    # 1.e Keep only released movies
    df = df[df['status'] == 'Released']

    # 1.f Adult flag
    df['adult'] = df['adult'].astype(bool)

    # 1.g Split list-like columns into tuples (for caching)
    list_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']
    for c in list_cols:
        df[c] = df[c].fillna('').apply(lambda x: tuple(i.strip() for i in x.split(',')) if x else tuple())

    # 1.h Extract year
    df['year'] = df['release_date'].dt.year

    return df

# 2. Extended Analyses & Metrics
@st.cache_data
def compute_numeric_corr(df: pd.DataFrame) -> pd.DataFrame:
    nums = ['vote_average', 'runtime', 'adult', 'popularity']
    corr = df[nums].corr()
    return corr.fillna(0)

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
    freq = exploded['production_companies'].value_counts().rename_axis('company').reset_index(name='appearances')
    return freq

@st.cache_data
def genre_popularity_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['decade'] = (df_copy['year'] // 10) * 10
    exploded = df_copy.explode('genres').dropna(subset=['genres'])
    grp = exploded.groupby(['decade', 'genres'])['popularity'].mean().reset_index()
    idx = grp.groupby('decade')['popularity'].idxmax()
    return grp.loc[idx].sort_values('decade')

# 3. Visualization Helpers

def create_budget_revenue_scatter(df: pd.DataFrame, log_x: bool, log_y: bool, size_max: int, color_metric: str):
    fig = px.scatter(
        df,
        x='budget',
        y='revenue',
        size='popularity',
        color=color_metric,
        hover_data=['title', 'year', 'vote_average', 'popularity'],
        title='Budget vs Revenue',
        size_max=size_max,
        color_continuous_scale='Viridis',
        template='plotly_white',
        log_x=log_x,
        log_y=log_y,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis=dict(title='Budget (USD)', tickprefix='$', ticks='outside'),
        yaxis=dict(title='Revenue (USD)', tickprefix='$', ticks='outside'),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def create_genre_popularity_bar(df: pd.DataFrame, top_n: int = 10):
    exploded = df.explode('genres')
    grp = exploded.groupby('genres')['popularity'].mean().nlargest(top_n).reset_index()
    return px.bar(
        grp,
        x='genres',
        y='popularity',
        title=f'Top {top_n} Genres by Avg Popularity',
        template='plotly_white'
    )


def create_popularity_over_time(df: pd.DataFrame):
    yr = df.groupby('year')['popularity'].mean().reset_index()
    return px.line(
        yr,
        x='year',
        y='popularity',
        title='Avg Movie Popularity Over Years',
        template='plotly_white'
    )


def create_top_movies_table(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.nlargest(n, 'revenue')[['title', 'year', 'revenue', 'vote_average', 'popularity']]

# -- Main App --
st.set_page_config(page_title='Cinema Trends Dashboard', layout='wide')
st.title('Cinema Trends Dashboard')

# Load & preprocess data
df = load_and_preprocess(DATA_PATH)

# Sidebar filters
st.sidebar.header('Filters')
all_genres = sorted({g for genres in df['genres'] for g in genres})
selected_genres = st.sidebar.multiselect('Select Genres', all_genres)

# Year range: discrete selection
current_year = datetime.now().year
min_year = int(df['year'].min())
max_year = min(int(df['year'].max()), current_year)
years = list(range(min_year, max_year + 1))
with st.sidebar.expander('Year Range', expanded=True):
    selected_years = st.sidebar.select_slider(
        'Select Year Range',
        options=years,
        value=(min_year, max_year),
        format_func=lambda y: str(y),
        help='Use the handles to set the start and end year'
    )

# Apply year + genre filters to dff
filtered = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
if selected_genres:
    dff = filtered[filtered['genres'].apply(lambda lst: any(g in lst for g in selected_genres))]
else:
    dff = filtered

# Create tabs
tabs = st.tabs([
    'Budget vs Revenue', 'Genre Popularity', 'Popularity Over Time',
    'Top Movies', 'Numeric Correlation', 'List-Column Counts',
    'Genre Success Metrics', 'Top-1000 Company Freq', 'Top Genre by Decade'
])

# Budget vs Revenue tab
with tabs[0]:
    st.subheader('Budget vs Revenue')
    log_x = st.checkbox('Log scale budget axis', key='bv_log_x')
    log_y = st.checkbox('Log scale revenue axis', key='bv_log_y')
    size_max = st.slider('Max marker size', 10, 100, 40, key='bv_size_max')
    color_metric = st.selectbox('Color by', ['vote_average', 'popularity'], index=0, key='bv_color')
    fig_br = create_budget_revenue_scatter(dff, log_x, log_y, size_max, color_metric)
    st.plotly_chart(fig_br, use_container_width=True)

# Genre Popularity (ignores genre multiselect, only applies year filter)
with tabs[1]:
    st.subheader('Top Genres by Avg Popularity')
    # use only year-filtered data for this chart
    fig_gp = create_genre_popularity_bar(filtered)
    st.plotly_chart(fig_gp, use_container_width=True)

# Popularity Over Time
with tabs[2]:
    st.plotly_chart(create_popularity_over_time(dff), use_container_width=True)

# Top Movies Table
with tabs[3]:
    st.dataframe(create_top_movies_table(dff), use_container_width=True)

# Numeric Correlation
with tabs[4]:
    corr = compute_numeric_corr(dff)
    st.write('### Numeric Correlation Matrix')
    st.dataframe(corr.style.background_gradient(), use_container_width=True)

# List-Column Counts
with tabs[5]:
    for col in ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']:
        st.write(f"#### Top 10 {col.replace('_', ' ').title()}")
        counts = list_column_counts(dff, col).head(10)
        st.bar_chart(counts.set_index(col)['count'])

# Genre Success Metrics
with tabs[6]:
    metrics = genre_success_metrics(dff)
    st.write('### Revenue, Rating & Popularity by Genre')
    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics['avg_revenue'])

# Top-1000 Company Frequency
with tabs[7]:
    freq = top1000_company_freq(dff)
    st.write('### Production Company Frequencies in Top 1,000 Revenue Movies')
    st.dataframe(freq.head(20), use_container_width=True)
    st.bar_chart(freq.set_index('company')['appearances'].head(20))

# Top Genre by Decade
with tabs[8]:
    top_decade = genre_popularity_by_decade(dff)
    st.write('### Most Popular Genre by Decade (Avg. Popularity)')
    st.table(top_decade.set_index('decade'))

# Footer
st.markdown(
    '*Project:* The Rise and Fall of Genres – CS661 Visual Analytics | *Group 20*'
)
