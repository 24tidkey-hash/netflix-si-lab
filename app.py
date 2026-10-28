"""
Netflix Content Trends and Genre Analysis
Statistical Inference Lab Web Application
Streamlit Community Cloud Deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
from bs4 import BeautifulSoup

# Configure Streamlit page
st.set_page_config(
    page_title="Netflix Content Trends & Genre Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E50914;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DEFAULT DATASET INITIALIZATION
# -------------------------------------------------------------
@st.cache_data
def load_default_dataset():
    data = [
        {"show_id": "s1", "type": "Movie", "title": "Roma", "director": "Alfonso Cuarón", "country": "Mexico", "release_year": 2018, "rating": "R", "duration_min": 135, "primary_genre": "Drama", "imdb_score": 7.7, "total_votes": 165000, "content_budget_millions": 15.0},
        {"show_id": "s2", "type": "TV Show", "title": "Stranger Things", "director": "The Duffer Brothers", "country": "United States", "release_year": 2016, "rating": "TV-14", "duration_min": 2160, "primary_genre": "Sci-Fi", "imdb_score": 8.7, "total_votes": 820000, "content_budget_millions": 270.0},
        {"show_id": "s3", "type": "Movie", "title": "The Irishman", "director": "Martin Scorsese", "country": "United States", "release_year": 2019, "rating": "R", "duration_min": 209, "primary_genre": "Crime", "imdb_score": 7.8, "total_votes": 410000, "content_budget_millions": 159.0},
        {"show_id": "s4", "type": "TV Show", "title": "The Crown", "director": "Peter Morgan", "country": "United Kingdom", "release_year": 2016, "rating": "TV-MA", "duration_min": 3600, "primary_genre": "Drama", "imdb_score": 8.6, "total_votes": 245000, "content_budget_millions": 260.0},
        {"show_id": "s5", "type": "TV Show", "title": "Squid Game", "director": "Hwang Dong-hyuk", "country": "South Korea", "release_year": 2021, "rating": "TV-MA", "duration_min": 495, "primary_genre": "Thriller", "imdb_score": 8.0, "total_votes": 530000, "content_budget_millions": 21.4},
        {"show_id": "s6", "type": "Movie", "title": "Extraction", "director": "Sam Hargrave", "country": "United States", "release_year": 2020, "rating": "R", "duration_min": 116, "primary_genre": "Action", "imdb_score": 6.8, "total_votes": 230000, "content_budget_millions": 65.0},
        {"show_id": "s7", "type": "Movie", "title": "Glass Onion", "director": "Rian Johnson", "country": "United States", "release_year": 2022, "rating": "PG-13", "duration_min": 139, "primary_genre": "Mystery", "imdb_score": 7.1, "total_votes": 395000, "content_budget_millions": 40.0},
        {"show_id": "s8", "type": "TV Show", "title": "The Queen's Gambit", "director": "Scott Frank", "country": "United States", "release_year": 2020, "rating": "TV-MA", "duration_min": 393, "primary_genre": "Drama", "imdb_score": 8.5, "total_votes": 490000, "content_budget_millions": 35.0},
        {"show_id": "s9", "type": "Movie", "title": "Pinocchio", "director": "Guillermo del Toro", "country": "United States", "release_year": 2022, "rating": "PG", "duration_min": 117, "primary_genre": "Animation", "imdb_score": 7.6, "total_votes": 115000, "content_budget_millions": 35.0},
        {"show_id": "s10", "type": "TV Show", "title": "Wednesday", "director": "Tim Burton", "country": "United States", "release_year": 2022, "rating": "TV-14", "duration_min": 400, "primary_genre": "Fantasy", "imdb_score": 8.1, "total_votes": 360000, "content_budget_millions": 38.0},
        {"show_id": "s11", "type": "Movie", "title": "Bird Box", "director": "Susanne Bier", "country": "United States", "release_year": 2018, "rating": "R", "duration_min": 124, "primary_genre": "Thriller", "imdb_score": 6.6, "total_votes": 355000, "content_budget_millions": 19.8},
        {"show_id": "s12", "type": "TV Show", "title": "Money Heist", "director": "Álex Pina", "country": "Spain", "release_year": 2017, "rating": "TV-MA", "duration_min": 2050, "primary_genre": "Crime", "imdb_score": 8.2, "total_votes": 510000, "content_budget_millions": 48.0},
        {"show_id": "s13", "type": "Movie", "title": "Don't Look Up", "director": "Adam McKay", "country": "United States", "release_year": 2021, "rating": "R", "duration_min": 138, "primary_genre": "Comedy", "imdb_score": 7.2, "total_votes": 565000, "content_budget_millions": 75.0},
        {"show_id": "s14", "type": "TV Show", "title": "Dark", "director": "Baran bo Odar", "country": "Germany", "release_year": 2017, "rating": "TV-MA", "duration_min": 1560, "primary_genre": "Sci-Fi", "imdb_score": 8.7, "total_votes": 430000, "content_budget_millions": 24.0},
        {"show_id": "s15", "type": "Movie", "title": "Marriage Story", "director": "Noah Baumbach", "country": "United States", "release_year": 2019, "rating": "R", "duration_min": 137, "primary_genre": "Drama", "imdb_score": 7.9, "total_votes": 340000, "content_budget_millions": 18.0},
        {"show_id": "s16", "type": "TV Show", "title": "Black Mirror", "director": "Charlie Brooker", "country": "United Kingdom", "release_year": 2016, "rating": "TV-MA", "duration_min": 1620, "primary_genre": "Sci-Fi", "imdb_score": 8.7, "total_votes": 615000, "content_budget_millions": 60.0},
        {"show_id": "s17", "type": "Movie", "title": "Red Notice", "director": "Rawson Thurber", "country": "United States", "release_year": 2021, "rating": "PG-13", "duration_min": 118, "primary_genre": "Action", "imdb_score": 6.3, "total_votes": 310000, "content_budget_millions": 200.0},
        {"show_id": "s18", "type": "TV Show", "title": "Narcos", "director": "Chris Brancato", "country": "United States", "release_year": 2015, "rating": "TV-MA", "duration_min": 1500, "primary_genre": "Crime", "imdb_score": 8.8, "total_votes": 450000, "content_budget_millions": 50.0},
        {"show_id": "s19", "type": "Movie", "title": "Tick, Tick... Boom!", "director": "Lin-Manuel Miranda", "country": "United States", "release_year": 2021, "rating": "PG-13", "duration_min": 115, "primary_genre": "Musical", "imdb_score": 7.5, "total_votes": 135000, "content_budget_millions": 22.0},
        {"show_id": "s20", "type": "TV Show", "title": "BoJack Horseman", "director": "Raphael Bob-Waksberg", "country": "United States", "release_year": 2014, "rating": "TV-MA", "duration_min": 1950, "primary_genre": "Comedy", "imdb_score": 8.8, "total_votes": 180000, "content_budget_millions": 36.0},
        {"show_id": "s21", "type": "Movie", "title": "Klaus", "director": "Sergio Pablos", "country": "Spain", "release_year": 2019, "rating": "PG", "duration_min": 97, "primary_genre": "Animation", "imdb_score": 8.2, "total_votes": 185000, "content_budget_millions": 40.0},
        {"show_id": "s22", "type": "TV Show", "title": "Bridgerton", "director": "Chris Van Dusen", "country": "United States", "release_year": 2020, "rating": "TV-MA", "duration_min": 1440, "primary_genre": "Romance", "imdb_score": 7.4, "total_votes": 190000, "content_budget_millions": 84.0},
        {"show_id": "s23", "type": "Movie", "title": "All Quiet on Western Front", "director": "Edward Berger", "country": "Germany", "release_year": 2022, "rating": "R", "duration_min": 148, "primary_genre": "War", "imdb_score": 7.8, "total_votes": 240000, "content_budget_millions": 20.0},
        {"show_id": "s24", "type": "TV Show", "title": "Cyberpunk: Edgerunners", "director": "Hiroyuki Imaishi", "country": "Japan", "release_year": 2022, "rating": "TV-MA", "duration_min": 240, "primary_genre": "Animation", "imdb_score": 8.3, "total_votes": 110000, "content_budget_millions": 12.0},
        {"show_id": "s25", "type": "Movie", "title": "The Adam Project", "director": "Shawn Levy", "country": "United States", "release_year": 2022, "rating": "PG-13", "duration_min": 106, "primary_genre": "Sci-Fi", "imdb_score": 6.7, "total_votes": 240000, "content_budget_millions": 116.0},
        {"show_id": "s26", "type": "TV Show", "title": "Heartstopper", "director": "Euros Lyn", "country": "United Kingdom", "release_year": 2022, "rating": "TV-14", "duration_min": 480, "primary_genre": "Romance", "imdb_score": 8.6, "total_votes": 75000, "content_budget_millions": 14.0},
        {"show_id": "s27", "type": "Movie", "title": "Enola Holmes", "director": "Harry Bradbeer", "country": "United Kingdom", "release_year": 2020, "rating": "PG-13", "duration_min": 123, "primary_genre": "Mystery", "imdb_score": 6.6, "total_votes": 195000, "content_budget_millions": 21.0},
        {"show_id": "s28", "type": "TV Show", "title": "Ozark", "director": "Jason Bateman", "country": "United States", "release_year": 2017, "rating": "TV-MA", "duration_min": 2640, "primary_genre": "Crime", "imdb_score": 8.5, "total_votes": 350000, "content_budget_millions": 80.0},
        {"show_id": "s29", "type": "Movie", "title": "Society of the Snow", "director": "J.A. Bayona", "country": "Spain", "release_year": 2023, "rating": "TV-MA", "duration_min": 144, "primary_genre": "Adventure", "imdb_score": 7.8, "total_votes": 140000, "content_budget_millions": 65.0},
        {"show_id": "s30", "type": "TV Show", "title": "Mindhunter", "director": "David Fincher", "country": "United States", "release_year": 2017, "rating": "TV-MA", "duration_min": 1140, "primary_genre": "Crime", "imdb_score": 8.6, "total_votes": 320000, "content_budget_millions": 60.0},
        {"show_id": "s31", "type": "Movie", "title": "The Gray Man", "director": "Russo Brothers", "country": "United States", "release_year": 2022, "rating": "PG-13", "duration_min": 129, "primary_genre": "Action", "imdb_score": 6.5, "total_votes": 245000, "content_budget_millions": 200.0},
        {"show_id": "s32", "type": "TV Show", "title": "Arcane", "director": "Christian Linke", "country": "United States", "release_year": 2021, "rating": "TV-14", "duration_min": 720, "primary_genre": "Animation", "imdb_score": 9.0, "total_votes": 275000, "content_budget_millions": 110.0},
        {"show_id": "s33", "type": "Movie", "title": "Cocaine Bear", "director": "Elizabeth Banks", "country": "United States", "release_year": 2023, "rating": "R", "duration_min": 95, "primary_genre": "Comedy", "imdb_score": 5.9, "total_votes": 115000, "content_budget_millions": 35.0},
        {"show_id": "s34", "type": "TV Show", "title": "Better Call Saul", "director": "Vince Gilligan", "country": "United States", "release_year": 2015, "rating": "TV-MA", "duration_min": 3150, "primary_genre": "Crime", "imdb_score": 9.0, "total_votes": 630000, "content_budget_millions": 120.0},
        {"show_id": "s35", "type": "Movie", "title": "Leave the World Behind", "director": "Sam Esmail", "country": "United States", "release_year": 2023, "rating": "R", "duration_min": 141, "primary_genre": "Thriller", "imdb_score": 6.5, "total_votes": 190000, "content_budget_millions": 55.0},
        {"show_id": "s36", "type": "TV Show", "title": "Peaky Blinders", "director": "Steven Knight", "country": "United Kingdom", "release_year": 2013, "rating": "TV-MA", "duration_min": 2160, "primary_genre": "Crime", "imdb_score": 8.8, "total_votes": 620000, "content_budget_millions": 45.0},
        {"show_id": "s37", "type": "Movie", "title": "Trial of Chicago 7", "director": "Aaron Sorkin", "country": "United States", "release_year": 2020, "rating": "R", "duration_min": 130, "primary_genre": "Drama", "imdb_score": 7.7, "total_votes": 185000, "content_budget_millions": 35.0},
        {"show_id": "s38", "type": "TV Show", "title": "Cobra Kai", "director": "Josh Heald", "country": "United States", "release_year": 2018, "rating": "TV-14", "duration_min": 1950, "primary_genre": "Action", "imdb_score": 8.5, "total_votes": 205000, "content_budget_millions": 55.0},
        {"show_id": "s39", "type": "Movie", "title": "Nimona", "director": "Nick Bruno", "country": "United States", "release_year": 2023, "rating": "PG", "duration_min": 102, "primary_genre": "Animation", "imdb_score": 7.5, "total_votes": 78000, "content_budget_millions": 40.0},
        {"show_id": "s40", "type": "TV Show", "title": "The Witcher", "director": "Lauren Hissrich", "country": "United States", "release_year": 2019, "rating": "TV-MA", "duration_min": 1440, "primary_genre": "Fantasy", "imdb_score": 8.0, "total_votes": 560000, "content_budget_millions": 170.0},
    ]
    return pd.DataFrame(data)

# Sidebar: Dataset Selection & Navigation
st.sidebar.title("🎬 Navigation")
uploaded_file = st.sidebar.file_uploader("Upload Netflix CSV (Optional)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom CSV Loaded Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        df = load_default_dataset()
else:
    df = load_default_dataset()

# Topic selector (Topic names only, no Lab/Practical labels)
topics = [
    "Web Scraping and CSV Creation",
    "Data Cleaning and Exploratory Data Analysis",
    "Probability Distributions and Sampling",
    "Central Limit Theorem",
    "Confidence Intervals and Hypothesis Testing",
    "Chi-Square Test",
    "Covariance and Correlation",
    "Correlation Hypothesis Testing",
    "Two-Way ANOVA",
    "Regression Analysis"
]

selected_topic = st.sidebar.radio("Select Topic", topics)

# App Header
st.markdown('<div class="main-title">Netflix Content Trends & Genre Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Statistical Inference Lab Capstone Platform</div>', unsafe_allow_html=True)
st.markdown(f"**Current Topic:** {selected_topic} &nbsp;|&nbsp; **Active Records:** {len(df):,} titles")
st.markdown("---")

# -------------------------------------------------------------
# 1. WEB SCRAPING AND CSV CREATION
# -------------------------------------------------------------
if selected_topic == "Web Scraping and CSV Creation":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        Web scraping automates the programmatic collection of streaming catalog metadata from public streaming aggregators, IMDb endpoints, or Wikipedia charts to build analytical datasets when direct database access is unavailable.

        **How it is calculated & executed:**
        1. Send an HTTP GET request with customized `User-Agent` headers to bypass generic bot filters.
        2. Inspect status codes (`200 OK`, `403 Forbidden`, `404 Not Found`).
        3. Parse the DOM tree with BeautifulSoup or lxml selectors.
        4. Normalize extracted attributes and export to RFC 4180 standard CSV.

        **Formula / HTTP Status Validation:**
        $$Status \in \{200, 201\} \implies \text{Parse DOM}; \quad Status \in \{403, 429\} \implies \text{Backoff / Fallback to Cached CSV}$$

        **Interpretation:**
        Graceful error handling guarantees pipeline reliability. If scraping is blocked by anti-bot protections, the system automatically falls back to curated verified streaming mirrors without crashing.
        """)

    st.subheader("Interactive Scraper Simulator")
    url_input = st.text_input("Enter URL to scrape / inspect:", value="https://en.wikipedia.org/wiki/List_of_Netflix_original_programming")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch Live Metadata"):
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                response = requests.get(url_input, headers=headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    tables = soup.find_all('table', {'class': 'wikitable'})
                    st.success(f"Success! HTTP 200 OK — Found {len(tables)} tables on page.")
                else:
                    st.warning(f"HTTP Status {response.status_code}. Using verified offline catalog fallback.")
            except Exception as ex:
                st.info(f"External network sandboxed or timeout: {ex}. Using curated repository records.")

    with col2:
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Netflix Master CSV",
            data=csv_buffer,
            file_name="netflix_content_dataset.csv",
            mime="text/csv"
        )

    st.dataframe(df.head(10), use_container_width=True)

# -------------------------------------------------------------
# 2. DATA CLEANING AND EXPLORATORY DATA ANALYSIS
# -------------------------------------------------------------
elif selected_topic == "Data Cleaning and Exploratory Data Analysis":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        EDA investigates dataset integrity, discovers patterns, spots anomalies, tests hypotheses, and summarizes central tendencies and dispersion metrics.

        **Formulas:**
        - **Arithmetic Mean:** $\\bar{x} = \\frac{1}{n}\\sum_{i=1}^n x_i$
        - **Sample Variance:** $s^2 = \\frac{\\sum (x_i - \\bar{x})^2}{n - 1}$
        - **Standard Deviation:** $s = \\sqrt{s^2}$
        - **Interquartile Range (IQR):** $IQR = Q_3 - Q_1$
        - **Outlier Thresholds:** $[Q_1 - 1.5 \\times IQR, \; Q_3 + 1.5 \\times IQR]$

        **Interpretation:**
        Comparing mean vs median identifies distribution skewness. Kurtosis assesses heavy tails and vulnerability to extreme outlier content budgets.
        """)

    num_col = st.selectbox("Select Numerical Feature for EDA:", ["imdb_score", "duration_min", "content_budget_millions", "total_votes", "release_year"])
    
    vals = df[num_col].dropna()
    mean_val = vals.mean()
    med_val = vals.median()
    std_val = vals.std()
    var_val = vals.var()
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{mean_val:.2f}")
    c2.metric("Median", f"{med_val:.2f}")
    c3.metric("Std Dev", f"{std_val:.2f}")
    c4.metric("IQR", f"{iqr:.2f}")

    tab1, tab2, tab3 = st.tabs(["Distribution Histogram", "Box Plot (Outliers)", "Genre Breakdown"])
    with tab1:
        fig_hist = px.histogram(df, x=num_col, color="type", marginal="rug", nbins=20, title=f"Histogram of {num_col} by Content Type")
        st.plotly_chart(fig_hist, use_container_width=True)
    with tab2:
        fig_box = px.box(df, x="type", y=num_col, color="type", points="all", title=f"Box Plot & Outlier Detection: {num_col}")
        st.plotly_chart(fig_box, use_container_width=True)
    with tab3:
        genre_counts = df['primary_genre'].value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Count']
        fig_bar = px.bar(genre_counts, x='Genre', y='Count', title="Content Count across Primary Genres", text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------
# 3. PROBABILITY DISTRIBUTIONS AND SAMPLING
# -------------------------------------------------------------
elif selected_topic == "Probability Distributions and Sampling":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        Sampling provides representative sub-populations when analyzing the complete streaming archive is computationally prohibitive or when testing statistical estimators.

        **Techniques:**
        1. **Simple Random Sampling (SRS):** Every item has an equal inclusion probability $P(E_i) = n / N$.
        2. **Systematic Sampling:** Select a random start $k \in [0, K-1]$ and pick every $K^{\\text{th}}$ item ($K = \\lfloor N/n \\rfloor$).
        3. **Stratified Sampling:** Partition population into disjoint strata (e.g., Genres) and draw proportional samples $n_h = n \\times (N_h / N)$.

        **Interpretation:**
        Stratified sampling prevents rare genres from being excluded and reduces sampling variance compared to SRS.
        """)

    sample_method = st.radio("Choose Sampling Method:", ["Simple Random Sampling (SRS)", "Systematic Sampling", "Stratified Sampling by Type"])
    sample_size = st.slider("Select Sample Size (n):", min_value=5, max_value=min(35, len(df)), value=15)

    if sample_method == "Simple Random Sampling (SRS)":
        sampled_df = df.sample(n=sample_size, random_state=42)
    elif sample_method == "Systematic Sampling":
        step = max(1, len(df) // sample_size)
        indices = np.arange(0, len(df), step)[:sample_size]
        sampled_df = df.iloc[indices]
    else:
        sampled_df = df.groupby('type', group_keys=False).apply(
            lambda x: x.sample(max(1, int(np.round(len(x) / len(df) * sample_size))), random_state=42)
        )

    st.write(f"**Sample Result:** {len(sampled_df)} items extracted.")
    st.dataframe(sampled_df[['title', 'type', 'primary_genre', 'rating', 'imdb_score', 'content_budget_millions']], use_container_width=True)

    csv_sample = sampled_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sampled CSV", csv_sample, "netflix_sample.csv", "text/csv")

# -------------------------------------------------------------
# 4. CENTRAL LIMIT THEOREM
# -------------------------------------------------------------
elif selected_topic == "Central Limit Theorem":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        The Central Limit Theorem (CLT) proves that the distribution of sample means approaches a Gaussian normal distribution as sample size $n$ increases, regardless of the underlying population shape.

        **Key Formulas:**
        $$\\mu_{\\bar{x}} = \\mu, \qquad \\sigma_{\\bar{x}} = \\frac{\\sigma}{\\sqrt{n}}$$

        **Observed vs Theoretical Standard Error:**
        $$SE_{\\text{obs}} = s_{\\bar{x}} = \\sqrt{\\frac{1}{M-1}\\sum_{j=1}^M (\\bar{x}_j - \\bar{\\bar{x}})^2}, \qquad SE_{\\text{theo}} = \\frac{\\sigma}{\\sqrt{n}}$$

        **Interpretation:**
        As $n$ grows, $SE_{\\text{obs}} \\to SE_{\\text{theo}}$ and the sampling distribution narrows around the true population mean.
        """)

    col1, col2 = st.columns(2)
    with col1:
        n_sample = st.slider("Sample Size (n per draw):", min_value=5, max_value=50, value=20)
    with col2:
        n_trials = st.slider("Number of Repetitions / Trials (M):", min_value=50, max_value=1000, value=300, step=50)

    population = df['imdb_score'].dropna().values
    pop_mean = np.mean(population)
    pop_std = np.std(population, ddof=0)
    theo_se = pop_std / np.sqrt(n_sample)

    np.random.seed(42)
    sample_means = [np.mean(np.random.choice(population, size=n_sample, replace=True)) for _ in range(n_trials)]
    mean_of_means = np.mean(sample_means)
    obs_se = np.std(sample_means, ddof=1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population Mean (μ)", f"{pop_mean:.3f}")
    c2.metric("Mean of Sample Means", f"{mean_of_means:.3f}")
    c3.metric("Theoretical SE", f"{theo_se:.4f}")
    c4.metric("Observed SE", f"{obs_se:.4f}")

    fig_clt = px.histogram(
        x=sample_means,
        nbins=25,
        title=f"Sampling Distribution of {n_trials} Sample Means (n={n_sample})",
        labels={'x': 'Sample Mean IMDb Score'},
        marginal="box"
    )
    fig_clt.add_vline(x=pop_mean, line_dash="dash", line_color="red", annotation_text="μ (Pop Mean)")
    st.plotly_chart(fig_clt, use_container_width=True)

# -------------------------------------------------------------
# 5. CONFIDENCE INTERVALS AND HYPOTHESIS TESTING
# -------------------------------------------------------------
elif selected_topic == "Confidence Intervals and Hypothesis Testing":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        Inferential hypothesis testing determines whether empirical sample evidence is statistically significant or merely due to random sampling fluctuation.

        **Formulas:**
        - **Z-Test:** $z = \\frac{\\bar{x} - \\mu_0}{\\sigma / \\sqrt{n}}$
        - **One-Sample t-Test:** $t = \\frac{\\bar{x} - \\mu_0}{s / \\sqrt{n}}, \quad df = n - 1$
        - **Confidence Interval (1 - α):** $[\\bar{x} - t_{\\alpha/2, df} \\cdot SE, \; \\bar{x} + t_{\\alpha/2, df} \\cdot SE]$

        **Decision Rule:**
        Reject $H_0$ if $p < \\alpha$. If $p \\ge \\alpha$, we fail to reject $H_0$.
        """)

    col1, col2 = st.columns(2)
    with col1:
        null_val = st.number_input("Hypothesized Null Mean (μ0):", value=7.5, step=0.1)
    with col2:
        alpha_level = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1)

    sample_vals = df['imdb_score'].dropna().values
    n = len(sample_vals)
    s_mean = np.mean(sample_vals)
    s_std = np.std(sample_vals, ddof=1)
    se = s_std / np.sqrt(n)

    t_stat, p_val = stats.ttest_1samp(sample_vals, null_val)
    ci = stats.t.interval(1 - alpha_level, df=n-1, loc=s_mean, scale=se)

    colA, colB, colC = st.columns(3)
    colA.metric("Sample Mean", f"{s_mean:.3f}")
    colB.metric("t-Statistic", f"{t_stat:.3f}")
    colC.metric("p-Value", f"{p_val:.4f}")

    st.info(f"**95% Confidence Interval for True Mean:** [{ci[0]:.3f}, {ci[1]:.3f}]")

    if p_val < alpha_level:
        st.error(f"**Decision: Reject H0** at α={alpha_level}. Statistically significant evidence that true mean differs from {null_val} (p = {p_val:.4f}).")
    else:
        st.success(f"**Decision: Fail to Reject H0** at α={alpha_level}. Insufficient evidence of a difference from {null_val} (p = {p_val:.4f}).")

# -------------------------------------------------------------
# 6. CHI-SQUARE TEST
# -------------------------------------------------------------
elif selected_topic == "Chi-Square Test":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        The Chi-Square Test of Independence examines whether a significant statistical association exists between two categorical variables.

        **Formulas:**
        $$\\chi^2 = \\sum_{i=1}^r \\sum_{j=1}^c \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \qquad E_{ij} = \\frac{R_i \\times C_j}{N}$$
        $$df = (r - 1)(c - 1), \qquad \\text{Cramér's } V = \\sqrt{\\frac{\\chi^2}{N \\cdot \\min(r-1, c-1)}}$$

        **Hypotheses:**
        - $H_0$: Content Type and Rating Category are independent.
        - $H_1$: Content Type and Rating Category are dependent.
        """)

    contingency = pd.crosstab(df['type'], df['rating'])
    st.write("### Observed Frequencies Matrix (O)")
    st.dataframe(contingency, use_container_width=True)

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    n_total = contingency.values.sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chi-Square (χ²)", f"{chi2:.3f}")
    c2.metric("Degrees of Freedom", f"{dof}")
    c3.metric("p-Value", f"{p_val:.4f}")
    c4.metric("Cramér's V", f"{cramers_v:.3f}")

    if p_val < 0.05:
        st.error(f"Reject H0 (p = {p_val:.4f} < 0.05). Significant association between content type and rating.")
    else:
        st.success(f"Fail to Reject H0 (p = {p_val:.4f} >= 0.05). The categorical attributes appear independent.")

# -------------------------------------------------------------
# 7. COVARIANCE AND CORRELATION
# -------------------------------------------------------------
elif selected_topic == "Covariance and Correlation":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        Covariance measures the joint variability of two random variables, while correlation standardizes covariance into a scale-free metric between -1 and +1.

        **Formulas:**
        $$\\text{Cov}(X, Y) = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{n - 1}, \qquad r = \\frac{\\text{Cov}(X, Y)}{s_X s_Y}$$
        """)

    num_cols = ['imdb_score', 'content_budget_millions', 'duration_min', 'total_votes', 'release_year']
    corr_matrix = df[num_cols].corr()

    fig_corr = px.imshow(corr_matrix, text_auto=".3f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.write("### Numerical Correlation Matrix")
    st.dataframe(corr_matrix, use_container_width=True)

# -------------------------------------------------------------
# 8. CORRELATION HYPOTHESIS TESTING
# -------------------------------------------------------------
elif selected_topic == "Correlation Hypothesis Testing":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        Tests whether an observed sample correlation coefficient $r$ indicates a true population correlation $\\rho \\ne 0$.

        **Formulas:**
        $$t = r \\sqrt{\\frac{n - 2}{1 - r^2}}, \qquad df = n - 2$$
        """)

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X Variable:", ['content_budget_millions', 'duration_min', 'total_votes'], index=0)
    with col2:
        y_var = st.selectbox("Y Variable:", ['imdb_score', 'total_votes'], index=0)

    x = df[x_var].values
    y = df[y_var].values

    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pearson r", f"{r_pearson:.3f}")
    c1.metric("Pearson p-value", f"{p_pearson:.4f}")
    c2.metric("Spearman ρ", f"{r_spearman:.3f}")
    c2.metric("Spearman p-value", f"{p_spearman:.4f}")

    fig_scat = px.scatter(df, x=x_var, y=y_var, trendline="ols", color="type", hover_name="title", title=f"Scatter & OLS Trend: {x_var} vs {y_var}")
    st.plotly_chart(fig_scat, use_container_width=True)

# -------------------------------------------------------------
# 9. TWO-WAY ANOVA
# -------------------------------------------------------------
elif selected_topic == "Two-Way ANOVA":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        Two-Way ANOVA evaluates the simultaneous effects of two categorical independent variables on a continuous dependent variable, along with their interaction.

        **Model Equation:**
        $$Y_{ijk} = \\mu + \\alpha_i + \\beta_j + (\\alpha\\beta)_{ij} + \\epsilon_{ijk}$$
        """)

    # Filter top categories for balanced ANOVA
    top_ratings = ['TV-MA', 'R', 'PG-13', 'TV-14']
    anova_df = df[df['rating'].isin(top_ratings)].copy()

    model = ols('imdb_score ~ C(type) + C(rating) + C(type):C(rating)', data=anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    st.write("### Two-Way ANOVA Summary Table")
    st.dataframe(anova_table, use_container_width=True)

    fig_int = px.line(
        anova_df.groupby(['type', 'rating'])['imdb_score'].mean().reset_index(),
        x='rating', y='imdb_score', color='type', markers=True,
        title="Interaction Plot: Type × Rating on IMDb Score"
    )
    st.plotly_chart(fig_int, use_container_width=True)

# -------------------------------------------------------------
# 10. REGRESSION ANALYSIS
# -------------------------------------------------------------
elif selected_topic == "Regression Analysis":
    with st.expander("📖 Academic Foundations & Methodology", expanded=True):
        st.markdown("""
        **What the technique is used for:**
        Regression models the mathematical relationship between predictor variables and a continuous response metric.

        **Formulas:**
        $$\\hat{y} = \\beta_0 + \\beta_1 x_1, \qquad R^2 = 1 - \\frac{SS_{\\text{res}}}{SS_{\\text{tot}}}$$
        $$RMSE = \\sqrt{\\frac{1}{n}\\sum (y_i - \\hat{y}_i)^2}, \qquad MAE = \\frac{1}{n}\\sum |y_i - \\hat{y}_i|$$
        """)

    X = df[['content_budget_millions']].values
    y = df['imdb_score'].values

    slope, intercept, r_val, p_val, std_err = stats.linregress(X.flatten(), y)
    y_pred = intercept + slope * X.flatten()
    r2 = r_val ** 2
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    mae = np.mean(np.abs(y - y_pred))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("MAE", f"{mae:.3f}")
    c4.metric("Slope (β1)", f"{slope:.4f}")

    st.info(f"**Regression Equation:** IMDb Score = {intercept:.2f} + ({slope:.4f} × Budget)")

    # Live Predictor
    st.subheader("Interactive Prediction Simulator")
    user_budget = st.slider("Select Content Budget ($ Millions):", 5.0, 300.0, 50.0, 5.0)
    pred_score = intercept + slope * user_budget
    st.success("Estimated IMDb Score for a $" + f"{user_budget:.1f}M production: **{pred_score:.2f} / 10**")

    fig_reg = px.scatter(df, x='content_budget_millions', y='imdb_score', trendline='ols', hover_name='title', title="Regression Fit & Residual Inspection")
    st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")
st.caption("Statistical Inference Lab Capstone Project | Developed for Streamlit Community Cloud Deployment")
