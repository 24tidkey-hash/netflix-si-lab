# Netflix Content Trends and Genre Analysis
### Statistical Inference Lab — College Capstone Project

A comprehensive web application demonstrating practical statistical inference, sampling theory, hypothesis testing, analysis of variance, and predictive modeling on Netflix's global content library.

---

## 📌 Project Overview
- **College Subject:** Statistical Inference Lab (SI Lab / Big Data Analytics)
- **Domain:** Media Analytics, Streaming Intelligence, Content Curation
- **Framework:** Streamlit, Python 3.10+, SciPy, Statsmodels, Plotly, Pandas
- **Deployment Platform:** Streamlit Community Cloud (Hosted via GitHub)

---

## 🧪 Laboratory Practicals Covered
1. **Web Scraping and CSV Creation:** Scraping structured film metadata with fallback resilient streaming catalogs.
2. **Data Cleaning and Exploratory Data Analysis:** Imputation, IQR outlier detection, summary statistics, histograms, and box plots.
3. **Probability Distributions and Sampling:** Simple Random Sampling (SRS), Systematic Sampling, and Stratified Sampling across genres.
4. **Central Limit Theorem (CLT):** Empirical simulation of sampling distribution of means, comparing observed standard error with theoretical $\sigma / \sqrt{n}$.
5. **Confidence Intervals and Hypothesis Testing:** One-sample Z-tests and Student's t-tests with parametric confidence intervals.
6. **Chi-Square Test of Independence:** Testing association between content type (Movie vs TV Show) and age rating classifications.
7. **Covariance and Correlation:** Evaluating bivariate linear and monotonic relationships between budgets, runtime, and IMDb scores.
8. **Correlation Hypothesis Testing:** Testing significance of Pearson's $r$ and Spearman's $\rho$ ($H_0: \rho = 0$).
9. **Two-Way ANOVA:** Factorial ANOVA testing main effects and interaction between Content Type and Rating Tier on audience rating.
10. **Regression Analysis:** Ordinary Least Squares (OLS) regression with $R^2$, MAE, RMSE evaluation, residual diagnostics, and live prediction simulator.

---

## 🚀 How to Run Locally

### 1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/netflix-statistical-inference-lab.git
cd netflix-statistical-inference-lab
```

### 2. Create and activate a virtual environment
```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (cmd):
python -m venv venv
venv\Scripts\activate
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application
```bash
streamlit run app.py
```
The application will open automatically at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud

1. Create a free account at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
2. Push this project folder to a public GitHub repository.
3. In Streamlit Community Cloud, click **"New app"**.
4. Select your repository, set the branch to `main`, and specify `app.py` as the Main file path.
5. Click **"Deploy"**! Your public URL will be generated within 2 minutes.

---

## 📊 Dataset Structure
The dataset contains curated records with fields:
- `show_id`: Unique identifier
- `type`: Movie or TV Show
- `title`: Title of release
- `director`: Director name
- `country`: Country of origin
- `release_year`: Year released
- `rating`: Content age classification (TV-MA, PG-13, TV-14, R, PG)
- `duration_min`: Runtime in minutes
- `primary_genre`: Dominant genre category
- `imdb_score`: Audience score (1.0 - 10.0)
- `total_votes`: IMDb vote count
- `content_budget_millions`: Estimated production budget in $M
