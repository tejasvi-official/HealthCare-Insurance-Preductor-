import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Insurance Predictor", layout="wide")

# ---------------- PREMIUM UI STYLE ---------------- #
st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-top: 0px;
    margin-bottom: 10px;
}

.navbar {
    display: flex;
    justify-content: center;
    gap: 25px;
    margin-top: 10px;
    margin-bottom: 10px;
}

.nav-item {
    background: #111827;
    padding: 10px 16px;
    border-radius: 10px;
    border: 1px solid #1f2937;
    color: #38bdf8;
    font-size: 14px;
    text-align: center;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1f2937;
}

.metric {
    font-size: 22px;
    font-weight: bold;
    color: #38bdf8;
}

.small {
    font-size: 13px;
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df.dropna()

@st.cache_data
def preprocess(df):
    df_copy = df.copy()
    le = LabelEncoder()
    for col in ['gender', 'diabetic', 'smoker', 'region']:
        df_copy[col] = le.fit_transform(df_copy[col])
    return df_copy

@st.cache_data
def train_model(df):
    df_ml = preprocess(df)

    X = df_ml[['age','gender','bmi','bloodpressure','diabetic','children','smoker','region']]
    y = df_ml['claim']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    lr = LinearRegression()

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    return rf, lr

# ---------------- DATA ---------------- #
df = load_data()
df_corr = preprocess(df)
rf_model, lr_model = train_model(df)

# ---------------- HEADER ---------------- #
st.markdown('<div class="main-title">🧠 AI Health Insurance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict insurance cost using Machine Learning + Correlation Analysis</div>', unsafe_allow_html=True)

# ---------------- NAVBAR ---------------- #
st.markdown("""
<div class="navbar">
    <div class="nav-item">📊 Dataset</div>
    <div class="nav-item">🤖 Models</div>
    <div class="nav-item">📈 Visuals</div>
    <div class="nav-item">📌 Insights</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("🧾 Patient Details")

age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["male","female"])
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
bp = st.sidebar.slider("Blood Pressure", 80, 140, 90)
diabetic = st.sidebar.selectbox("Diabetic", ["No","Yes"])
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["No","Yes"])
region = st.sidebar.selectbox("Region", ["southeast","southwest","northeast","northwest"])

# encoding
gender_enc = 1 if gender == "male" else 0
diabetic_enc = 1 if diabetic == "Yes" else 0
smoker_enc = 1 if smoker == "Yes" else 0
region_map = {"southeast":0,"southwest":1,"northeast":2,"northwest":3}
region_enc = region_map[region]

input_data = np.array([[age, gender_enc, bmi, bp, diabetic_enc, children, smoker_enc, region_enc]])

# ---------------- TABS ---------------- #
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🤖 Models", "📈 Visuals", "📌 Insights"])

# ---------------- DATASET TAB ---------------- #
with tab1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head())

# ---------------- MODEL TAB ---------------- #
with tab2:
    st.subheader("🤖 Model Prediction")

    if st.button("🚀 Predict Cost"):
        rf_pred = rf_model.predict(input_data)[0]
        lr_pred = lr_model.predict(input_data)[0]
        avg = (rf_pred + lr_pred) / 2

        col1, col2, col3 = st.columns(3)

        col1.markdown(f'<div class="card"><div class="small">Random Forest</div><div class="metric">${rf_pred:.2f}</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="card"><div class="small">Linear Regression</div><div class="metric">${lr_pred:.2f}</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="card"><div class="small">AI Average</div><div class="metric">${avg:.2f}</div></div>', unsafe_allow_html=True)

# ---------------- VISUAL TAB ---------------- #
with tab3:
    st.subheader("📈 Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="smoker", y="claim", ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="bmi", y="claim", ax=ax)
        st.pyplot(fig)

# ---------------- INSIGHTS TAB ---------------- #
with tab4:
    st.subheader("📌 Correlation & Insights")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("""
    ### 🔍 Key Insights
    - 💰 Smoker is the strongest factor  
    - 📈 BMI increases claim  
    - 🎂 Age impacts cost  
    - 🧬 Health conditions raise risk  
    """)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown("🚀 Built with Streamlit | ML Project")