import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="USA Housing AI", layout="wide")

# -----------------------------
# PREMIUM UI (GLASS + ANIMATION)
# -----------------------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: 0.3s ease;
}
.glass:hover {
    transform: translateY(-5px);
}

/* Title */
.title {
    font-size: 34px;
    font-weight: 700;
    color: white;
    text-align: center;
}

/* Subtitle */
.subtitle {
    color: #cbd5f5;
    text-align: center;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
    transition: 0.3s;
}
.stButton > button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Dashboard", "🤖 AI Assistant"])

# -----------------------------
# HOME TAB
# -----------------------------
with tab1:
    st.markdown("""
    <div class="glass">
        <div class="title">🏡 USA Housing Price Predictor</div>
        <div class="subtitle">
            Predict housing prices using machine learning models
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown("<div class='glass'>📊 Data Insights</div>", unsafe_allow_html=True)
    col2.markdown("<div class='glass'>🏠 Price Prediction</div>", unsafe_allow_html=True)
    col3.markdown("<div class='glass'>🤖 AI Assistant</div>", unsafe_allow_html=True)

# -----------------------------
# MODEL FUNCTIONS
# -----------------------------
def preprocess(df, target):
    df = df.drop_duplicates().ffill()
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    y = df[target]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    return model, X.columns, score

def predict(model, sample, cols):
    sample = sample.reindex(columns=cols, fill_value=0)
    return model.predict(sample)[0]

# -----------------------------
# DASHBOARD TAB
# -----------------------------
with tab2:

    st.markdown("<div class='glass'>📊 Upload Dataset</div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload Housing Dataset")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        target = st.selectbox("Select Target (Price)", df.columns)

        if st.button("Train Model"):

            with st.spinner("Training model..."):

                X, y = preprocess(df, target)
                model, cols, score = train_model(X, y)

                sample = X.iloc[0:1]

                # Save
                st.session_state["model"] = model
                st.session_state["cols"] = cols
                st.session_state["sample"] = sample
                st.session_state["df"] = df

            st.success(f"Model trained! R² Score: {score:.3f}")

            # Visualization
            st.markdown("### 📈 Feature Distribution")
            df.hist(figsize=(10,6))
            st.pyplot(plt)

# -----------------------------
# CHATBOT LOGIC
# -----------------------------
def local_chatbot(query):
    query = query.lower()

    df = st.session_state.get("df")
    model = st.session_state.get("model")

    if model is None:
        return "Please train the model first."

    # prediction from query
    numbers = re.findall(r"\d+\.?\d*", query)

    if numbers:
        value = float(numbers[0])
        sample = st.session_state["sample"].copy()

        # assume first column numeric feature
        col = sample.columns[0]
        sample[col] = value

        pred = predict(model, sample, st.session_state["cols"])
        return f"Predicted house price is ${pred:,.2f}"

    if "columns" in query:
        return f"Columns: {', '.join(df.columns)}"

    if "rows" in query:
        return f"Dataset has {df.shape[0]} rows"

    if "summary" in query:
        return df.describe().to_string()

    return "Ask about price prediction, dataset, or features."

# -----------------------------
# CHATBOT TAB
# -----------------------------
with tab3:

    st.markdown("<div class='glass'>🤖 AI Housing Assistant</div>", unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.info("Train model first in Dashboard")

    else:
        if "chat" not in st.session_state:
            st.session_state.chat = []

        user_input = st.chat_input("Ask about housing prices...")

        if user_input:
            reply = local_chatbot(user_input)

            st.session_state.chat.append(("You", user_input))
            st.session_state.chat.append(("AI", reply))

        for role, msg in st.session_state.chat:
            with st.chat_message("user" if role=="You" else "assistant"):
                st.write(msg)
