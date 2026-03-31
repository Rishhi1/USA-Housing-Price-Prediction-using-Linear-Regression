
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import bcrypt
import openai
import re

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Pricing SaaS", layout="wide")

# -----------------------------
# DATABASE SETUP
# -----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT,
    password TEXT
)
""")
conn.commit()

# -----------------------------
# AUTH FUNCTIONS
# -----------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def create_user(username, password):
    c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    data = c.fetchone()
    if data:
        return check_password(password, data[0])
    return False

# -----------------------------
# LOGIN UI
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Login / Signup")

    choice = st.selectbox("Login or Signup", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Signup":
        if st.button("Create Account"):
            create_user(username, password)
            st.success("Account created!")

    else:
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# -----------------------------
# OPENAI SETUP
# -----------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Dashboard", "🤖 AI Assistant"])

# -----------------------------
# HOME
# -----------------------------
with tab1:
    st.markdown("## 🚀 AI Pricing Intelligence Platform")
    st.write("Optimize pricing using AI")

# -----------------------------
# DASHBOARD
# -----------------------------
with tab2:

    uploaded_file = st.file_uploader("Upload Dataset")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)

        if st.button("Train Model"):

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            score = r2_score(y_test, preds)

            st.success(f"Model R²: {score:.3f}")

            sample = X.iloc[0:1]

            st.session_state["model"] = model
            st.session_state["X_cols"] = X.columns
            st.session_state["sample"] = sample

# -----------------------------
# CHATBOT
# -----------------------------
with tab3:

    st.subheader("🤖 AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask anything...")

    def gpt_reply(prompt):
        if not openai.api_key:
            return "No API key provided"

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    if user_input:

        reply = gpt_reply(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": reply})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
