import os
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = "student_model.joblib"

# -------------------------
# Dataset
# -------------------------
def generate_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    study_hours = rng.uniform(0, 10, n)
    marks = rng.uniform(0, 100, n)

    score = study_hours * 9 + marks * 0.6
    passed = (score >= 60).astype(int)

    X = np.column_stack((study_hours, marks))
    y = passed

    return X, y

# -------------------------
# Train model once
# -------------------------
def train_and_save():
    X, y = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump((model, acc), MODEL_PATH)
    return model, acc

# -------------------------
# Load model
# -------------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓")

st.title("🎓 Student Pass / Fail Prediction")
st.caption("Logistic Regression model (Hugging Face compatible)")

model, accuracy = load_model()

st.success(f"Model accuracy: {accuracy*100:.2f}%")

st.divider()

hours = st.number_input("Study hours per day", 0.0, 12.0, 3.0, 0.5)
marks = st.number_input("Previous exam marks", 0, 100, 55)

if st.button("Predict Result"):
    X_input = np.array([[hours, marks]])
    prob = model.predict_proba(X_input)[0][1]
    pred = model.predict(X_input)[0]

    if pred == 1:
        st.success(f"✅ PASS  (Probability: {prob*100:.1f}%)")
    else:
        st.error(f"❌ FAIL  (Probability: {(1-prob)*100:.1f}%)")

st.caption("Model trains once, then loads from disk.")
