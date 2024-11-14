import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import gdown

# Title
st.title("Mental Health and Depression Analysis")

# Load data
st.info("Loading Dataset...")
url = "https://raw.githubusercontent.com/jeffreyjoy271114/Kaggle-PlayGround-Series-S04E11-Mental-health-Depression-Prediction/refs/heads/main/Mental_health_clean_data.csv"
df = pd.read_csv(url)
X_raw = df.drop("Depression", axis=1)
y_raw = df["Depression"]

# Sidebar input features
with st.sidebar:
    st.header("Input Features")
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Age = st.slider("Age", min_value=18, max_value=60, step=1)
    City = st.selectbox("City", df["City"].unique())
    Working_Professional_or_Student = st.selectbox("Working Professional or Student", ("Working Professional", "Student"))
    Academic_Pressure = st.selectbox("Academic Pressure", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    Work_Pressure = st.selectbox("Work Pressure", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    CGPA = st.slider("CGPA (0.0 if Working Professional)", 0.0, 10.0)
    Study_Satisfaction = st.selectbox("Study Satisfaction", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    Job_Satisfaction = st.selectbox("Job Satisfaction", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    Sleep_Duration = st.selectbox("Sleep Duration", df["Sleep Duration"].unique())
    Dietary_Habits = st.selectbox("Dietary Habits", df["Dietary Habits"].unique())
    Degree = st.selectbox("Degree", df["Degree"].unique())
    Have_you_ever_had_suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    Financial_Stress = st.selectbox("Financial Stress", [1.0, 2.0, 3.0, 4.0, 5.0])

# Prepare input data
input_data = {
    "Gender": Gender,
    "Age": Age,
    "City": City,
    "Working Professional or Student": Working_Professional_or_Student,
    "Academic Pressure": Academic_Pressure,
    "Work Pressure": Work_Pressure,
    "CGPA": CGPA,
    "Study Satisfaction": Study_Satisfaction,
    "Job Satisfaction": Job_Satisfaction,
    "Sleep Duration": Sleep_Duration,
    "Dietary Habits": Dietary_Habits,
    "Degree": Degree,
    "Have you ever had suicidal thoughts?": Have_you_ever_had_suicidal_thoughts,
    "Financial Stress": Financial_Stress,
}
input_row = pd.DataFrame([input_data])

# Align input_row with training columns
st.info("Aligning Features...")
file_id = "1JbgPP424Z9_tB0Y50QBqxjebRMbq8dwR"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "columns.pkl", quiet=False)
columns = load("columns.pkl")
input_row = pd.get_dummies(input_row).reindex(columns=columns, fill_value=0)

# Load model
st.info("Loading Model...")
model_file_id = "1JbgPP424Z9_tB0Y50QBqxjebRMbq8dwR"
gdown.download(f"https://drive.google.com/uc?id={model_file_id}", "model.pkl", quiet=False)
clf = load("model.pkl")

# Predict
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Display results
st.subheader("Prediction")
st.write(f"Depression Risk: {'Yes' if prediction[0] else 'No'}")
st.write("Prediction Probabilities:", prediction_proba)
