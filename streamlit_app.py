import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import requests
from joblib import load

# Title for the app
st.title("Mental Health and Depression Analysis")

# Information about the dataset
st.info("""
This dataset provides insights into how everyday factors might correlate with mental health risks, making it a valuable resource for machine learning models.
""")

# Load dataset
with st.expander('**Data**'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/jeffreyjoy271114/Kaggle-PlayGround-Series-S04E11-Mental-health-Depression-Prediction/refs/heads/main/Mental_health_clean_data.csv')
    st.dataframe(df)

    st.write('**X**')
    X_raw = df.drop('Depression', axis=1)
    st.dataframe(X_raw)

    st.write('**y**')
    y_raw = df.Depression
    st.write(y_raw)

# Sidebar for Input Features
with st.sidebar:
    st.header('Input Features')
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    Age = st.slider('Age', min_value=18, max_value=60, step=1)
    City = st.selectbox('City', df["City"].unique())
    Working_Professional_or_Student = st.selectbox('Working Professional or Student', ('Working Professional', 'Student'))
    Academic_Pressure = st.selectbox('Academic Pressure (0.0 if Working Professional)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    Work_Pressure = st.selectbox('Work Pressure (0.0 if Student)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    CGPA = st.slider('CGPA (0.0 if Working Professional)', 0.0, 10.0)
    Study_Satisfaction = st.selectbox('Study Satisfaction (0.0 if Working Professional)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    Job_Satisfaction = st.selectbox('Job Satisfaction (0.0 if Student)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    Sleep_Duration = st.selectbox('Sleep Duration', df["Sleep Duration"].unique())
    Dietary_Habits = st.selectbox('Dietary Habits', df["Dietary Habits"].unique())
    Degree = st.selectbox('Degree', df["Degree"].unique())
    Have_you_ever_had_suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts ?', ('Yes', 'No'))
    Financial_Stress = st.selectbox('Financial Stress', (1.0, 2.0, 3.0, 4.0, 5.0))

# Prepare input data
data = {
    'Gender': Gender,
    'Age': Age,
    'City': City,
    'Working Professional or Student': Working_Professional_or_Student,
    'Academic Pressure': Academic_Pressure,
    'Work Pressure': Work_Pressure,
    'CGPA': CGPA,
    'Study Satisfaction': Study_Satisfaction,
    'Job Satisfaction': Job_Satisfaction,
    'Sleep Duration': Sleep_Duration,
    'Dietary Habits': Dietary_Habits,
    'Degree': Degree,
    'Have you ever had suicidal thoughts ?': Have_you_ever_had_suicidal_thoughts,
    'Financial Stress': Financial_Stress
}
input_df = pd.DataFrame([data])

# Label Encoding
st.info("Encoding Features Using LabelEncoder...")

# Columns to encode
encode_columns = ['Gender', 'City', 'Working Professional or Student', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts?']

# Load the encoders (assuming these were saved during model training)
encoders_url = "https://huggingface.co/jeffrey-joy/mental_health_Data/resolve/main/model.pkl"  # Replace with your actual URL
response = requests.get(encoders_url)

if response.status_code == 200:
    with open("encoders.pkl", "wb") as f:
        f.write(response.content)
    encoders = load("encoders.pkl")
    st.success("Encoders loaded successfully!")
else:
    st.error("Failed to load encoders. Please check the URL and try again.")

# Apply encoders to input data
for col in encode_columns:
    encoder = encoders.get(col)
    if encoder:
        input_df[col] = encoder.transform(input_df[col])

# Ensure input matches model's training format
input_encoded = input_df.reindex(columns=X_raw.columns, fill_value=0)

# Load model from Hugging Face
st.info("Loading the model from Hugging Face...")
model_url = "https://huggingface.co/jeffrey-joy/mental_health_Data/resolve/main/model.pkl"  # Replace with your actual URL
response = requests.get(model_url)

if response.status_code == 200:
    with open("model.pkl", "wb") as f:
        f.write(response.content)
    model = load("model.pkl")
    st.success("Model loaded successfully!")
else:
    st.error("Failed to load the model. Please check the URL and try again.")

# Make predictions
if 'model' in locals():
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)

    # Display results
    st.subheader('Prediction Results')
    st.write(f"Depression Risk: {'Yes' if prediction[0] else 'No'}")
    st.write("Prediction Probabilities:")
    st.write(pd.DataFrame(prediction_proba, columns=['No Depression', 'Depression']))
