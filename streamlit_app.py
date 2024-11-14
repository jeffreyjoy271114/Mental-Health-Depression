import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder

# Title for the app
st.title("Mental Health and Depression Analysis")

# Model URL on Hugging Face (replace with your actual model's URL)
model_url = "https://huggingface.co/jeffrey-joy/mental_health_Data/resolve/main/model.pkl"  # Replace with actual URL

# Download the model from Hugging Face
st.info("Loading model from Hugging Face...")
output_path = "model.pkl"
hf_hub_download(repo_id="jeffrey-joy/mental_health_Data", filename="model.pkl", local_dir=output_path)

# Load the model using joblib
clf = load(output_path)
st.success("Model loaded successfully from Hugging Face!")

# Load the label encoders used during training (if you have saved them)
# You can save them when training, and then load here
label_encoders = {}

# Information about the dataset
st.info(""" 
This dataset was collected as part of a comprehensive survey aimed at understanding the factors contributing to depression risk among adults. 
It was collected during an anonymous survey conducted between January and June 2023 across various cities, targeting individuals from diverse backgrounds and professions.
""")

# Define the LabelEncoder for each categorical feature
categorical_columns = ['Gender', 'City', 'Working Professional or Student', 
                       'Profession', 'Sleep Duration', 'Dietary Habits', 
                       'Degree', 'Have you ever had suicidal thoughts ?',
                       'Family History of Mental Illness']

# Input features
with st.sidebar:
  st.header('Input Features')
  Gender = st.selectbox('Gender', ('Male', 'Female'))
  Age = st.slider('Age', min_value=18, max_value=60, step=1)
  City = st.selectbox('City', ('Kalyan', 'Patna', 'Vasai-Virar', 'Kolkata', 'Ahmedabad', 'Meerut', 'Ludhiana', 'Pune', 'Rajkot', 'Visakhapatnam', 'Srinagar', 'Mumbai', 'Indore', 'Agra', 'Surat', 'Varanasi', 'Vadodara', 'Hyderabad', 'Kanpur', 'Jaipur', 'Thane', 'Lucknow', 'Nagpur', 'Bangalore', 'Chennai', 'Ghaziabad', 'Delhi', 'Bhopal', 'Faridabad', 'Nashik'))
  Working_Professional_or_Student = st.selectbox('Working Professional or Student', ('Working Professional', 'Student'))
  Profession = st.selectbox('Profession', ('Student', 'Teacher', 'Professional Workers', 'Content Writer', 'Architect', 'Consultant', 'HR Manager', 'Pharmacist', 'Doctor', 'Business Analyst', 'Chemist', 'Entrepreneur', 'Chef', 'Educational Consultant', 'Data Scientist', 'Researcher', 'Lawyer', 'Customer Support', 'Marketing Manager', 'Pilot', 'Travel Consultant', 'Plumber', 'Sales Executive', 'Manager', 'Judge', 'Electrician', 'Financial Analyst', 'Software Engineer', 'Civil Engineer', 'UX/UI Designer', 'Digital Marketer' ,'Accountant', 'Finanancial Analyst', 'Mechanical Engineer', 'Graphic Designer', 'Research Analyst', 'Investment Banker'))
  Academic_Pressure = st.selectbox('Academic Pressure (0.0 if Working Professional)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  Work_Pressure = st.selectbox('Work Pressure	(0.0 if Student)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  CGPA = st.slider('CGPA (0.0 if Working Professional)', 0.0, 10.0)
  Study_Satisfaction = st.selectbox('Study Satisfaction (0.0 if Working Professional)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  Job_Satisfaction = st.selectbox('Job Satisfaction (0.0 if Student)', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  Sleep_Duration = st.selectbox('Sleep Duration', ('Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours'))
  Dietary_Habits = st.selectbox('Dietary Habits', ('Moderate', 'Unhealthy', 'Healthy'))	
  Degree = st.selectbox('Degree', ('Class 12', 'B.Ed', 'B.Arch', 'B.Com', 'B.Pharm', 'BCA' ,'M.Ed', 'MCA', 'BBA', 'BSc', 'MSc', 'LLM', 'M.Pharm', 'M.Tech', 'B.Tech', 'LLB', 'BHM', 'MBA', 'BA', 'ME', 'MD', 'MHM', 'PhD', 'BE', 'M.Com', 'MBBS', 'MA', 'Others'))
  Have_you_ever_had_suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts ?', ('Yes', 'No'))	
  Work_Study_Hours = st.selectbox('Work/Study Hours', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
  Financial_Stress = st.selectbox('Financial Stress', (1.0, 2.0, 3.0, 4.0, 5.0))
  Family_History_of_Mental_Illness = st.selectbox('Family History of Mental Illness', ('Yes', 'No'))

# Create a DataFrame for the input features
data = {'Gender' : Gender,
        'Age' : Age,
        'City' : City,
        'Working Professional or Student' :  Working_Professional_or_Student,
        'Profession' : Profession,
        'Academic Pressure' : Academic_Pressure, 
        'Work Pressure' : Work_Pressure,
        'CGPA' : CGPA,
        'Study Satisfaction' : Study_Satisfaction,
        'Job Satisfaction' : Job_Satisfaction, 
        'Sleep Duration' : Sleep_Duration,
        'Dietary Habits' : Dietary_Habits, 
        'Degree' : Degree,
        'Have you ever had suicidal thoughts ?' : Have_you_ever_had_suicidal_thoughts,
        'Work/Study Hours' : Work_Study_Hours,
        'Financial Stress' : Financial_Stress,
        'Family History of Mental Illness' : Family_History_of_Mental_Illness}
input_df = pd.DataFrame(data, index = [0])

# Apply Label Encoding to the categorical columns
input_df_encoded = input_df.copy()
for column in categorical_columns:
    encoder = LabelEncoder()
    input_df_encoded[column] = encoder.fit_transform(input_df_encoded[column])

# Now use this encoded data for prediction
prediction = clf.predict(input_df_encoded)
prediction_proba = clf.predict_proba(input_df_encoded)

# Display the predicted probabilities
df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['No Depression', 'Depression']
df_prediction_proba.rename(columns={0: 'No Depression', 1: 'Depression'})

# Display the predicted Results
st.subheader('Predicted Results')
st.dataframe(df_prediction_proba)

mental_biforcate = np.array(['You are NOT a victim of Depression', 'You are a victim of Depression'])
st.success(str(mental_biforcate[prediction][0]))
