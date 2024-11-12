import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title for the app
st.title("Mental Health and Depression Analysis")

# Information about the dataset
st.info("""
This dataset was collected as part of a comprehensive survey aimed at understanding the factors contributing to depression risk among adults. 
It was collected during an anonymous survey conducted between January and June 2023 across various cities, targeting individuals from diverse backgrounds and professions.

Participants, ranging from 18 to 60 years old, voluntarily provided inputs on factors such as:
- **Age**
- **Gender**
- **City**
- **Degree**
- **Job satisfaction**
- **Study satisfaction**
- **Study/work hours**
- **Family history**, among others.

Participants were asked to provide inputs without requiring any professional mental health assessments or diagnostic test scores. The target variable, 'Depression', represents whether the individual is at risk of depression, marked as 'Yes' or 'No', based on their responses to lifestyle and demographic factors.

The dataset provides insights into how everyday factors might correlate with mental health risks, making it a useful resource for machine learning models aimed at mental health prediction. It is particularly valuable for identifying key contributors to mental health challenges in a non-clinical setting.
""")
with st.expander('**Data**'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/jeffreyjoy271114/Kaggle-PlayGround-Series-S04E11-Mental-health-Depression-Prediction/refs/heads/main/Mental_health_clean_data.csv')
  df

  st.write('**X**')
  X = df.drop('Depression', axis = 1)
  X
  
  st.write('**y**')
  y = df.Depression
  y

with st.expander('**Data Visualization**'):
  # Display a count plot for the 'Depression' column
  st.subheader("Count Plot of Gender")
  fig, ax = plt.subplots()
  sns.countplot(x='Gender', data = df, ax = ax)
  st.pyplot(fig) 

  st.subheader("Count Plot for Working Professional or Students")
  fig, ax = plt.subplots()
  sns.countplot(x='Working Professional or Student', data = df, ax = ax)
  st.pyplot(fig)

  st.subheader("Count Plot for Academic Pressure")
  fig, ax = plt.subplots()
  sns.countplot(x='Academic Pressure', data = df, ax = ax)
  st.pyplot(fig)
  
  st.subheader("Count Plot for Work Pressure")
  fig, ax = plt.subplots()
  sns.countplot(x='Work Pressure', data = df, ax = ax)
  st.pyplot(fig)

  st.subheader("Count Plot for Study Satisfaction")
  fig, ax = plt.subplots()
  sns.countplot(x='Study Satisfaction', data = df, ax = ax)
  st.pyplot(fig)

  st.subheader("Count Plot for Job Satisfaction")
  fig, ax = plt.subplots()
  sns.countplot(x='Job Satisfaction', data = df, ax = ax)
  st.pyplot(fig)

  st.subheader("Count Plot for Sleep Duration")
  fig, ax = plt.subplots()
  sns.countplot(x='Sleep Duration', data = df, ax = ax)
  st.pyplot(fig)

  st.subheader("Count Plot for Financial Stress")
  fig, ax = plt.subplots()
  sns.countplot(x='Financial Stress', data = df, ax = ax)
  st.pyplot(fig)

  st.subheader("Count Plot for Depression")
  fig, ax = plt.subplots()
  sns.countplot(x='Depression', data = df, ax = ax)
  st.pyplot(fig)  


# Data Preparations
with st.sidebar:
  st.header('Input Features')
  Gender = st.selectbox('Gender', ('Male', 'Female'))
  City = st.selectbox('City', ('Kalyan', 'Patna', 'Vasai-Virar', 'Kolkata', 'Ahmedabad', 'Meerut', 'Ludhiana', 'Pune', 'Rajkot', 'Visakhapatnam', 'Srinagar', 'Mumbai', 'Indore', 'Agra', 'Surat', 'Varanasi', 'Vadodara', 'Hyderabad', 'Kanpur', 'Jaipur', 'Thane', 'Lucknow', 'Nagpur', 'Bangalore', 'Chennai', 'Ghaziabad', 'Delhi', 'Bhopal', 'Faridabad', 'Nashik'))
  Working_Professional_or_Student = st.selectbox('Working Professional or Student', ('Working Professional', 'Student'))
  Profession = st.selectbox('Profession', ('Student', 'Teacher', 'Professional Workers', 'Content Writer', 'Architect', 'Consultant', 'HR Manager', 'Pharmacist', 'Doctor', 'Business Analyst', 'Chemist', 'Entrepreneur', 'Chef', 'Educational Consultant', 'Data Scientist', 'Researcher', 'Lawyer', 'Customer Support', 'Marketing Manager', 'Pilot', 'Travel Consultant', 'Plumber', 'Sales Executive', 'Manager', 'Judge', 'Electrician', 'Financial Analyst', 'Software Engineer', 'Civil Engineer', 'UX/UI Designer', 'Digital Marketer' ,'Accountant', 'Finanancial Analyst', 'Mechanical Engineer', 'Graphic Designer', 'Research Analyst', 'Investment Banker'))
  Academic_Pressure	= st.selectbox('Academic Pressure', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  Work_Pressure = st.selectbox('Work Pressure	', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  Study_Satisfaction = st.selectbox('Study Satisfaction', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  Job_Satisfaction = st.selectbox('Job Satisfaction', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
  Sleep_Duration = st.selectbox('Sleep Duration', ('Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours'))
  Dietary_Habits = st.selectbox('Dietary Habits', ('Moderate', 'Unhealthy', 'Healthy'))	
  Degree = st.selectbox('Degree', ('Class 12', 'B.Ed', 'B.Arch', 'B.Com', 'B.Pharm', 'BCA' ,'M.Ed', 'MCA', 'BBA', 'BSc', 'MSc', 'LLM', 'M.Pharm', 'M.Tech', 'B.Tech', 'LLB', 'BHM', 'MBA', 'BA', 'ME', 'MD', 'MHM', 'PhD', 'BE', 'M.Com', 'MBBS', 'MA', 'Others'))
  Have_you_ever_had_suicidal_thoughts = st.selectbox('Have you ever had suicidal thoughts ?', ('Yes', 'No'))	
  Work/Study_Hours = st.selectbox('Work/Study Hours', (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
  Financial_Stress = st.selectbox('Financial Stress', (1.0, 2.0, 3.0, 4.0, 5.0))
  Family_History_of_Mental_Illness = st.selectbox('Family History of Mental Illness', ('Yes', 'No'))
  

