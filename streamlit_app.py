import streamlit as st
import pandas 
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


