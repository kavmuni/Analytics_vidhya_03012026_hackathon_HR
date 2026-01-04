import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb


st.title("HR Promotion Prediction Web App")
hr_train_df = pd.read_csv('Dataset/train.csv')
# Input fields for user to enter employee data
department = st.selectbox("Department", options=hr_train_df['department'].unique())
region = st.selectbox("Region", options=hr_train_df['region'].unique())
education = st.selectbox("Education", options=hr_train_df['education'].unique())
gender = st.selectbox("Gender", options=hr_train_df['gender'].unique())
recruitment_channel = st.selectbox("Recruitment Channel", options=hr_train_df['recruitment_channel'].unique())
no_of_trainings = st.number_input("Number of Trainings", min_value=0, max_value=20, value=1)
age = st.number_input("Age", min_value=18, max_value=65, value=30)
previous_year_rating = st.number_input("Previous Year Rating", min_value=1, max_value=5, value=3)
length_of_service = st.number_input("Length of Service (years)", min_value=0, max_value=40, value=5)
KPIs_met_gt_80 = st.selectbox("KPIs Met > 80%", options=[0, 1])
awards_won = st.selectbox("Awards Won", options=[0, 1])
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=70)

# Load the model
model_pipeline = jb.load('model/xgb_hr_model.pkl')

# Create a DataFrame for the input data
input_data = pd.DataFrame({
        'department': [department],
        'region': [region],
        'education': [education],
        'gender': [gender],
        'recruitment_channel': [recruitment_channel],
        'no_of_trainings': [no_of_trainings],
        'age': [age],
        'previous_year_rating': [previous_year_rating],
        'length_of_service': [length_of_service],
        'KPIs_met_gt_80': [KPIs_met_gt_80],
        'awards_won': [awards_won],
        'avg_training_score': [avg_training_score]
    })

# Button to make prediction
if st.button("Predict Promotion"):
    # Make prediction
    prediction = model_pipeline.predict(input_data)
    # Display result
    if prediction[0] == 1:
        st.success("The employee is likely to be promoted.")
    else:
        st.info("The employee is unlikely to be promoted.")

