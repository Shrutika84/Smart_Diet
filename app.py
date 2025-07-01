# streamlit_app/app.py

import streamlit as st
import pandas as pd
from src.predict import predict_diet

st.set_page_config(page_title="AI Diet Recommendation", layout="centered")
st.title("🥗 Smart Diet Recommender")
st.markdown("Get personalized diet advice powered by machine learning and LLMs.")

with st.form("diet_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.radio("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=20.0, value=70.0)
    height = st.number_input("Height (cm)", min_value=100, value=170)
    bmi = round(weight / ((height / 100) ** 2), 1)
    disease = st.selectbox("Disease Type", ["None", "Obesity", "Diabetes", "Hypertension", "Anemia"])
    severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
    activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "Active"])
    calories = st.slider("Daily Caloric Intake", 1000, 4000, 2000)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100.0, 300.0, 180.0)
    bp = st.slider("Blood Pressure (mmHg)", 90, 200, 120)
    glucose = st.slider("Glucose (mg/dL)", 70.0, 250.0, 100.0)
    restriction = st.selectbox("Dietary Restrictions", ["None", "Low_Sugar", "Vegetarian"])
    allergies = st.selectbox("Allergies", ["None", "Peanuts", "Lactose"])
    cuisine = st.selectbox("Preferred Cuisine", ["Indian", "Chinese", "Italian", "Mexican"])
    exercise = st.slider("Weekly Exercise (hrs)", 0.0, 15.0, 3.0)
    adherence = st.slider("Diet Plan Adherence (%)", 0.0, 100.0, 80.0)
    imbalance = st.slider("Nutrient Imbalance Score", 0.0, 5.0, 2.5)

    submitted = st.form_submit_button("Get Diet Recommendation")

if submitted:
    input_data = {
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'Weight_kg': weight,
        'Height_cm': height,
        'BMI': bmi,
        'Disease_Type': ["None", "Obesity", "Diabetes", "Hypertension", "Anemia"].index(disease),
        'Severity': ["Mild", "Moderate", "Severe"].index(severity),
        'Physical_Activity_Level': ["Sedentary", "Moderate", "Active"].index(activity),
        'Daily_Caloric_Intake': calories,
        'Cholesterol_mg/dL': cholesterol,
        'Blood_Pressure_mmHg': bp,
        'Glucose_mg/dL': glucose,
        'Dietary_Restrictions': ["None", "Low_Sugar", "Vegetarian"].index(restriction),
        'Allergies': ["None", "Peanuts", "Lactose"].index(allergies),
        'Preferred_Cuisine': ["Indian", "Chinese", "Italian", "Mexican"].index(cuisine),
        'Weekly_Exercise_Hours': exercise,
        'Adherence_to_Diet_Plan': adherence,
        'Dietary_Nutrient_Imbalance_Score': imbalance
    }

    with st.spinner("Generating personalized diet plan..."):
        diet_type, meal_plan = predict_diet(input_data)

    st.success(f"Predicted Diet Type: {diet_type}")
    st.markdown("---")
    st.markdown("### 🥘 Personalized Meal Plan")
    st.text(meal_plan)
