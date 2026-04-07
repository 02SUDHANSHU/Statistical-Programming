import streamlit as st
import pandas as pd
import joblib

model = joblib.load('RandomForest_model1.pkl')
scaler = joblib.load('RandomForest_scaler1.pkl')

st.title("Smart Insurance Premium Predictor")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    height_cm = st.number_input("Height (cm)", 100.0, 250.0, 170.0)

with col2:
    surgeries = st.selectbox("Major Surgeries", [0, 1, 2, 3])
    diabetes = st.checkbox("Diabetes")
    bp_probs = st.checkbox("Blood Pressure Problems")
    chronic = st.checkbox("Chronic Diseases")
    transplant = st.checkbox("Previous Transplants")
    cancer_hist = st.checkbox("Family Cancer History")

height_m = height_cm / 100
calculated_bmi = round(weight / (height_m ** 2), 2)

risk_score = 0
if diabetes: risk_score += 1
if bp_probs: risk_score += 1
if chronic: risk_score += 1
if transplant: risk_score += 1
if surgeries: risk_score += 1

risk_score = min(risk_score, 5)

st.info(f"Calculated BMI: **{calculated_bmi}** | Estimated Health Risk Score: **{risk_score}**")

if st.button("Predict Premium"):

    input_features = pd.DataFrame([[
        age, 
        risk_score, 
        1 if transplant else 0,
        surgeries,
        1 if chronic else 0,
        1 if bp_probs else 0,
        weight,
        calculated_bmi,
        1 if cancer_hist else 0
    ]], columns=['Age', 'HealthRiskScore', 'AnyTransplants', 'NumberOfMajorSurgeries', 
                 'AnyChronicDiseases', 'BloodPressureProblems', 'Weight', 'BMI', 
                 'HistoryOfCancerInFamily'])

    scaled_input = scaler.transform(input_features)
    prediction = model.predict(scaled_input)
    
    st.success(f"Estimated Annual Premium: ₹{prediction[0]:,.2f}")