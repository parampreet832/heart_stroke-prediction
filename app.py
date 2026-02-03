import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scalar = joblib.load('scalar.pkl')
expected_columns = joblib.load('columns.pkl')

st.title('Heart Stroke Prediction by Parampreet Kaur')
st.markdown('Provide the following details')

age = st.slider("Age",18,100,40)
sex = st.selectbox("Sex", ['M', 'F'])
chestpain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
restingbp = st.number_input("Resting Blood Pressure (mm HG)", 80, 200, 100)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restingecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button('Predict'):
    raw_input = {
    'Age' : age,
    'RestingBP' : restingbp,
    'Cholesterol' : cholesterol,
    'FastingBS' : fastingbs,
    'MaxHR' : max_hr,
    'Oldpeak' : oldpeak,
    'Sex_' + sex : 1,
    'ChestPainType_' + chestpain : 1,
    'RestingECG_' + restingecg : 1,
    'ExerciseAngina' + exercise_angina : 1,
    'ST_Slope_' + st_slope : 1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
         input_df[col] = 0

    input_df = input_df[expected_columns]
    scaler_input = scalar.transform(input_df)
    prediction = model.predict(scaler_input)[0]
         
    if prediction == 1:
        st.error("High Risk of Heart Diesease")
    else:
        st.success("Low Risk of Heart Diesease")