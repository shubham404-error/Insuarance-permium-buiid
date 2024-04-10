import streamlit as st
import pandas as pd
import pickle

# Load the model
file = open("./gradient_boosting_regressor_model.pkl", 'rb')
model = pickle.load(file)

# Load the data for dropdown options
data = pd.read_csv('./clean_data.csv')

st.title("Medical Insurance Cost Prediction")

# Input fields
age = st.number_input("Age", min_value=1, value=30)
sex = st.selectbox("Sex", options=sorted(data['sex'].unique()))
bmi = st.number_input("BMI", min_value=10.0, value=25.0, format="%.1f")
children = st.number_input("Number of Children", min_value=0, value=0)
smoker = st.selectbox("Smoker", options=sorted(data['smoker'].unique()))
region = st.selectbox("Region", options=sorted(data['region'].unique()))

# Prediction button
if st.button("Predict Cost"):
    # Create input dataframe
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.success(f"Predicted Medical Insurance Cost: ${prediction:.2f}")
