import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open("linear_reg.pkl", 'rb'))

# Load the data for dropdown options
data = pd.read_csv('./clean_data.csv')
st.set_page_config(page_title=InsuaranceWiz, page_icon='🧙‍♂️', layout="centered",
# App Title
st.title("InsuaranceWiz🧙‍♂️🪄-Medical Insurance Cost Prediction")

# Sidebar with Explanations
with st.sidebar:
    st.header("About the Data")
    st.markdown("""
    This app predicts medical insurance costs based on several factors:

    * **Age:** Age of the individual.
    * **Sex:** Gender of the individual (Male or Female).
    * **BMI:** Body Mass Index, a measure of body fat based on height and weight.
    * **Children:** Number of children the individual has.
    * **Smoker:** Whether the individual is a smoker or not.
    * **Region:** The region where the individual resides.

    **Note:** The model is trained on historical data and provides an estimate. Actual costs may vary.
    """)

# Input fields in the main area
st.header("Enter Your Information")
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
