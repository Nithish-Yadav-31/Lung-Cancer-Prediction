import streamlit as st
import joblib
import pandas as pd

# Load the SVM model
svm_model = joblib.load('svm_model.pkl')

# Unique values for each parameter
unique_values = {
    'Alcoholuse': [1, 2, 3, 4, 5],
    'DustAllergy': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'OccuPationalHazards': [1, 2, 3, 4, 5, 6, 7, 8],
    'GeneticRisk': [1, 2, 3, 4, 5, 6, 7, 8],
    'Obesity': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'CoughingofBlood': [1, 2, 3, 4, 5, 6, 7, 8],
    'Fatigue': [1, 2, 3, 4, 5, 6, 7, 8],
}

# Streamlit app code
st.title('Lung Disease Prediction')

# Create dropdowns for user input
selected_values = {}
for param, values in unique_values.items():
    selected_values[param] = st.selectbox(f'Select {param}', values)

# Predict when the user clicks the button
if st.button('Predict'):
    user_data = pd.DataFrame(selected_values, index=[0])
    prediction = svm_model.predict(user_data)
    if prediction[0] == 0:
        st.success('The person is safe and less likely to get cancer.')
    else:
        st.warning('The person has to be treated ASAP! and is very likely to have cancer.')

