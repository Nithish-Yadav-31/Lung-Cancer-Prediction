from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained SVM model
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

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html', unique_values=unique_values)

# Define a route to handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {}

        # Get selected values from the form and convert them to integers
        for param in unique_values.keys():
            input_data[param] = int(request.form.get(param))

        # Create a DataFrame from the user's selected values
        user_data = pd.DataFrame(input_data, index=[0])

        # Perform prediction using the loaded SVM model
        prediction = svm_model.predict(user_data)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
