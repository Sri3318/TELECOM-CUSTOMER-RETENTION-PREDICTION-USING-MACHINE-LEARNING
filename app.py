import numpy as np
import pandas as pd
import pickle
import sklearn
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the updated model and preprocessor
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gathering updated inputs
    tenure = int(request.form.get('tenure'))
    InternetService = request.form.get('InternetService')  # Categorical (DSL, Fiber, None)
    StreamingTV = request.form.get('StreamingTV')  # Categorical (Yes/No)
    StreamingMovies = request.form.get('StreamingMovies')  # Categorical (Yes/No)
    TechSupport = request.form.get('TechSupport')  # Categorical (Yes/No)
    Contract = request.form.get('Contract')  # Categorical (Month-to-month, One year, Two year)
    MonthlyCharges = float(request.form.get('MonthlyCharges'))
    TotalCharges = float(request.form.get('TotalCharges'))

    # Create DataFrame for prediction
    inputs = pd.DataFrame([[tenure, InternetService, StreamingTV, StreamingMovies, 
                            TechSupport, Contract, MonthlyCharges, TotalCharges]], 
                          columns=['tenure', 'InternetService', 'StreamingTV', 'StreamingMovies', 
                                   'TechSupport', 'Contract', 'MonthlyCharges', 'TotalCharges'])

    # Apply preprocessing manually
    input_processed = preprocessor.transform(inputs)  # Ensure preprocessor.pkl is correctly loaded

    # Make prediction
    prediction = model.predict(input_processed)
    churn_risk_scores = np.round(model.predict_proba(input_processed)[:, 1] * 100, 2)

    # Churn flag
    prediction_text = 'YES' if prediction == 1 else 'NO'

    return render_template('predict.html', prediction=prediction_text, churn_risk_scores=churn_risk_scores, inputs=request.form)

if __name__ == '__main__':
    app.run(debug=True)
