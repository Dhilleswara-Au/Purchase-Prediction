from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Model names and display names
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'K-Nearest Neighbors': 'knn.joblib',
    'SVM (Linear)': 'svm.joblib',
    'Kernel SVM (RBF)': 'kernel_svm.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'Random Forest': 'random_forest.joblib',
}

# Load scaler and models at startup
scaler = joblib.load('scaler.joblib')
models = {name: joblib.load(fname) for name, fname in MODEL_FILES.items()}
accuracies = joblib.load('accuracies.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = None
    user_input = {'Age': '', 'EstimatedSalary': ''}
    if request.method == 'POST':
        try:
            age = float(request.form['Age'])
            salary = float(request.form['EstimatedSalary'])
            user_input = {'Age': age, 'EstimatedSalary': salary}
            X = np.array([[age, salary]])
            X_scaled = scaler.transform(X)
            prediction_results = {}
            for model_name, model in models.items():
                pred = model.predict(X_scaled)[0]
                prediction = 'Will Buy' if pred == 1 else 'Will Not Buy'
                acc = accuracies.get(model_name, None)
                prediction_results[model_name] = {'prediction': prediction, 'accuracy': acc}
        except Exception as e:
            prediction_results = {'Error': str(e)}
    return render_template('index.html', prediction_results=prediction_results, user_input=user_input)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
