from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Initialize Flask app
app = Flask(__name__)


# 1. Load and preprocess the dataset
def load_and_preprocess_data():
    file_path = 'C:/Users/ShivaRam/Downloads/synthetic_loan_data.csv'  # Update with your dataset path
    loan_data = pd.read_csv(file_path)

    # Encode categorical variables
    loan_data['Gender'] = loan_data['Gender'].map({'Male': 1, 'Female': 0})
    loan_data['Married'] = loan_data['Married'].map({'Yes': 1, 'No': 0})
    loan_data['Education'] = loan_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    loan_data['Self_Employed'] = loan_data['Self_Employed'].map({'Yes': 1, 'No': 0})
    loan_data['Loan_Status'] = loan_data['Loan_Status'].map({'Y': 1, 'N': 0})

    return loan_data


# 2. Balance the dataset using SMOTE
def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# 3. Compare algorithms and train the best model
def compare_and_train_best_model(data):
    # Define features and target variable
    X = data.drop(columns=['Loan_Status'])
    y = data['Loan_Status']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance the training set using SMOTE
    X_train_resampled, y_train_resampled = balance_classes(X_train_scaled, y_train)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                                        random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Compare models
    results = []
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        # Train the model
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })

        # Update best model
        if f1 > best_f1:
            best_model = model
            best_f1 = f1

    # Create a DataFrame for results
    results_df = pd.DataFrame(results)
    print("Model Comparison Results:")
    print(results_df)

    # Save the best model and scaler
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(best_model, 'best_loan_eligibility_model.pkl')
    print(f"Best Model: {results_df.loc[results_df['F1 Score'].idxmax(), 'Model']} saved successfully!")

    # Plot confusion matrix for the best model
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Eligible', 'Eligible'],
                yticklabels=['Not Eligible', 'Eligible'])
    plt.title("Confusion Matrix of Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Train the model if files are missing
if not os.path.exists('best_loan_eligibility_model.pkl') or not os.path.exists('scaler.pkl'):
    print("Training models and selecting the best...")
    data = load_and_preprocess_data()
    compare_and_train_best_model(data)

# Load the trained model and scaler
model = joblib.load('best_loan_eligibility_model.pkl')
scaler = joblib.load('scaler.pkl')


# 4. Flask routes for prediction
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve user inputs from the form
        gender = 1 if request.form['gender'] == 'Male' else 0
        married = 1 if request.form['married'] == 'Yes' else 0
        education = 1 if request.form['education'] == 'Graduate' else 0
        self_employed = 1 if request.form['self_employed'] == 'Yes' else 0
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        credit_history = int(request.form['credit_history'])
        credit_score = float(request.form['credit_score'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Credit_Score': credit_score
        }])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict eligibility
        prediction = model.predict(input_scaled)[0]
        result = "Eligible for Loan" if prediction == 1 else "Not Eligible for Loan"

        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
