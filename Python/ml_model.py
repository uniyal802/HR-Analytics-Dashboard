import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# PROJECT PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    CURRENT_DIR,
    '..',
    'dataset'
)

MODELS_DIR = os.path.join(
    CURRENT_DIR,
    '..',
    'models'
)

REPORTS_DIR = os.path.join(
    CURRENT_DIR,
    '..',
    'reports'
)

# Create folders if not exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    csv_files = [
        file for file in os.listdir(DATASET_DIR)
        if file.endswith('.csv')
    ]

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in dataset folder."
        )

    csv_path = os.path.join(
        DATASET_DIR,
        csv_files[0]
    )

    print(f"\nLoading Dataset: {csv_files[0]}")

    return pd.read_csv(csv_path)

# =========================================================
# PREPROCESS DATA
# =========================================================

def preprocess_data(df):

    # Convert Attrition
    df['Attrition'] = df['Attrition'].map({
        'Yes': 1,
        'No': 0
    })

    # Selected Features
    features = [
        'Age',
        'MonthlyIncome',
        'DistanceFromHome',
        'JobSatisfaction',
        'WorkLifeBalance',
        'YearsAtCompany'
    ]

    X = df[features]

    y = df['Attrition']

    return X, y

# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):

    model_path = os.path.join(
        MODELS_DIR,
        'hr_attrition_model.pkl'
    )

    joblib.dump(model, model_path)

    print(f"\nModel Saved Successfully:\n{model_path}")

# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    try:

        # Load dataset
        df = load_dataset()

        # Preprocess
        X, y = preprocess_data(df)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Train model
        model = train_model(
            X_train,
            y_train
        )

        # Evaluate model
        evaluate_model(
            model,
            X_test,
            y_test
        )

        # Save model
        save_model(model)

        print("\nML Pipeline Executed Successfully.")

    except Exception as error:

        print(f"\nERROR: {error}")

# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == '__main__':
    main()