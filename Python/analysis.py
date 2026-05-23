import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PROJECT PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(CURRENT_DIR, '..', 'dataset')
REPORTS_DIR = os.path.join(CURRENT_DIR, '..', 'reports')

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

    csv_path = os.path.join(DATASET_DIR, csv_files[0])

    print(f"\nLoading Dataset: {csv_files[0]}")

    return pd.read_csv(csv_path)


# =========================================================
# BASIC ANALYSIS
# =========================================================

def basic_analysis(df):
    print("\nFIRST 5 ROWS")
    print(df.head())

    print("\nDATASET INFO")
    print(df.info())

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    print(f"\nTotal Employees: {len(df)}")

    attrition_count = df['Attrition'].value_counts()

    print("\nAttrition Count")
    print(attrition_count)

    attrition_rate = (
        attrition_count['Yes'] / len(df)
    ) * 100

    print(f"\nAttrition Rate: {attrition_rate:.2f}%")

    avg_salary = df['MonthlyIncome'].mean()

    print(f"\nAverage Salary: {avg_salary:.2f}")


# =========================================================
# SAVE CHART
# =========================================================

def save_chart(chart_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    filepath = os.path.join(
        REPORTS_DIR,
        f'{chart_name}_{timestamp}.png'
    )

    plt.savefig(filepath)

    print(f"\nChart Saved: {filepath}")


# =========================================================
# VISUALIZATIONS
# =========================================================

def employees_by_department(df):
    department_count = df['Department'].value_counts()

    plt.figure(figsize=(10, 6))

    department_count.plot(kind='bar')

    plt.title('Employees by Department')
    plt.xlabel('Department')
    plt.ylabel('Employees')

    plt.tight_layout()

    save_chart('employees_by_department')

    plt.show()


def salary_distribution(df):
    plt.figure(figsize=(10, 6))

    plt.hist(df['MonthlyIncome'], bins=20)

    plt.title('Salary Distribution')
    plt.xlabel('Monthly Income')
    plt.ylabel('Employees')

    plt.tight_layout()

    save_chart('salary_distribution')

    plt.show()


def attrition_vs_overtime(df):
    overtime_analysis = pd.crosstab(
        df['OverTime'],
        df['Attrition']
    )

    print("\nAttrition vs Overtime")
    print(overtime_analysis)

    overtime_analysis.plot(kind='bar')

    plt.title('Attrition vs Overtime')
    plt.xlabel('Overtime')
    plt.ylabel('Employees')

    plt.tight_layout()

    save_chart('attrition_vs_overtime')

    plt.show()


def job_satisfaction_analysis(df):
    job_satisfaction = (
        df.groupby('JobSatisfaction')['EmployeeNumber']
        .count()
    )

    plt.figure(figsize=(8, 5))

    job_satisfaction.plot(kind='bar')

    plt.title('Job Satisfaction Analysis')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Employees')

    plt.tight_layout()

    save_chart('job_satisfaction')

    plt.show()


def correlation_heatmap(df):
    numeric_df = df.select_dtypes(
        include=['int64', 'float64']
    )

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(16, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.1f',
        cmap='coolwarm'
    )

    plt.title('Correlation Heatmap')

    plt.tight_layout()

    save_chart('correlation_heatmap')

    plt.show()


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

def train_ml_model(df):

    ml_df = df.copy()

    ml_df['Attrition'] = ml_df['Attrition'].map({
        'Yes': 1,
        'No': 0
    })

    features = [
        'Age',
        'MonthlyIncome',
        'DistanceFromHome',
        'JobSatisfaction',
        'WorkLifeBalance',
        'YearsAtCompany'
    ]

    X = ml_df[features]

    y = ml_df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

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


# =========================================================
# EXPORT CLEANED DATA
# =========================================================

def export_dataset(df):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    filename = os.path.join(
        REPORTS_DIR,
        f'cleaned_hr_data_{timestamp}.csv'
    )

    df.to_csv(filename, index=False)

    print(f"\nDataset Exported: {filename}")


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():

    try:
        df = load_dataset()

        basic_analysis(df)

        employees_by_department(df)

        salary_distribution(df)

        attrition_vs_overtime(df)

        job_satisfaction_analysis(df)

        correlation_heatmap(df)

        train_ml_model(df)

        export_dataset(df)

        print("\nProject Executed Successfully.")

    except Exception as error:
        print(f"\nERROR: {error}")


if __name__ == '__main__':
    main()