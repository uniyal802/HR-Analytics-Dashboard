import matplotlib.pyplot as plt
import seaborn as sns


def plot_salary_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(df['MonthlyIncome'], bins=20)

    plt.title('Salary Distribution')

    plt.xlabel('Monthly Income')

    plt.ylabel('Employees')

    plt.show()