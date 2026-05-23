import mysql.connector
import pandas as pd


def connect_database():

    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='YOUR_PASSWORD',
        database='HRAnalytics'
    )

    return connection


def fetch_employee_data():

    connection = connect_database()

    query = "SELECT * FROM employees"

    df = pd.read_sql(query, connection)

    connection.close()

    return df