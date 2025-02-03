# To be able to import your function, you need to add the src/ directory to the Python path.
import pandas as pd
# For Label Encoding
from sklearn import preprocessing
from sqlalchemy import create_engine
from functions import extract_clean , transform , load_to_postgres
from fintech_dahsboard import create_dashboard

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Define the DAG
default_args = {
    "owner": "Shahenda",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'fintech_etl_pipeline',
    default_args=default_args,
    description='titanic etl pipeline',
)


with DAG(
    dag_id = 'fintech_etl_pipeline',
    schedule_interval = '@once', # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args = default_args,
    tags = ['fintech-pipeline'],
)as dag:
    # Define the tasks
    extract_clean = PythonOperator(
        task_id = 'extract_clean',
        python_callable = extract_clean,
        op_kwargs = {
            'file': '/opt/airflow/data/fintech_data_21_52_23665.csv'
        }
    )

    transform = PythonOperator(
        task_id = 'transform',
        python_callable = transform,
        op_kwargs = {
            'file': '/opt/airflow/data/fintech_clean.csv'
        }
    )

    load_to_db = PythonOperator(
        task_id = 'load_to_db',
        python_callable = load_to_postgres,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_transformed.csv'
        }
    )

    create_dashboard = PythonOperator(
        task_id = 'create_dashboard',
        python_callable = create_dashboard,
        op_kwargs = {
            'data_path': '/opt/airflow/data/fintech_transformed.csv'
        }
    )

    extract_clean >> transform >> load_to_db >> create_dashboard