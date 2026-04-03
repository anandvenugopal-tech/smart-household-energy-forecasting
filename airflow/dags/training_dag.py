from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "anand",
    "start_date": datetime(2026, 3, 27),
    "retries": 1
}

with DAG(
    dag_id = "energy_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False
) as dag:
    ingest = BashOperator(
        task_id = 'ingest_data',
        bash_command = "python src/ingest.py"
    )
    clean = BashOperator(
        task_id = 'clean_data',
        bash_command = "python src/clean_data.py"
    )
    feature = BashOperator(
        task_id = "feature_engineering",
        bash_command = "python src/build_features.py"
    )
    train = BashOperator(
        task_id = "train_model",
        bash_command = "python src/train_ml.py"
    )
    ingest >> clean >> feature >> train

    