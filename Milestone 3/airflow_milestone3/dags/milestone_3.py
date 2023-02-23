from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from milestone_1 import clean_and_transform 
from milestone_2 import add_feature
from sqlalchemy import create_engine
import pandas as pd
from dashbord import create_dashboard




def load_to_postgres(filename1 , filename2 ): 
    df = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/uk_accidents')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    try:
        df.to_sql(name = 'UK_Accidents_1999',con = engine,if_exists='fail')
        df2.to_sql(name = 'lookup_table',con = engine,if_exists='fail')
    except ValueError:
        print('Table already exists')

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'uk_accidents_etl_pipeline',
    default_args=default_args,
    description='uk accidents etl pipeline',
)
with DAG(
    dag_id = 'uk_accidents_etl_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['uk_accidents-pipeline'],
)as dag:
    task1= PythonOperator(
        task_id = 'milestone_1',
        python_callable = clean_and_transform,
        op_kwargs={
            "filename": '/opt/airflow/data/Accidents_UK.parquet'
        } ,
    )
    task2= PythonOperator(
        task_id = 'milestone_2',
        python_callable = add_feature,
        op_kwargs={
            "computed_dataset": "/opt/airflow/data/Accidents_UK_API_CALL.csv",
            "cleaned_dataset": "/opt/airflow/data/accident_cleaned_stage1.csv"
        },
    )
    task3=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "filename1": "/opt/airflow/data/accident_cleaned_stage2.csv",
            "filename2": "/opt/airflow/data/lookup_table.csv"
        },
    )
    task4 = PythonOperator(
        task_id = 'create_dashboard',
        python_callable= create_dashboard,
        op_kwargs={
            "filename1": "/opt/airflow/data/accident_cleaned_stage0.csv",
            "filename2": "/opt/airflow/data/Accidents_UK.parquet"
        },
    )
    task1 >> task2 >> task3 >> task4