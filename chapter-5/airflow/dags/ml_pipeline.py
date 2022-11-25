from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

# DAG属性定义
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(31),
    # 填入邮箱，方便失败、重试时发送邮件
    'email': ['mlops@4paradigm.com'],
    # 失败时发邮件告警
    'email_on_failure': True,
    'email_on_retry': False,
    # 重试次数
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),bu
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    'trigger_rule': 'all_success'
}

# 定义dag
dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='A simple Machine Learning pipeline for spam message classification',
    schedule_interval=timedelta(days=1),
)

# 下载训练数据环节，用于下载标记好的垃圾短信、邮件数据，每天都有增量数据加入
download_train_data = BashOperator(
    task_id='download_train_data',
    bash_command='python3 /opt/airflow/dags/download.py --mode train',
    dag=dag,
)

# 训练环节，基于新的数据进行训练，得到更鲁棒的模型
train = BashOperator(
    task_id='train',
    depends_on_past=False,
    bash_command='python3 /opt/airflow/dags/train.py',
    retries=3,
    dag=dag,
)

# 下载推理数据环节，用于拉取线上业务需要推理的数据
download_inference_data = BashOperator(
    task_id='download_inference_data',
    depends_on_past=False,
    bash_command='python3 /opt/airflow/dags/download.py --mode inference',
    retries=3,
    dag=dag,
)

# 推理环节，用于对每天需要推理的业务数据进行推理
inference = BashOperator(
    task_id='inference',
    depends_on_past=False,
    bash_command='python3 /opt/airflow/dags/inference.py',
    retries=3,
    dag=dag,
)

# download images runs first, then train, then down.
download_train_data >> train >> download_inference_data >> inference
