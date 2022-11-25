import mlflow
import click
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

current_dir = os.path.dirname(__file__)

@click.command(help="Run feature")
@click.option("--file_path")
def task(file_path):
    with mlflow.start_run() as mlrun:
        heart_disease = pd.read_csv(file_path)
        # 数据清理
        heart_disease.drop(['id', 'dataset'], axis=1, inplace=True)

        # 缺失值处理
        heart_disease["trestbps"].fillna(heart_disease["trestbps"].mean(), inplace=True)
        heart_disease["chol"].fillna(heart_disease["chol"].mean(), inplace=True)
        heart_disease["fbs"].fillna(heart_disease["fbs"].mode()[0], inplace=True)
        heart_disease["restecg"].fillna(heart_disease["restecg"].mode()[0], inplace=True)
        heart_disease["thalch"].fillna(heart_disease["thalch"].mean(), inplace=True)
        heart_disease["exang"].fillna(heart_disease["exang"].mode()[0], inplace=True)
        heart_disease["oldpeak"].fillna(heart_disease["oldpeak"].mean(), inplace=True)
        heart_disease["slope"].fillna(heart_disease["slope"].mode()[0], inplace=True)
        heart_disease["ca"].fillna(heart_disease["ca"].mean(), inplace=True)
        heart_disease["thal"].fillna(heart_disease["thal"].mode()[0], inplace=True)

        # one hot 编码
        cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(heart_disease[cat_features].values)

        matrix = enc.transform(heart_disease[cat_features].values).toarray()
        feature_labels = np.array(enc.categories_).ravel()

        col_names = []
        for col in cat_features:
            for val in heart_disease[col].unique():
                col_names.append("{}_{}".format(col, val))

        onehot_pdf = pd.DataFrame(data=matrix, columns=col_names, dtype=int)
        remian_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
        remain_pdf = heart_disease[remian_features]
        final_pdf = pd.concat([onehot_pdf, remain_pdf], axis=1)

        csv_file_path = os.path.join(current_dir, 'data/data.csv')
        final_pdf.to_csv(csv_file_path)

        mlflow.log_artifacts("data", artifact_path="data")


if __name__ == '__main__':
    task()
