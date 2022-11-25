import mlflow
import click

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

@click.command(help="Run train")
@click.option("--data_path")
def task(data_path):
    with mlflow.start_run() as mlrun:
        final_pdf = pd.read_csv(data_path)

        # 数据整理
        data = final_pdf.drop('num', axis=1)
        label = final_pdf['num']
        x = data.values
        y = label.values

        # 数据切分，分为训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # 选择分类算法，此处选择随机森林，你也可以切换算法做实验，mlflow正好可以帮你完成切换的算法及生成的模型记录
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # 记录模型，方便在MLflow ui中查看模型
        mlflow.sklearn.log_model(model, "model")

        # 测试数据预测
        y_pred = model.predict(x_test)

        # 指标计算，并将所有指标记载下来
        report = classification_report(y_test, y_pred, output_dict=True)
        clsf_report = pd.DataFrame(report).transpose()
        clsf_report.to_csv('classification_report.csv', index=True)
        mlflow.log_artifact('classification_report.csv')
        print(report)

        # 总体的大指标，记录下来，可在mlflow ui上直观可视
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # 记录模型指标，方便在今后换模型的时候进行对比
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('f1', f1)

        # 记录训练全过程，选用的算法
        mlflow.log_artifact('train.py')


if __name__ == '__main__':
    task()
