import mlflow
import os

import click

current_dir = os.path.dirname(__file__)


@click.command()
def workflow():
    with mlflow.start_run() as active_run:
        print("Launching 'download'")
        download_run = mlflow.run(".", "download", parameters={})
        download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
        file_path = os.path.join(download_run.info.artifact_uri, "data/raw.csv")
        print("Finish 'download'")

        print("Launching 'feature'")
        file_path = os.path.join(current_dir, "data/raw.csv")
        process_run = mlflow.run(".", "feature", parameters={"file_path": file_path})
        process_run = mlflow.tracking.MlflowClient().get_run(process_run.run_id)
        print("Finish 'feature'")

        print("Launching 'train'")
        data_path_uri = os.path.join(current_dir, "data/data.csv")
        train_run = mlflow.run(".", "train", parameters={"data_path": data_path_uri})
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)
        print("Finish 'train'")


if __name__ == '__main__':
    workflow()
