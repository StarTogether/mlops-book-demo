#!/bin/bash
set -ex

#
git clone https://github.com/ybz21/mlflow-docker-compose.git
cd mlops-book-demo/chapter-5/mlflow/mlflow-docker-compose
docker-compose up -d --build
cd -


export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

mkdir -p ~/.aws
cat <<EOF > ~/.aws/credentials
[default]
aws_access_key_id=minio
aws_secret_access_key=minio123
EOF

mlflow run heart_disease_cls