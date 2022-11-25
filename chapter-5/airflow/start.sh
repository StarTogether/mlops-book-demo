#!/bin/bash
set -ex

docker build -t apache/airflow:2.3.3_custom .

docker-compose up -d