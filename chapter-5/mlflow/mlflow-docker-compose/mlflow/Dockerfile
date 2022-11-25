FROM python:3.9-slim-buster
# Install python packages
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python