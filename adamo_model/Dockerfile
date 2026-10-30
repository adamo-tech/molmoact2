FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential python3.12-dev python3.12-venv libasound2t64 libglib2.0-0t64 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /venv

ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 AWS_CONFIG_FILE=/opt/adamo-model-runtime/runtime/aws_config

COPY molmoact2/adamo_model/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY adamo-model-runtime /opt/adamo-model-runtime
COPY molmoact2 /opt/molmoact2
RUN pip install --no-cache-dir --no-deps /opt/adamo-model-runtime /opt/molmoact2
