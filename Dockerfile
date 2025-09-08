FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# base tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git tini && \
    rm -rf /var/lib/apt/lists/*

# project files
COPY requirements.txt /app/requirements.txt
COPY realtime_summarizer.py /app/realtime_summarizer.py

# deps (CPU용 PyTorch 별도)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /app/requirements.txt

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python","-u","/app/realtime_summarizer.py","--batch-size","1","--write"]
