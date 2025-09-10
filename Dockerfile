FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/cache/hf \
    HF_HOME=/cache/hf \
    SUMMARIZER_PATH=/app/t5_summarize.py \
    FIREBASE_CRED_PATH=/run/secrets/firebase.json \
    API_DATA_LANG=ko \
    MIN_BUFFER_SIZE=5 \
    CLEANUP_INTERVAL_SEC=3600 \
    DELETE_OLDER_THAN_DAYS=180 \
    TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential git tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Torch: 일반 PyPI → 실패 시 CPU 전용 인덱스로 재시도
RUN pip install --no-cache-dir torch==2.3.1 \
 || pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

COPY main.py t5_summarize.py /app/

RUN useradd -m appuser && chown -R appuser /app
USER appuser

VOLUME ["/cache"]

ENTRYPOINT ["python", "-u", "main.py"]
