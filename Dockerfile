FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-fraudsim.txt /app/requirements-fraudsim.txt
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements-fraudsim.txt

COPY fraudsim /app/fraudsim
COPY configs /app/configs

CMD ["uvicorn", "fraudsim.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
