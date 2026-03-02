FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/RMFS01

# Runtime deps for scientific stack used by this project
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pythonProject5/requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install -r /tmp/requirements-docker.txt

COPY . /workspace/RMFS01

ENV PYTHONPATH=/workspace/RMFS01
WORKDIR /workspace/RMFS01/pythonProject5

# Default: quick benchmark (same parameters as AGENTS quick bench)
CMD ["python", "main.py", "--quick-bench", "--prefix", "demo01", "--bench-iters", "200", "--bench-seeds", "0", "--gammas", "0", "--bench-compare", "bench_baseline_demo01.json", "--bench-tol", "0.001"]
