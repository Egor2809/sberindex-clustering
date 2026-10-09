FROM python:3.10-slim
ARG SOURCE_REVISION
ENV SOURCE_REVISION=${SOURCE_REVISION}
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libarchive-tools && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-models.txt -r requirements-visuals.txt
COPY . .
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 CUDA_VISIBLE_DEVICES=-1
CMD ["python", "scripts/run_contest.py", "--help"]
