FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system iforest \
    && adduser --system --ingroup iforest --no-create-home iforest

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY train.py infer.py ./

# The pipeline writes artifacts/ and scored.csv to the working directory.
RUN mkdir -p /app/artifacts /app/data && chown -R iforest:iforest /app

USER iforest

# Default: train on the bundled synthetic data and print the evaluation
# report. Mount real data at /app/data and override the command to use it,
# e.g.: docker run -v $PWD/data:/app/data <img> python infer.py --model
#   artifacts/iforest_v1.0.0.joblib --csv /app/data/telemetry.csv --out /app/data/scored.csv
CMD ["python", "train.py"]
