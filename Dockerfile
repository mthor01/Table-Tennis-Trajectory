# Container with everything needed to run the pipeline and the tests.
#
#   docker build -t table-tennis-trajectory .
#   docker run --rm -v "$(pwd):/app" table-tennis-trajectory ./run_pipeline.sh --quick
#
# Mounting the repository to /app makes the results in output/ appear on the host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# libglib2.0-0 is needed by OpenCV (headless) and PySceneDetect
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh", "--quick"]
