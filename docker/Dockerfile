# CPU / x86 deployment image: ONNX Runtime inference + full pipeline.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY perception ./perception
COPY scripts ./scripts
RUN pip install --no-cache-dir .
RUN useradd --create-home app && chown -R app /app
USER app
HEALTHCHECK CMD python -c "import perception.pipeline, onnxruntime" || exit 1
ENTRYPOINT ["python", "scripts/run_video.py"]
