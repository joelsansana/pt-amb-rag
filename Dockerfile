FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv export --no-dev --frozen --no-emit-project -o requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --target=/deps

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN useradd --create-home appuser
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages
COPY src ./src
COPY scripts ./scripts
COPY corpora.yaml ./corpora.yaml
COPY data/processed ./data/processed
ENV PYTHONPATH=/app/src
RUN mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "pt_amb_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
