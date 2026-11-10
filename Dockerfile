FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
WORKDIR /opt/rankforge
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/huggingface \
    RANKFORGE_HOST=0.0.0.0 RANKFORGE_PORT=8000 \
    RANKFORGE_DATASET_DIR=/resources/dataset \
    RANKFORGE_DENSE_MODEL_DIR=/resources/models/dense \
    RANKFORGE_RERANK_MODEL_DIR=/resources/models/reranker \
    RANKFORGE_DENSE_INDEX=/resources/indexes/scifact-minilm-linux.npz \
    RANKFORGE_POLICY_DIR=/resources/policy
COPY docker/requirements.lock /opt/rankforge/docker/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps -r docker/requirements.lock && python -m pip check \
    && groupadd --gid 10001 rankforge && useradd --uid 10001 --gid 10001 --no-create-home rankforge
COPY app/ ./app/
COPY docker/*.py ./docker/
USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 CMD ["python", "docker/healthcheck.py"]
CMD ["python", "docker/entrypoint.py"]
