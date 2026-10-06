# QENIVO on-premise service image (CPU). For the GPU image build from an NVIDIA CUDA runtime base
# and install the [gpu] extra; see packaging/README.md.
FROM python:3.12-slim
LABEL org.opencontainers.image.title="qenivo" \
      org.opencontainers.image.description="Sovereign certified optimisation engine: CLI, API and planner console" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN useradd --create-home --uid 10001 qenivo
WORKDIR /opt/qenivo
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir . && python -c "import qenivo, sys; print('qenivo', qenivo.__version__)"
USER qenivo
EXPOSE 8765
# Bind to all interfaces inside the container; publish only to the plant network you choose.
ENTRYPOINT ["qenivo"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765"]
