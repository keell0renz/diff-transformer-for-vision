FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /diff-transformer-for-vision

COPY . .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_HTTP_TIMEOUT=120

RUN uv venv

RUN uv sync

CMD ["/bin/bash"]