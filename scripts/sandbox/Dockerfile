ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim
RUN pip install --no-cache-dir scipy pandas numpy matplotlib sympy statsmodels
RUN useradd -m sandbox
USER sandbox
WORKDIR /workspace
ENTRYPOINT ["python", "/workspace/code.py"]
