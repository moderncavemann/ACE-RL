# syntax=docker/dockerfile:1.7

FROM python:3.8.20-slim-bullseye AS abides-markets

ARG ABIDES_REPOSITORY=https://github.com/jpmorganchase/abides-jpmc-public.git
ARG ABIDES_COMMIT=f9cbe51342b7dedd9587e4e069040d68a5c6477f

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        pip==23.3.2 \
        setuptools==68.2.2 \
        wheel==0.41.3 \
        Cython==0.29.37

RUN git clone --filter=blob:none "${ABIDES_REPOSITORY}" /opt/abides \
    && git -C /opt/abides checkout "${ABIDES_COMMIT}" \
    && test "$(git -C /opt/abides rev-parse HEAD)" = "${ABIDES_COMMIT}"

# Ray and RLlib are not needed by the event kernel or market simulator. Keep
# them out of the first target so dependency failures remain attributable.
RUN python -m pip install --no-cache-dir \
        coloredlogs==15.0.1 \
        numpy==1.22.0 \
        pandas==1.2.4 \
        psutil==5.8.0 \
        scipy==1.10.0 \
        tqdm==4.61.1 \
    && python -m pip install --no-cache-dir --no-deps \
        -e /opt/abides/abides-core \
        -e /opt/abides/abides-markets

LABEL org.opencontainers.image.source="${ABIDES_REPOSITORY}" \
      org.opencontainers.image.revision="${ABIDES_COMMIT}"

WORKDIR /workspace
COPY smoke_rmsc.py /opt/ace-abides/smoke_rmsc.py

ENTRYPOINT ["python", "/opt/ace-abides/smoke_rmsc.py"]
CMD ["--help"]


FROM abides-markets AS abides-rmsc04

# RMSC04 imports the legacy pomegranate order-size model. Pin the release used
# by the archived ABIDES repository instead of accepting the incompatible 1.x
# API that caused the previous GeneralMixtureModel import failure.
RUN python -m pip install --no-cache-dir pomegranate==0.14.5

CMD ["--config", "rmsc04"]


FROM abides-rmsc04 AS abides-gym

# ABIDES-Gym imports Ray from its package initializer even when RLlib is not
# used by ACE-RL, so the archived compatible versions must be present.
COPY requirements-gym.lock /opt/ace-abides/requirements-gym.lock
RUN python -m pip install --no-cache-dir \
        setuptools==57.5.0 \
        wheel==0.37.1 \
    && python -m pip install --no-cache-dir \
        --requirement /opt/ace-abides/requirements-gym.lock \
    && python -m pip install --no-cache-dir --no-deps \
        -e /opt/abides/abides-gym

COPY smoke_gym.py /opt/ace-abides/smoke_gym.py
ENTRYPOINT ["python", "/opt/ace-abides/smoke_gym.py"]
CMD []


FROM abides-rmsc04 AS abides-ace

# ACE-RL keeps its adapter outside the archived upstream tree. The adapter uses
# the official Kernel, ExchangeAgent, TradingAgent, and OrderBook without
# importing Ray or RLlib into this runtime.
COPY ace_abides /opt/ace-abides/ace_abides
COPY smoke_multiagent.py /opt/ace-abides/smoke_multiagent.py
COPY smoke_routing.py /opt/ace-abides/smoke_routing.py
COPY validate_economics.py /opt/ace-abides/validate_economics.py
ENV PYTHONPATH=/opt/ace-abides

ENTRYPOINT ["python", "/opt/ace-abides/smoke_multiagent.py"]
CMD []
