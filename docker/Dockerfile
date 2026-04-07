FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
WORKDIR /kaggle

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt update -y && \
    apt install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    sudo \
    git \
    npm \
    # for pyenv
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    # for opencv
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN npm install -g @openai/codex
RUN npm install -g @anthropic-ai/claude-code

ARG UID
ARG USER

RUN useradd $USER -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER

ARG PYTHON_VERSION=3.12.0
ENV PYENV_ROOT=/home/$USER/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash && \
    pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}

RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/home/$USER/.local/bin:$PATH"

RUN uv pip install --system --upgrade pip
RUN uv pip install --system --no-cache-dir \
    isort \
    ruff \
    numpy \
    pandas \
    polars \
    tqdm \
    jupyter \
    rich \
    seaborn \
    matplotlib \
    wandb \
    tyro \
    pyarrow \
    scikit-learn \
    lightgbm \
    huggingface_hub[cli] \
    h5py \
    litellm \
    rapidfuzz

RUN uv pip install --system --no-cache-dir vllm==0.17.0
RUN uv pip install --system --no-cache-dir lightning timm accelerate datasets sacrebleu randomname pypdfium2 kaggle