# ------------------------------------------------------------
# 1. BASE NVIDIA OFICIAL (CUDA 12.1 + cuDNN8)
# ------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ------------------------------------------------------------
# 2. DEPENDÊNCIAS DO SISTEMA
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------
# 3. PACOTES PYTHON PARA TREINO DE LLM
# ------------------------------------------------------------
RUN pip install --upgrade pip

RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    transformers>=4.44.0 \
    datasets>=2.20.0 \
    accelerate>=0.33.0 \
    peft>=0.11.0 \
    sentencepiece \
    evaluate \
    huggingface_hub \
    protobuf

# ------------------------------------------------------------
# 4. WORKDIR
# ------------------------------------------------------------
WORKDIR /workspace/HPC

# ------------------------------------------------------------
# 5. DECLARAÇÃO DOS VOLUMES
# ------------------------------------------------------------
# input → onde você coloca scripts/data (montado como volume externo)
# output → onde ficam logs e checkpoints
VOLUME ["/workspace/HPC/input"]
VOLUME ["/workspace/HPC/output"]

# ------------------------------------------------------------
# 6. COMANDO PADRÃO
# ------------------------------------------------------------
CMD ["/bin/bash"]
