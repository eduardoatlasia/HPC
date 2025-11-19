# ------------------------------------------------------------
# 1. BASE NGC: PyTorch + CUDA + cuDNN já configurados
# ------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:23.10-py3

# evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive
# e padroniza para o horario de SP (pq Sao Carlos)
ENV TZ=America/Sao_Paulo

# ------------------------------------------------------------
# 2. DEPENDÊNCIAS DE SISTEMA ÚTEIS
#    (a imagem já tem muita coisa, aqui é só o essencial extra)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# garantir alias "python"
RUN ln -s /usr/bin/python3 /usr/bin/python || true

# ------------------------------------------------------------
# 3. BIBLIOTECAS PYTHON PARA LLM / Qwen3
# ------------------------------------------------------------
RUN pip install --upgrade pip

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
# 4. WORKDIR DO PROJETO
# ------------------------------------------------------------
WORKDIR /workspace/HPC

# ------------------------------------------------------------
# 5. VOLUMES PARA INPUT / OUTPUT
#    (serão montados via -v no docker run)
# ------------------------------------------------------------
VOLUME ["/workspace/HPC/input"]
VOLUME ["/workspace/HPC/output"]

# ------------------------------------------------------------
# 6. COMANDO PADRÃO
# ------------------------------------------------------------
CMD ["/bin/bash"]
