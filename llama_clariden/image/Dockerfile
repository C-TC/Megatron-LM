FROM nvcr.io/nvidia/pytorch:24.01-py3
 
# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive
 
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        automake \
        autoconf \
        libtool \
        wget \
        libpmi2-0-dev \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install datasets accelerate evaluate tokenizers sentencepiece transformers nltk deepspeed flash-attn nvtx
