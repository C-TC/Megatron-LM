#!/bin/bash

BASE_PATH="$(cd "$(dirname "$0")" && pwd)"

export HF_DATASETS_CACHE="${BASE_PATH}/.cache"
python -c 'from datasets import load_dataset; ds = load_dataset("stas/oscar-en-10k", split="train", keep_in_memory=False); ds.to_json(f"data/oscar-en-10k.jsonl", orient="records", lines=True, force_ascii=False)'

python ${BASE_PATH}/../tools/preprocess_data.py \
       --input ${BASE_PATH}/data/oscar-en-10k.jsonl \
       --output-prefix ${BASE_PATH}/data/oscar-en-10k-meg-llama\
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${BASE_PATH}/tokenizer/tokenizer.model \
       --workers 32 \
       --append-eod
