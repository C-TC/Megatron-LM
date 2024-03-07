# Pretraining LLama2 on Clariden

## Container
Dockerfile & toml file in `image` dir.

Dockerfile based on NGC PyTorch container.


## Tokenizer

Insider container, cd to `llama_clariden` folder, execute the following: (make sure you have logged into huggingface in cli)

```bash
mkdir -p tokenizer && cd tokenizer
huggingface-cli download meta-llama/Llama-2-7b tokenizer.model --local-dir .
cd ..
```

## Dataset

> Notice: set `HF_DATASETS_CACHE` to a dir in SCRATCH memory to download faster, and no exceeding user dir storage limit.

Insider container, cd to `llama_clariden` folder, execute the `preprocess.sh` to download the dataset.

## Pretrain SLURM script

Some key variable in `pretrain_llama2.sbatch`:
- `--nodes`: number of nodes in cluster
- `MODEL_SIZE`: 7B/13B/70B

Before submitting, change `BASE_PATH` in sbatch file to correct location.

During configuration, only 1 process on each node. Then torchrun will spawn the rest processes, so that 1 process controls one GPU, this is suggested by torch.distributed team.

`sbatch pretrain_llama2.sbatch` to submit scaling job.

## Traces
Custom `nvtx` annotations were added to the megatron source code to navigate the traces better. 

`sbatch pretarin_llama2_traces.sbatch` to submit job including tracing (Nvidia Nsight Systems). This will generate `.nsys-rep` file that can be downloaded and opened with the Nvidia Nsight Systems GUI to investigate traces. Configuration of what to trace is defined at the end of the `.sbatch` script.
