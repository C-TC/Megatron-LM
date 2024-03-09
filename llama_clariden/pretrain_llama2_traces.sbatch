#!/bin/bash -l
#SBATCH --job-name="megatron-llama2-pretrain"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=nvgpu
#SBATCH --time=00:20:00            # total run time limit (HH:MM:SS)


# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed training variables
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Parallelism variables
TP=4
PP=1
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network size variables
MODEL_SIZE=7

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=4096

# Paths
BASE_PATH="/iopsstor/scratch/cscs/akubicek/training/Megatron-LM/llama_clariden" # FIXME
cd ${BASE_PATH}
SRC_PATH=${BASE_PATH}/../pretrain_gpt.py

LOG_NAME=llama2-${MODEL_SIZE}b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

DATA_PATH=${BASE_PATH}/data/oscar-en-10k-meg-llama_text_document
DATA_CACHE_PATH="${BASE_PATH}/data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
TOKENIZER_PATH=${BASE_PATH}/tokenizer/tokenizer.model

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank \${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       "    

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --position-embedding-type rope \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --make-vocab-size-divisible-by 1 \
       --norm-epsilon ${NORM_EPS} \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
       "

LOGGING_ARGS=" \
       --log-throughput \
       --timing-log-level 0 \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-world-size-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 1 \
       --global-batch-size 32 \
       --train-iters 10000 \
       --log-interval 1 \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --optimizer adam \
       --exit-interval 2 \
       --recompute-activations \
       --recompute-granularity selective \
       --tensorboard-dir ${BASE_PATH} \
       "

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       --finetune \
       --no-load-optim \
       --no-load-rng \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       "

VALIDATION_ARGS=" \
       --eval-interval 1000 \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --data-cache-path ${DATA_CACHE_PATH} \
       "

CMD="\
       ${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       "

NSYS="\
       nsys profile \
       --trace='nvtx,cuda,mpi' \
       --cuda-memory-usage='true' \
       --output='${BASE_PATH}/trace_%q{SLURM_NODEID}_%q{SLURM_PROCID}.nsys-rep' \
       "

RUN="${NSYS}${CMD}"

srun --mpi=pmi2 --environment=megatron bash -c "
export NODE_RANK=\${SLURM_NODEID}
echo ${RUN}
${RUN} 2>&1 | tee ${LOG_PATH}
"