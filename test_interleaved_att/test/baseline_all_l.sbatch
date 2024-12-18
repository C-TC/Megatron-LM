#!/bin/bash -l
#SBATCH --job-name="iv_att"
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=normal
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00            # total run time limit (HH:MM:SS)


GLOBAL_VARS="\
# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

export OMP_NUM_THREADS=1

# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


export WANDB_API_KEY=d70812b5d01c206c64129377acd778f3871cbf20
"

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
GBS=64
ITERS=20000

# 525M tokens / job
# 8 jobs -> 4B tokens -> 7500 iters 

# Network size variables
MODEL_SIZE=8

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif   [[ ${MODEL_SIZE} == 8 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=8; NUM_LAYERS=32; FFN_HIDDEN_SIZE=14336; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.0
MAX_LR=5e-4
MIN_LR=1e-5
MAX_SEQ_LEN=8192
MAX_POSITION_EMBEDDINGS=131072

# Paths
# job submit path
BASE_PATH=${SLURM_SUBMIT_DIR}
source /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/source_me.sh
cd ${BASE_PATH}
SRC_PATH=${MEGATRON_PATH}/pretrain_gpt.py

JOB_ID=${SLURM_JOB_ID}
LOG_NAME=iv_att_model${MODEL_SIZE}_baseline_all_l
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}
TB_PATH="${BASE_PATH}/log/${LOG_NAME}/tensorboard"
mkdir -p ${TB_PATH}

DATA_CACHE_PATH="${BASE_PATH}/data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
mkdir -p ${SAVE_PATH}

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --rdzv_id=${SLURM_JOBID} \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       --overlap-grad-reduce \
       --overlap-param-gather \
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
       --make-vocab-size-divisible-by 64 \
       --norm-epsilon ${NORM_EPS} \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
	--attention-softmax-in-fp32 \
	--rotary-base 500000 \
	--rotary-percent 1.0 \
	--use-rope-scaling \
       --window-size 4096 \
       --local-attention-every-n-layers 1 \
       "

LOGGING_ARGS=" \
       --log-throughput \
       --log-params-norm \
       --timing-log-level 0 \
       --log-timers-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-progress \
       --log-validation-ppl-to-tensorboard \
       --wandb-project iv_att \
       --wandb-exp-name ${LOG_NAME} \
       --wandb-save-dir ${TB_PATH} \
       --tensorboard-dir ${TB_PATH} \
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
    --micro-batch-size 2 \
    --global-batch-size ${GBS} \
    --train-iters ${ITERS} \
    --log-interval 4 \
    --disable-bias-linear \
    --cross-entropy-loss-fusion \
    --use-flash-attn \
    --optimizer adam \
    --train-sync-interval 4 \
    "

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.01 \
       "

LEARNING_RATE_ARGS=" \
       --lr-decay-style WSD \
	--lr-wsd-decay-style linear \
	--lr-wsd-decay-iters $(($ITERS/10)) \
	--lr-warmup-iters $(($ITERS/20)) \
    	--min-lr ${MIN_LR} \
       --lr ${MAX_LR} \
       "

CHECKPOINTING_ARGS=" \
       --save ${SAVE_PATH} \
       --load ${SAVE_PATH} \
       --save-interval 400 \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       "

VALIDATION_ARGS=" \
       --eval-interval 600 \
       --exit-signal-handler \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 9990,8,2 \
       --seq-length ${MAX_SEQ_LEN} \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model meta-llama/Meta-Llama-3-8B \
       --data-cache-path ${DATA_CACHE_PATH} \
       "

TE_ARGS=" \
    --transformer-impl transformer_engine \
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
       ${TE_ARGS} \
       "

RUN="${CMD}"

COUNTER_FILE="${TB_PATH}/job_counter.txt"
# Initialize counter if it doesn't exist
if [ ! -f "$COUNTER_FILE" ]; then
    echo "0" > "$COUNTER_FILE"
fi
# Read current count
COUNT=$(cat "$COUNTER_FILE")
# Maximum number of reruns (original run + 2 requeues = 3 total runs)
MAX_RERUNS=12
echo "Current run number: $((COUNT + 1))"


srun --mpi=pmi2 --environment=megatron bash -c "
export NODE_RANK=\${SLURM_NODEID}
${GLOBAL_VARS}
echo ${RUN}
${RUN} 2>&1 | tee ${LOG_PATH}
"
COUNT=$((COUNT + 1))
echo $COUNT > "$COUNTER_FILE"

if [ $COUNT -le $MAX_RERUNS ]; then
    echo "Requeuing job for run $((COUNT + 1))"
    scontrol requeue $SLURM_JOB_ID
fi
