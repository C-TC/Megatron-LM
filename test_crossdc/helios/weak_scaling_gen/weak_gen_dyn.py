
TB_sizes = ["TB13", "TB70"]
for TB_size in TB_sizes:
    TP = 4
    DP = 1
    if TB_size == "TB13":
        layer_per_pp = 4
    else:
        layer_per_pp = 2
    for PP in [4,8,16,32]:
        num_nodes = TP * DP * PP // 4
        num_layers = PP * layer_per_pp
        GBS = PP * 2 * DP
        seq_len = 4096
        stages_per_dc = PP // 2
        for schedule in ['ud', 'wave']:
            if schedule == 'ud':
                profile_schedule = 'ZBH1'
                chunks = 1
            else:
                profile_schedule = 'ZBV'
                chunks = 2
            layer_per_virtual_stage = layer_per_pp // chunks

            job_str =  f"""\
#!/bin/bash -l
#SBATCH --job-name="dbg"
#SBATCH --nodes={num_nodes}
#SBATCH --gpus-per-node=4
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgethzdcs2024-gpu-gh200
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00

# Helios env
source $SCRATCH/cross_dc/env.sh

export FI_MR_CACHE_MONITOR=userfaultfd                                                                                                                                          
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=256
export NCCL_CROSS_NIC=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET="AWS Libfabric"
export MPICH_GPU_SUPPORT_ENABLED=0
export CXI_FORK_SAFE="1"
export CXI_FORK_SAFE_HP="1"
export FI_CXI_DISABLE_CQ_HUGETLB="1"
export NCCL_NET_GDR_LEVEL="3"

# cdc latency injection
# export TORCH_NCCL_ENABLE_TIMING=1
# export TORCH_NCCL_CUDA_EVENT_CACHE=1

# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=16
# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed training variables
NNODES=${{SLURM_NNODES}}
GPUS_PER_NODE=4
GPU_NUM=$((${{GPUS_PER_NODE}}*${{NNODES}}))
WORLD_SIZE=$((${{GPUS_PER_NODE}}*${{NNODES}}))
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Parallelism variables
TP={TP}
PP={PP}
DP=$((${{GPU_NUM}}/${{TP}}/${{PP}}))

# Network size variables
MODEL_SIZE={TB_size}

if   [[ ${{MODEL_SIZE}} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=64; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == "tiny_test" ]]; then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8; NUM_LAYERS=8; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == "tiny_ault" ]]; then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=8; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == "TB13" ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == "TB70" ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${{MODEL_SIZE}}"; exit 1
fi

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN={seq_len}
MAX_POSITION_EMBEDDINGS=${{MAX_SEQ_LEN}}
NUM_LAYERS={num_layers}

# Paths
BASE_PATH="$SCRATCH/cross_dc/Megatron-LM/test_crossdc/helios/weak_scaling_gen"
SCRIPT_NAME=$(basename "$0")
SCRIPT_BASENAME="${{SCRIPT_NAME%.*}}"

# get job id
JOB_ID=${{SLURM_JOB_ID}}

# create Job log path = script name + job id
JOB_LOG_PATH="${{BASE_PATH}}/job_logs/weak_{TB_size}_{PP}_{schedule}"


source ${{BASE_PATH}}/../source_me.sh

# switch megatron branch crossdc
cd ${{MEGATRON_PATH}}
# git checkout crossdc

cd ${{BASE_PATH}}
SRC_PATH=${{MEGATRON_PATH}}/pretrain_gpt.py

LOG_NAME=llama2-${{MODEL_SIZE}}_TP${{TP}}_PP${{PP}}_DP${{DP}}
LOG_PATH="${{JOB_LOG_PATH}}/${{LOG_NAME}}/node${{NODE_RANK}}.log"
mkdir -p ${{JOB_LOG_PATH}}/${{LOG_NAME}}
TB_PATH="${{JOB_LOG_PATH}}/${{LOG_NAME}}/tensorboard"
mkdir -p ${{TB_PATH}}

DATA_CACHE_PATH="${{JOB_LOG_PATH}}/.data_cache/${{LOG_NAME}}"
mkdir -p ${{DATA_CACHE_PATH}}

# SAVE_PATH=${{BASE_PATH}}/checkpoint/${{LOG_NAME}}

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${{GPUS_PER_NODE}} \
       --nnodes ${{NNODES}} \
       --node_rank \${{NODE_RANK}} \
       --master_addr ${{MASTER_ADDR}} \
       --master_port ${{MASTER_PORT}} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${{TP}} \
       --pipeline-model-parallel-size ${{PP}} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       "    

NETWORK_SIZE_ARGS=" \
       --num-layers ${{NUM_LAYERS}} \
       --hidden-size ${{HIDDEN_SIZE}} \
       --num-attention-heads ${{NUM_HEAD}} \
       --group-query-attention \
       --num-query-groups ${{NUM_QUERY_GROUP}} \
       --ffn-hidden-size ${{FFN_HIDDEN_SIZE}} \
       --position-embedding-type rope \
       --max-position-embeddings ${{MAX_POSITION_EMBEDDINGS}} \
       --make-vocab-size-divisible-by 64 \
       --norm-epsilon ${{NORM_EPS}} \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       "
LOGGING_ARGS=""
# LOGGING_ARGS=" \
#        --log-throughput \
#        --timing-log-level 0 \
#        --log-timers-to-tensorboard \
#        --log-validation-ppl-to-tensorboard \
#        --log-memory-to-tensorboard \
#        --log-world-size-to-tensorboard \
#        "

REGULATIZATION_ARGS=" \
       --attention-dropout ${{DROP_OUT}} \
       --hidden-dropout ${{DROP_OUT}} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
    --micro-batch-size 1 \
    --global-batch-size {GBS} \
    --train-iters 1000 \
    --log-interval 1 \
    --disable-bias-linear \
    --cross-entropy-loss-fusion \
    --use-flash-attn \
    --optimizer adam \
    --tensorboard-dir ${{TB_PATH}} \
    --no-barrier-with-level-1-timing \
    --no-align-grad-reduce \
    --no-align-param-gather \
    "


INITIALIZATION_ARGS=" \
       --seed 42 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${{MAX_LR}} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr ${{MIN_LR}} \
       "

CHECKPOINTING_ARGS=""
# CHECKPOINTING_ARGS=" \
#        --finetune \
#        --no-load-optim \
#        --no-load-rng \
#        "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       "

VALIDATION_ARGS=" \
       --eval-interval 1000 \
       "

DATA_ARGS=" \
       --data-path ${{DATA_PATH}} \
       --split 949,50,1 \
       --seq-length ${{MAX_SEQ_LEN}} \
       --num-workers 0 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${{TOKENIZER_PATH}} \
       --data-cache-path ${{DATA_CACHE_PATH}} \
       "

TE_ARGS=" \
    --transformer-impl local \
    "

PROFILE_ARGS=" \
    --use-pytorch-profiler \
    --profile \
    --profile-step-start 3 \
    --profile-step-end 5 \
    --profile-ranks 0 2 \
    "

#     --cdc_verbose_print
#     --cdc_print_rank -1

CDC_ARGS=" \
    --enable_cdcpp_scheduler \
    --head_tail_as_one_layer \
    --num-layers-per-virtual-pipeline-stage {layer_per_virtual_stage} \
    --cdc_verbose_print 1 \
    --no-check-for-nan-in-loss-and-grad \
    "

CDC_PROF_ARGS="\
    --enable_cdc_profile \
    --static_schedule {profile_schedule} \
    --exit-interval 4 \
    "

CDC_EXECUTE_ARGS="\
    --cdc_verbose_print 1 \
    --dynamic_schedule {schedule} \
    --exit-interval 5 \
    --num_dc 2 \
    --pp_stages_per_dc {stages_per_dc} \
    --cdc_latency 2 \
    --dynamic_mem_factor 1.05 \
    --train-sync-interval 1 \
    --cdc_exp_logging \
    --cdc_exp_tf_block_size {int(TB_size[2:])} \
    --cdc_exp_override_latency_ms 0 2 8 32 \
    --cdc_exp_override_latency_test_iters 10 \
    "

CMD="\
       ${{LAUNCHER}} \
       ${{SRC_PATH}} \
       ${{DISTRIBUTED_ARGS}} \
       ${{NETWORK_SIZE_ARGS}} \
       ${{LOGGING_ARGS}} \
       ${{REGULATIZATION_ARGS}} \
       ${{TRAINING_ARGS}} \
       ${{INITIALIZATION_ARGS}} \
       ${{LEARNING_RATE_ARGS}} \
       ${{CHECKPOINTING_ARGS}} \
       ${{MIXED_PRECISION_ARGS}} \
       ${{VALIDATION_ARGS}} \
       ${{DATA_ARGS}} \
       ${{MOE_ARGS}} \
       ${{TE_ARGS}} \
       "
CDC_PROF_CMD="${{CMD}} ${{CDC_ARGS}} ${{CDC_PROF_ARGS}}"
CDC_EXECUTE_CMD="${{CMD}} ${{CDC_ARGS}} ${{CDC_EXECUTE_ARGS}}"

PROF_RUN="${{CDC_PROF_CMD}}"
EXECUTE_RUN="${{CDC_EXECUTE_CMD}}"

"""
            prof_str = f"""\
srun bash -c "
export NODE_RANK=\${{SLURM_NODEID}}
mkdir -p \${{SCRATCH}}/tmp/\${{SLURM_JOBID}}/\${{NODE_RANK}}
export TORCHINDUCTOR_CACHE_DIR=\${{SCRATCH}}/tmp/\${{SLURM_JOBID}}/\${{NODE_RANK}}
echo \${{SCRATCH}}/tmp/\${{SLURM_JOBID}}/\${{NODE_RANK}}
echo ${{PROF_RUN}}
python -c 'import torch; print(f\\"torch version: {{torch.__version__}}\\"); print(f\\"torch path: {{torch.__path__}}\\")'
${{PROF_RUN}} 2>&1 | tee ${{LOG_PATH}}
"
"""

            exec_str = f"""\
srun bash -c "
export NODE_RANK=\${{SLURM_NODEID}}
mkdir -p \${{SCRATCH}}/tmp/\${{SLURM_JOBID}}/\${{NODE_RANK}}
export TORCHINDUCTOR_CACHE_DIR=\${{SCRATCH}}/tmp/\${{SLURM_JOBID}}/\${{NODE_RANK}}
echo \${{SCRATCH}}/tmp/\${{SLURM_JOBID}}/\${{NODE_RANK}}
echo ${{EXECUTE_RUN}}
python -c 'import torch; print(f\\"torch version: {{torch.__version__}}\\"); print(f\\"torch path: {{torch.__path__}}\\")'
${{EXECUTE_RUN}} 2>&1 | tee ${{LOG_PATH}}
"
"""
            # write the job string to a file
            with open(f"weak_{TB_size}_{PP}_{schedule}_prof.sh", "w") as f:
                f.write('\n'.join([job_str, prof_str]))
            with open(f"weak_{TB_size}_{PP}_{schedule}_exec.sh", "w") as f:
                f.write('\n'.join([job_str, exec_str]))

        
        


