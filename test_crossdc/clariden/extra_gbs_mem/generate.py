
# num_nodes = 2
TP = 4
PP = 8
DP = 1
num_nodes = TP * PP * DP // 4
num_layers = 64
model_size = 70
seq_len = 4096
# job_title = ""
mbs = 1
# gbs = 8
# layers_per_chunk = 2
# extra_cdc_args = ""
lat_bw_as_F_stage_pairs = "0,0 0.25,0.25 0.25,2 2,0.25 2,2"
num_dc = 2
pp_stages_per_dc = PP // num_dc
cdc_exp_per_cfg_test_iters = 64


def get_job_str(job_title, gbs, layers_per_chunk, extra_cdc_args, dynamic_extra_mem_factor):
    return f"""\
#!/bin/bash -l
#SBATCH --job-name="7.2:{job_title}"
#SBATCH --nodes={num_nodes}                   # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH -c 288
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --time=04:00:00            # total run time limit (HH:MM:SS)


GLOBAL_ARGS="\
export CUDA_DEVICE_MAX_CONNECTIONS=0
# export NCCL_DEBUG=INFO
export NCCL_DEBUG=DEBUG
export OMP_NUM_THREADS=1
export TRITON_HOME=${{SCRATCH}}/crosspipe_ae/Megatron-LM/test_crossdc/.triton_cache
export TRITON_CACHE_DIR=${{SCRATCH}}/crosspipe_ae/Megatron-LM/test_crossdc/.triton_cache
"
mkdir -p ${{SCRATCH}}/crosspipe_ae/Megatron-LM/test_crossdc/.triton_cache

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
MODEL_SIZE={model_size}

if   [[ ${{MODEL_SIZE}} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=8; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == 34 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8; NUM_LAYERS=40; FFN_HIDDEN_SIZE=22016; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == 405 ]];  then HIDDEN_SIZE=16384;  NUM_HEAD=128; NUM_QUERY_GROUP=16;  NUM_LAYERS=128; FFN_HIDDEN_SIZE=53248; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == "tiny_test" ]]; then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8; NUM_LAYERS=8; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${{MODEL_SIZE}} == "tiny_ault" ]]; then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=8; NUM_LAYERS=16; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${{MODEL_SIZE}}"; exit 1
fi

NUM_LAYERS={num_layers}

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN={seq_len}
MAX_POSITION_EMBEDDINGS=${{MAX_SEQ_LEN}}

# Paths
BASE_PATH="${{SCRATCH}}/crosspipe_ae/Megatron-LM/test_crossdc/clariden/extra_gbs_mem"
SCRIPT_NAME=$(basename "$0")
SCRIPT_BASENAME="${{SCRIPT_NAME%.*}}"

# get job id
JOB_ID=${{SLURM_JOB_ID}}

# create Job log path = script name + job id
JOB_LOG_PATH="${{BASE_PATH}}/job_logs/{job_title}_${{JOB_ID}}"


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
LAUNCHER=" \\
       torchrun \\
       --nproc_per_node ${{GPUS_PER_NODE}} \\
       --nnodes ${{NNODES}} \\
       --node_rank \\${{NODE_RANK}} \\
       --master_addr ${{MASTER_ADDR}} \\
       --master_port ${{MASTER_PORT}} \\
       "

DISTRIBUTED_ARGS=" \\
       --tensor-model-parallel-size ${{TP}} \\
       --pipeline-model-parallel-size ${{PP}} \\
       --distributed-backend nccl \\
       --use-distributed-optimizer \\
       --sequence-parallel \\
       "    

NETWORK_SIZE_ARGS=" \\
       --num-layers ${{NUM_LAYERS}} \\
       --hidden-size ${{HIDDEN_SIZE}} \\
       --num-attention-heads ${{NUM_HEAD}} \\
       --group-query-attention \\
       --num-query-groups ${{NUM_QUERY_GROUP}} \\
       --ffn-hidden-size ${{FFN_HIDDEN_SIZE}} \\
       --position-embedding-type rope \\
       --max-position-embeddings ${{MAX_POSITION_EMBEDDINGS}} \\
       --make-vocab-size-divisible-by 64 \\
       --norm-epsilon ${{NORM_EPS}} \\
       --normalization RMSNorm \\
       --swiglu \\
       --untie-embeddings-and-output-weights \\
       "
LOGGING_ARGS=""

REGULATIZATION_ARGS=" \\
       --attention-dropout ${{DROP_OUT}} \\
       --hidden-dropout ${{DROP_OUT}} \\
       --weight-decay 1e-1 \\
       --clip-grad 1.0 \\
       --adam-beta1 0.9 \\
       --adam-beta2 0.95 \\
       --adam-eps 1e-8 \\
       "

TRAINING_ARGS=" \\
    --micro-batch-size {mbs} \\
    --global-batch-size {gbs} \\
    --train-iters 1000 \\
    --log-interval 1 \\
    --disable-bias-linear \\
    --cross-entropy-loss-fusion \\
    --use-flash-attn \\
    --optimizer adam \\
    --tensorboard-dir ${{TB_PATH}} \\
    --no-barrier-with-level-1-timing \\
    --no-align-grad-reduce \\
    --no-align-param-gather \\
    --no-check-for-nan-in-loss-and-grad \\
    --distributed-timeout-minutes 10 \\
    "


INITIALIZATION_ARGS=" \\
       --seed 42 \\
       --init-method-std 0.02 \\
       "

LEARNING_RATE_ARGS=" \\
       --lr ${{MAX_LR}} \\
       --lr-decay-style cosine \\
       --lr-warmup-fraction 0.1 \\
       --min-lr ${{MIN_LR}} \\
       "

CHECKPOINTING_ARGS=""
# CHECKPOINTING_ARGS=" \\
#        --finetune \\
#        --no-load-optim \\
#        --no-load-rng \\
#        "

MIXED_PRECISION_ARGS=" \\
       --bf16 \\
       "

VALIDATION_ARGS=" \\
       --eval-interval 1000 \\
       "

DATA_ARGS=" \\
       --data-path ${{DATA_PATH}} \\
       --split 949,50,1 \\
       --seq-length ${{MAX_SEQ_LEN}} \\
       --num-workers 0 \\
       --tokenizer-type Llama2Tokenizer \\
       --tokenizer-model ${{TOKENIZER_PATH}} \\
       --data-cache-path ${{DATA_CACHE_PATH}} \\
       "

TE_ARGS=" \\
    --transformer-impl local \\
    "

PROFILE_ARGS=" \\
    --use-pytorch-profiler \\
    --profile \\
    --profile-step-start 3 \\
    --profile-step-end 5 \\
    --profile-ranks 0 2 \\
    "

NSYS_PROFILE_ARGS=" \\
    --profile \\
    --profile-step-start 3 \\
    --profile-step-end 7 \\
    --profile-ranks 0 2 4 6 \\
    "

CDC_ARGS=" \\
    --enable_cdcpp_scheduler \\
    --head_tail_as_one_layer \\
    --num-layers-per-virtual-pipeline-stage {layers_per_chunk} \\
    --cdc_profile_iter 2 \\
    --exit-interval 4 \\
    --num_dc {num_dc} \\
    --pp_stages_per_dc {pp_stages_per_dc} \\
    --train-sync-interval 1 \\
    --cdc_exp_logging \\
    --dynamic_extra_mem_factor {dynamic_extra_mem_factor} \\
    --cdc_exp_tf_block_size {model_size} \\
    --cdc_exp_per_cfg_test_iters {cdc_exp_per_cfg_test_iters} \\
    --cdc_exp_dump_execution_plan \\
    --cdc_latency_bandwidth_delay_as_F_stage {lat_bw_as_F_stage_pairs} \\
    --cdc_verbose_print 1 \\
    {extra_cdc_args} \\
    "

CMD="\
       ${{LAUNCHER}} \\
       ${{SRC_PATH}} \\
       ${{DISTRIBUTED_ARGS}} \\
       ${{NETWORK_SIZE_ARGS}} \\
       ${{LOGGING_ARGS}} \\
       ${{REGULATIZATION_ARGS}} \\
       ${{TRAINING_ARGS}} \\
       ${{INITIALIZATION_ARGS}} \\
       ${{LEARNING_RATE_ARGS}} \\
       ${{CHECKPOINTING_ARGS}} \\
       ${{MIXED_PRECISION_ARGS}} \\
       ${{VALIDATION_ARGS}} \\
       ${{DATA_ARGS}} \\
       ${{MOE_ARGS}} \\
       ${{TE_ARGS}} \\
       "
NSYS="\
       nsys profile \\
       --trace='nvtx,cuda,osrt' \\
       --output='${{TB_PATH}}/trace_%q{{RANK}}_%q{{SLURM_NODEID}}_%q{{SLURM_LOCALID}}.nsys-rep' \\
       --force-overwrite true \\
       --capture-range=cudaProfilerApi \\
       --capture-range-end=stop \\
       "
# NSYS=""
CDC_CMD="${{CMD}} ${{CDC_ARGS}}"
# CDC_CMD="${{NSYS}} ${{CMD}} ${{CDC_ARGS}} ${{NSYS_PROFILE_ARGS}}"

srun --mpi=pmi2 --environment=megatron_cdc numactl --membind=0-3 bash -c "
${{GLOBAL_ARGS}}
export NODE_RANK=\${{SLURM_NODEID}}
export PATH=${{SCRATCH}}/crosspipe_ae/cplex_arm/cpoptimizer/bin/arm64_linux:\${{PATH}}
export PYTHONPATH=${{SCRATCH}}/crosspipe_ae/pytorch:\${{PYTHONPATH}}
export LD_LIBRARY_PATH=${{SCRATCH}}/crosspipe_ae/pytorch/build/lib:${{SCRATCH}}/crosspipe_ae/acl/build:\${{LD_LIBRARY_PATH}}
echo ${{CDC_CMD}}
python -c 'import torch; print(f\\"torch version: {{torch.__version__}}\\"); print(f\\"torch path: {{torch.__path__}}\\")'
${{CDC_CMD}} 2>&1 | tee ${{LOG_PATH}}
"

"""




job_dict = {}
    
for schedules in ['1F1B', 'ZBH1', 'ZBV', 'wave', 'ud', 'subud']:
    layers_per_chunk = num_layers // PP // 2 if schedules in ['ZBV', 'wave'] else num_layers // PP
    if schedules in ['1F1B', 'ZBH1', 'ZBV']:
        extra_cdc_args = f"--static_schedule {schedules}"
    else:
        extra_cdc_args = f"--dynamic_schedule {schedules}"
        if schedules == 'subud':
            extra_cdc_args += " --num_subparts 4"
    # case 1
    gbs = mbs * DP * PP * 2
    extra_mem_factor = 0.0
    job_title = f'{schedules}_c1gbs2m0'
    job_dict[job_title] = get_job_str(job_title, gbs, layers_per_chunk, extra_cdc_args, extra_mem_factor)
    
    # case 2 recompute + GBS
    gbs = mbs * DP * PP * 4
    extra_mem_factor = 0.0
    recompute_args = " --recompute-granularity full  --recompute-method uniform  --recompute-num-layers 1 "
    job_title = f'{schedules}_c2gbs4m0R'
    job_dict[job_title] = get_job_str(job_title, gbs, layers_per_chunk, extra_cdc_args + recompute_args, extra_mem_factor)
    
    if schedules in ['1F1B', 'ZBH1', 'ZBV']:
        continue
    
    # case 3 GBS + extra mem
    gbs = mbs * DP * PP * 4
    extra_mem_factor = 1.0
    job_title = f'{schedules}_c3gbs4m1'
    job_dict[job_title] = get_job_str(job_title, gbs, layers_per_chunk, extra_cdc_args, extra_mem_factor)       
        
                        
for job_title, job_str in job_dict.items():
    # write job file
    with open(f"{job_title}.sbatch", "w") as f:
        f.write(job_str)



