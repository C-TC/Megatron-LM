#!/bin/bash -l
#SBATCH --job-name="eval"
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=normal
#SBATCH --cpus-per-task=64
#SBATCH --environment=megatron
#SBATCH --time=12:00:00            # total run time limit (HH:MM:SS)

cleanup() {
    echo "Cleaning up background processes..."
    if [ ! -z "$eval_pid" ]; then
        kill -9 $eval_pid 2>/dev/null
        wait $eval_pid 2>/dev/null
    fi
}

EVAL_SCRIPT_PATH=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/
EVAL_SCRIPT=${EVAL_SCRIPT_PATH}/eval.sh
CHECKPOINT_BASE_PATH=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/

MODELS=(
    "iv_att_model8_1l1g"
    "iv_att_model8_1l3g"
    "iv_att_model8_3l1g"
    "iv_att_model8_baseline"
    "iv_att_model8_baseline_all_l"
)

cd /capstor/scratch/cscs/ctianche/playground/Megatron-LM

for MODEL in "${MODELS[@]}"
do
    echo "Evaluating model: ${MODEL}"
    bash ${EVAL_SCRIPT} ${CHECKPOINT_BASE_PATH}/${MODEL}  --tp 4 --port 5000 &
    eval_pid=$!

    echo "Waiting for server to initialize..."
    sleep 60

    HF_HOME=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/.hf_cache/ lm_eval \
    --model local-completions \
    --tasks arc_challenge,arc_easy,commonsense_qa,hellaswag,mmlu,openbookqa,piqa,winogrande \
    --model_args base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=meta-llama/Meta-Llama-3-8B \
    --batch_size=10 \
    --output_path ${CHECKPOINT_BASE_PATH}/${MODEL}/

    cleanup
    sleep 5
done

