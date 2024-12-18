TP=4
PORT=5000
usage () {
	echo "Usage: llama.sh <checkpoint> [options...]"
	echo "<checkpoint>: path"
	echo "Options:"
	echo " --help: Displays this message"
	echo " --tp <tp>: Size of TP"
	echo " --port <port>: Port"
}


if [[ $# -eq 0 ]]; then
	echo "Invalid argument count: $#"
	usage
	exit 1
fi

CHECKPOINT_PATH=$1
shift

while [[ $# -gt 0 ]]; do
	case $1 in
		--help)
			usage; exit 0;;
		--tp)
			TP=$2; shift 2;;
		--port)
			PORT=$2; shift 2;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done


DISTRIBUTED_ARGS=(
	--nproc_per_node $TP
        --nnodes 1
	--node_rank 0
        --master_addr localhost
	--master_port $(($PORT + 1000))
)

EVAL_ARGS=(
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size 1  \
	--use-checkpoint-args \
	--load $CHECKPOINT_PATH
	--bf16
	--micro-batch-size 10
	--tokenizer-type HuggingFaceTokenizer \
	--tokenizer-model meta-llama/Meta-Llama-3-8B \
	--seed 42
	--port $PORT
)

export HF_HOME=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/.hf_cache
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=64

CMD="torchrun ${DISTRIBUTED_ARGS[@]} tools/run_text_generation_server.py ${EVAL_ARGS[@]}"

echo Command:
echo $CMD
echo -----

$CMD

# bash eval.sh /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_1l1g/ --tp 4 --port 5000
# HF_HOME=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/.hf_cache/ lm_eval --model local-completions --tasks arc_challenge,arc_easy,commonsense_qa,hellaswag,mmlu,openbookqa,piqa,winogrande --model_args base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=meta-llama/Meta-Llama-3-8B --batch_size=10 --output_path /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_1l1g


# bash eval.sh /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_1l3g/ --tp 4 --port 5000
# HF_HOME=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/.hf_cache/ lm_eval --model local-completions --tasks arc_challenge,arc_easy,commonsense_qa,hellaswag,mmlu,openbookqa,piqa,winogrande --model_args base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=meta-llama/Meta-Llama-3-8B --batch_size=10 --output_path /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_1l3g


# bash eval.sh /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_3l1g/ --tp 4 --port 5000
# HF_HOME=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/.hf_cache/ lm_eval --model local-completions --tasks arc_challenge,arc_easy,commonsense_qa,hellaswag,mmlu,openbookqa,piqa,winogrande --model_args base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=meta-llama/Meta-Llama-3-8B --batch_size=10 --output_path /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_3l1g


# bash eval.sh /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_baseline/ --tp 4 --port 5000
# HF_HOME=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/.hf_cache/ lm_eval --model local-completions --tasks arc_challenge,arc_easy,commonsense_qa,hellaswag,mmlu,openbookqa,piqa,winogrande --model_args base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=meta-llama/Meta-Llama-3-8B --batch_size=10 --output_path /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_baseline


# bash eval.sh /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_baseline_all_l/ --tp 4 --port 5000
# HF_HOME=/capstor/scratch/cscs/ctianche/playground/Megatron-LM/.hf_cache/ lm_eval --model local-completions --tasks arc_challenge,arc_easy,commonsense_qa,hellaswag,mmlu,openbookqa,piqa,winogrande --model_args base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=meta-llama/Meta-Llama-3-8B --batch_size=10 --output_path /capstor/scratch/cscs/ctianche/playground/Megatron-LM/test_interleaved_att/test/checkpoint/iv_att_model8_baseline_all_l
