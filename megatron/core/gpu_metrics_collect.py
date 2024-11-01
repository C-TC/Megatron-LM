import os
import signal
import subprocess
import tempfile

from megatron.core import parallel_state
from megatron.training import get_args

_GPU_METRICS_COLLECTOR = None


def get_gpu_metrics_collector():
    global _GPU_METRICS_COLLECTOR
    if _GPU_METRICS_COLLECTOR is None:
        args = get_args()
        _GPU_METRICS_COLLECTOR = GpuMetricCollect(args)
    return _GPU_METRICS_COLLECTOR


gpu_metrics = [
    "timestamp",
    "index",
    "memory.total",
    "memory.reserved",
    "memory.used",
    "memory.free",
    "fan.speed",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "ecc.errors.corrected.volatile.device_memory",
    "ecc.errors.corrected.volatile.dram",
    "ecc.errors.corrected.volatile.l1_cache",
    "ecc.errors.corrected.volatile.l2_cache",
    "ecc.errors.corrected.volatile.texture_memory",
    "ecc.errors.corrected.volatile.register_file",
    "ecc.errors.corrected.volatile.cbu",
    "ecc.errors.corrected.volatile.sram",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.corrected.aggregate.device_memory",
    "ecc.errors.corrected.aggregate.dram",
    "ecc.errors.corrected.aggregate.l1_cache",
    "ecc.errors.corrected.aggregate.l2_cache",
    "ecc.errors.corrected.aggregate.texture_memory",
    "ecc.errors.corrected.aggregate.register_file",
    "ecc.errors.corrected.aggregate.cbu",
    "ecc.errors.corrected.aggregate.sram",
    "ecc.errors.corrected.aggregate.total",
    "ecc.errors.uncorrected.volatile.device_memory",
    "ecc.errors.uncorrected.volatile.dram",
    "ecc.errors.uncorrected.volatile.l1_cache",
    "ecc.errors.uncorrected.volatile.l2_cache",
    "ecc.errors.uncorrected.volatile.texture_memory",
    "ecc.errors.uncorrected.volatile.register_file",
    "ecc.errors.uncorrected.volatile.cbu",
    "ecc.errors.uncorrected.volatile.sram",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.device_memory",
    "ecc.errors.uncorrected.aggregate.dram",
    "ecc.errors.uncorrected.aggregate.l1_cache",
    "ecc.errors.uncorrected.aggregate.l2_cache",
    "ecc.errors.uncorrected.aggregate.texture_memory",
    "ecc.errors.uncorrected.aggregate.register_file",
    "ecc.errors.uncorrected.aggregate.cbu",
    "ecc.errors.uncorrected.aggregate.sram",
    "ecc.errors.uncorrected.aggregate.total",
    "retired_pages.single_bit_ecc.count",
    "retired_pages.double_bit.count",
    "retired_pages.pending",
    "temperature.gpu",
    "temperature.gpu.tlimit",
    "temperature.memory",
    "power.draw",
    "power.draw.instant",
    "clocks.sm",
    "clocks.mem",
    "fabric.state",
]


class GpuMetricCollect:
    """
    Logging GPU metrics from nvidia-smi.
    If tensorboard_dir is given, logs are written to tensorboard_dir/gpu_metrics_job<job_id>.
    Otherwise, logs are written to SLURM_SUBMIT_DIR/gpu_metrics_job<job_id>.
    Default interval is 100ms.
    If total number of GPUs is less than or equal to 64, logs are written on each node.
    Otherwise, logs are written on some data parallel ranks (dp_ranks_to_sample, default is <= 16).

    TODO: consider CP and EP.

    """

    def __init__(self, args, dp_ranks_to_sample=16, interval_ms=100):
        self.args = args
        self.log_folder = None
        self.interval_ms = interval_ms
        if hasattr(args, "tensorboard_dir") and args.tensorboard_dir is not None:
            self.log_folder = args.tensorboard_dir
        else:
            # get slurm job submit dir
            self.log_folder = os.environ.get("SLURM_SUBMIT_DIR", None)
            assert (
                self.log_folder is not None
            ), "No log path specified and SLURM_SUBMIT_DIR not found."

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        dp_size = parallel_state.get_data_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()

        self.local_rank = int(os.environ["LOCAL_RANK"])

        if tp_size * pp_size * dp_size <= 64:
            # log on each node
            self.should_log = self.local_rank == 0
        else:
            # log on some dp ranks
            self.should_log = dp_rank < dp_ranks_to_sample and self.local_rank == 0

        self.job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        self.node_id = os.environ.get("SLURMD_NODENAME", "unknown")
        self.log_folder = os.path.join(self.log_folder, f"gpu_metrics_job{self.job_id}")
        os.makedirs(self.log_folder, exist_ok=True)
        self.smi_log_file_name = (
            f"nvidia-smi_TP{tp_rank}_PP{pp_rank}_DP{dp_rank}_node_{self.node_id}.csv"
        )
        self.smi_log_file = os.path.join(self.log_folder, self.smi_log_file_name)
        self.pm_log_file_name = (
            f"pm_counter_TP{tp_rank}_PP{pp_rank}_DP{dp_rank}_node_{self.node_id}.csv"
        )
        self.pm_log_file = os.path.join(self.log_folder, self.pm_log_file_name)
        self.pm_script_content = f'''#!/bin/bash
OUTPUT_FILE="{self.pm_log_file}"

# Create directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Create header
echo "timestamp,$(cd /sys/cray/pm_counters && ls * | tr '\n' ',' | sed 's/,$//')" > "$OUTPUT_FILE"

# Function to collect measurements
collect_measurement() {{
    timestamp=$(date "+%Y/%m/%d %H:%M:%S.%3N")
    values=$(cd /sys/cray/pm_counters && ls * | while read file; do
        value=$(cat "$file" 2>/dev/null | awk '{{print $1}}')
        echo "${{value:-NA}}"
    done | tr '\n' ',')
    echo "$timestamp,${{values%,}}" >> "$OUTPUT_FILE"
}}

# Main loop
while true; do
    collect_measurement
    sleep {self.interval_ms / 1000}
done
'''
        # Create temporary script file
        self.pm_script = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        )
        self.pm_script.write(self.pm_script_content)
        self.pm_script.close()
        os.chmod(self.pm_script.name, 0o755)

        self.smi_process = None
        self.smi_process_fp = None
        self.pm_process = None

    # destructor
    def __del__(self):
        if self.should_log and self.smi_process is not None:
            self.stop_collection_process()

    def launch_collection_process(self):
        if not self.should_log:
            return

        smi_cmd = f"nvidia-smi --query-gpu={','.join(gpu_metrics)} --format=csv,nounits -lms {self.interval_ms} --filename={os.path.join(self.log_folder, self.smi_log_file_name)} "
        self.smi_process_fp = open(self.smi_log_file, "w")
        self.smi_process = subprocess.Popen(
            smi_cmd.split(),
            shell=False,
            stdout=self.smi_process_fp,
            stderr=subprocess.PIPE,
        )
        self.pm_process = subprocess.Popen(
            [self.pm_script.name],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop_collection_process(self):
        try:
            if self.smi_process is not None:
                self.smi_process.terminate()
                try:
                    self.smi_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.smi_process.kill()
                self.smi_process = None
            if self.pm_process is not None:
                self.pm_process.send_signal(signal.SIGTERM)
                try:
                    self.pm_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.pm_process.kill()
                os.unlink(self.pm_script.name)
                self.pm_process = None
            if self.smi_process_fp is not None:
                self.smi_process_fp.close()
                self.smi_process_fp = None
        except Exception as e:
            print(f"Error stopping GPU metrics collection process: {e}")
