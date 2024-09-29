from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum, value

from .model_config import LLAMA_SIZE_TO_CONFIG, CUSTOM_SIZE_TO_CONFIG, LlamaConfig
from .pipeline_config import SystemConfig
from .pipeline import GpipePipeline, Hanayo1F1BPipeline, HeuristicWaveZBPipeline, Interleaved1F1BPipeline, OneFOneBPipeline, TaskNode, InterleavedTaskNode, ZBH1Pipeline
from util import generate_comm_mat, scale_to_int
from .auto_schedule import gurobi_options

import pulp
import gurobipy as gp

@dataclass
class SimConfig:
    model_cfg: LlamaConfig
    seq_len: int
    mbs: int
    tp: int
    pp: int
    dp: int
    cp: int
    num_mb_per_pp_stage: int
    num_chunks: int
    gpu_mem_bytes: int
    gpu_avg_perf_flops: int
    num_DC: int
    intra_DC_bandwidth: int # bandwidth for DP x CP weight sync
    DC_comm_latency: float
    DC_comm_bandwidth: int


    
class SimCfgGen:
    def __init__(
        self,
        simcfg: SimConfig,
    ):
        self.cfg = simcfg.model_cfg
        self.seq_len = simcfg.seq_len
        self.mbs = simcfg.mbs
        self.tp = simcfg.tp
        self.pp = simcfg.pp
        self.dp = simcfg.dp
        self.cp = simcfg.cp
        self.num_mb_per_pp_stage = simcfg.num_mb_per_pp_stage
        self.num_chunks = simcfg.num_chunks
        self.gpu_mem_bytes = simcfg.gpu_mem_bytes
        self.gpu_avg_perf_flops = simcfg.gpu_avg_perf_flops
        self.num_DC = simcfg.num_DC
        self.intra_DC_bandwidth = simcfg.intra_DC_bandwidth
        self.DC_comm_latency = simcfg.DC_comm_latency
        self.DC_comm_bandwidth = simcfg.DC_comm_bandwidth
        # assume zero2, bf16 training
        self.time_scale_factor: Optional[float] = None

    def get_forward_flops_per_layer(self):
        mbs = self.mbs
        seq_len = self.seq_len
        hidden_size = self.cfg.hidden_size
        intermediate_size = self.cfg.intermediate_size
        alpha_kv = self.cfg.num_key_value_heads / self.cfg.num_attention_heads
        # 2bs^2d + 4bsd^2 + 4\alpha_{kv}bsd^2 + 6bsdd_{MLP}
        return (
            2 * mbs * seq_len**2 * hidden_size
            + 4 * mbs * seq_len * hidden_size**2
            + 4 * alpha_kv * mbs * seq_len * hidden_size**2
            + 6 * mbs * seq_len * hidden_size * intermediate_size
        )

    def get_dgrad_flops_per_layer(self):
        mbs = self.mbs
        seq_len = self.seq_len
        hidden_size = self.cfg.hidden_size
        intermediate_size = self.cfg.intermediate_size
        alpha_kv = self.cfg.num_key_value_heads / self.cfg.num_attention_heads
        # 5bs^2d + 4bsd^2 + 4\alpha_{kv}bsd^2 + 6bsdd_{MLP}
        return (
            5 * mbs * seq_len**2 * hidden_size
            + 4 * mbs * seq_len * hidden_size**2
            + 4 * alpha_kv * mbs * seq_len * hidden_size**2
            + 6 * mbs * seq_len * hidden_size * intermediate_size
        )

    def get_wgrad_flops_per_layer(self):
        mbs = self.mbs
        seq_len = self.seq_len
        hidden_size = self.cfg.hidden_size
        intermediate_size = self.cfg.intermediate_size
        alpha_kv = self.cfg.num_key_value_heads / self.cfg.num_attention_heads
        # 4bsd^2 + 4\alpha_{kv}bsd^2 + 6bsdd_{MLP}
        return (
            4 * mbs * seq_len * hidden_size**2
            + 4 * alpha_kv * mbs * seq_len * hidden_size**2
            + 6 * mbs * seq_len * hidden_size * intermediate_size
        )

    def get_num_layers_per_block(self):
        assert (
            self.cfg.num_hidden_layers % (self.num_chunks * self.pp) == 0
        ), "num_layers not divisible by num_chunks * pp"
        return self.cfg.num_hidden_layers // self.num_chunks // self.pp

    def get_forward_block_runtime(self):
        layers_per_block = self.get_num_layers_per_block()
        return (
            self.get_forward_flops_per_layer()
            * layers_per_block
            / self.gpu_avg_perf_flops
            / self.tp
            / self.cp
        )

    def get_dgrad_block_runtime(self):
        layers_per_block = self.get_num_layers_per_block()
        return (
            self.get_dgrad_flops_per_layer()
            * layers_per_block
            / self.gpu_avg_perf_flops
            / self.tp
            / self.cp
        )

    def get_wgrad_block_runtime(self):
        layers_per_block = self.get_num_layers_per_block()
        return (
            self.get_wgrad_flops_per_layer()
            * layers_per_block
            / self.gpu_avg_perf_flops
            / self.tp
            / self.cp
        )

    def get_inter_DC_comm_time(self):
        mbs = self.mbs
        seq_len = self.seq_len
        hidden_size = self.cfg.hidden_size
        comm_vol = 2 * mbs * seq_len * hidden_size * self.dp
        return comm_vol / self.DC_comm_bandwidth + self.DC_comm_latency

    def get_num_param_per_layer(self):
        mbs = self.mbs
        seq_len = self.seq_len
        hidden_size = self.cfg.hidden_size
        intermediate_size = self.cfg.intermediate_size
        alpha_kv = self.cfg.num_key_value_heads / self.cfg.num_attention_heads
        # 2(1 + \alpha_{kv})d^2 + 3dd_{MLP}
        num_param_per_layer = (
            2 * (1 + alpha_kv) * hidden_size**2 + 3 * hidden_size * intermediate_size
        )
        return num_param_per_layer

    def get_avail_gpu_mem_for_activation(self):
        total_mem = self.gpu_mem_bytes
        num_param_per_layer = self.get_num_param_per_layer()
        num_param = num_param_per_layer * self.cfg.num_hidden_layers / self.pp
        # mem: 2N + 16N / dp
        model_mem_per_gpu = (2 * num_param + 16 * num_param / self.dp / self.cp) / self.tp
        assert model_mem_per_gpu < total_mem, "no space for activation"
        return total_mem - model_mem_per_gpu

    def get_forward_block_mem_incr(self, layer_recompute=False):
        mbs = self.mbs
        seq_len = self.seq_len
        hidden_size = self.cfg.hidden_size
        intermediate_size = self.cfg.intermediate_size
        alpha_kv = self.cfg.num_key_value_heads / self.cfg.num_attention_heads

        if layer_recompute:
            return (
                2
                * mbs
                * seq_len
                * hidden_size
                * self.get_num_layers_per_block()
                / self.tp
                / self.cp
            )

        # 12bsd + 4\alpha_{kv}bsd + 8bsd_{MLP}
        return (
            (
                12 * mbs * seq_len * hidden_size
                + 4 * alpha_kv * mbs * seq_len * hidden_size
                + 8 * mbs * seq_len * intermediate_size
            )
            * self.get_num_layers_per_block()
            / self.tp
            / self.cp
        )

    def get_wgrad_block_mem_incr(self):
        mbs = self.mbs
        seq_len = self.seq_len
        hidden_size = self.cfg.hidden_size
        intermediate_size = self.cfg.intermediate_size
        alpha_kv = self.cfg.num_key_value_heads / self.cfg.num_attention_heads
        # 12bsd + 4\alpha_{kv}bsd + 6bsd_{MLP}
        return (
            -(
                12 * mbs * seq_len * hidden_size
                + 4 * alpha_kv * mbs * seq_len * hidden_size
                + 6 * mbs * seq_len * intermediate_size
            )
            * self.get_num_layers_per_block()
            / self.tp
            / self.cp
        )

    def get_dgrad_block_mem_incr(self):
        return -self.get_forward_block_mem_incr() - self.get_wgrad_block_mem_incr()

    def get_tokens_per_iteration(self):
        return self.seq_len * self.mbs * self.num_mb_per_pp_stage * self.pp * self.dp

    def print_stats(self):
        print(f"Model size: {self.cfg.name}")
        print(f"Seq len: {self.seq_len}")
        print(f"MBs: {self.mbs}")
        print(f"TP: {self.tp}")
        print(f"PP: {self.pp}")
        print(f"DP: {self.dp}")
        print(f'CP: {self.cp}')
        print(f"Num MB per PP stage: {self.num_mb_per_pp_stage}")
        print(f"Num chunks: {self.num_chunks}")
        print(f"GPU mem: {self.gpu_mem_bytes}")
        print(f"GPU avg perf: {self.gpu_avg_perf_flops}")
        print(f"Num DC: {self.num_DC}")
        print(f"DC comm latency (ms): {self.DC_comm_latency * 1000}")
        print(f"DC comm bandwidth (GB/s): {self.DC_comm_bandwidth / 10**9}")
        print(f"Num layers per block: {self.get_num_layers_per_block()}")
        print(
            f"Forward flops per layer (TFLOPS): {self.get_forward_flops_per_layer() / 10**12}"
        )
        print(
            f"Dgrad flops per layer (TFLOPS): {self.get_dgrad_flops_per_layer() / 10**12}"
        )
        print(
            f"Wgrad flops per layer (TFLOPS): {self.get_wgrad_flops_per_layer() / 10**12}"
        )
        print(f"Forward block runtime (ms): {self.get_forward_block_runtime() * 1000}")
        print(f"Dgrad block runtime (ms): {self.get_dgrad_block_runtime() * 1000}")
        print(f"Wgrad block runtime (ms): {self.get_wgrad_block_runtime() * 1000}")
        print(f"Inter DC comm time (ms): {self.get_inter_DC_comm_time() * 1000}")
        print(
            f"Avail GPU mem for activation (GB): {self.get_avail_gpu_mem_for_activation() / 1024**3}"
        )
        print(
            f"Forward block mem incr (GB): {self.get_forward_block_mem_incr() / 1024**3}"
        )
        print(f"Dgrad block mem incr (GB): {self.get_dgrad_block_mem_incr() / 1024**3}")
        print(f"Wgrad block mem incr (GB): {self.get_wgrad_block_mem_incr() / 1024**3}")

    def get_system_config(self, combine_B_W=False, round_to_int=True):
        if round_to_int:
            # prepare for MILP solver
            (T_F, T_B, T_W, T_C), time_scale_factor = scale_to_int(
                [
                    self.get_forward_block_runtime(),
                    self.get_dgrad_block_runtime(),
                    self.get_wgrad_block_runtime(),
                    self.get_inter_DC_comm_time(),
                ]
            )
        else:
            T_F, T_B, T_W, T_C = (
                self.get_forward_block_runtime(),
                self.get_dgrad_block_runtime(),
                self.get_wgrad_block_runtime(),
                self.get_inter_DC_comm_time(),
            )
            time_scale_factor = 1
        
        T_C = generate_comm_mat(self.num_DC, self.pp // self.num_DC, 0, T_C)
        self.time_scale_factor = time_scale_factor
        
        if round_to_int:
            # prepare for MILP solver
            (M_F, M_B, M_W, M_Limit), _ = scale_to_int(
                [
                    self.get_forward_block_mem_incr(),
                    self.get_dgrad_block_mem_incr(),
                    self.get_wgrad_block_mem_incr(),
                    self.get_avail_gpu_mem_for_activation(),
                ]
            )
        else:
            M_F, M_B, M_W, M_Limit = (
                self.get_forward_block_mem_incr(),
                self.get_dgrad_block_mem_incr(),
                self.get_wgrad_block_mem_incr(),
                self.get_avail_gpu_mem_for_activation(),
            )
            
        M_W = -M_F - M_B

        if combine_B_W:
            M_B += M_W
            M_W = 0
            T_B += T_W
            T_W = 0
        
        return SystemConfig(
            num_devices=self.pp,
            num_microbatches=self.num_mb_per_pp_stage * self.pp,
            num_chunks=self.num_chunks,
            T_F=T_F,
            T_B=T_B,
            T_W=T_W,
            T_C=T_C,
            M_F=M_F,
            M_B=M_B,
            M_W=M_W,
            M_Limit=M_Limit,
        )

    def get_optimal_runtime(self):
        T_F, T_B, T_W = (
            self.get_forward_block_runtime(),
            self.get_dgrad_block_runtime(),
            self.get_wgrad_block_runtime(),
        )
        return (T_F + T_B + T_W) * self.num_chunks * self.num_mb_per_pp_stage * self.pp
    
    def get_DP_comm_time_per_chunk(self):
        # BF16 RS + AG
        # assume 0 latency
        num_params = self.get_num_param_per_layer() * self.cfg.num_hidden_layers / self.pp / self.num_chunks
        comm_group_size = self.dp * self.cp
        return 2 * (comm_group_size - 1) / comm_group_size * 2 * num_params / self.intra_DC_bandwidth 
    
    def get_total_runtime_with_DP_comm(self, scheduled_tasks: List[List[TaskNode]]):
        is_chunked = isinstance(scheduled_tasks[0][0], InterleavedTaskNode)
        is_wgrad_split = any([task.task_type == "W" for task in scheduled_tasks[0]])
        dev_total_time_list = [tasks[-1].completion_time - tasks[0].start_time for tasks in scheduled_tasks]
        dev_chunk_DP_start_time_list = [[] for _ in range(self.pp)]
        for device_id, device_tasks in enumerate(scheduled_tasks):
            dev_start_time = device_tasks[0].start_time
            
            if not is_chunked:
                for task in reversed(device_tasks):
                    if (is_wgrad_split and task.task_type == "W") or (not is_wgrad_split and task.task_type == "B"):
                        dev_chunk_DP_start_time_list[device_id].append(task.completion_time - dev_start_time)
                        break
            else:
                for chunk_id in range(self.num_chunks):
                    for task in reversed(device_tasks):
                        if task.chunk_id == chunk_id:
                            if (is_wgrad_split and task.task_type == "W") or (not is_wgrad_split and task.task_type == "B"):
                                dev_chunk_DP_start_time_list[device_id].append(task.completion_time - dev_start_time)
                                break
        # sort 
        dev_chunk_DP_start_time_list = [sorted(times) for times in dev_chunk_DP_start_time_list]
        # get runtime + exposed DP comm time
        dev_total_time_with_DP_comm = []
        for total_comp_time, chunk_DP_start_time in zip(dev_total_time_list, dev_chunk_DP_start_time_list):
            if not is_chunked:
                dev_total_time_with_DP_comm.append(max(total_comp_time, chunk_DP_start_time[-1] + self.get_DP_comm_time_per_chunk()))
            else:
                DP_comm_per_chunk = self.get_DP_comm_time_per_chunk()
                prev_chunk_finish = chunk_DP_start_time[0] + DP_comm_per_chunk
                for chunk_DP_start in chunk_DP_start_time[1:]:
                    prev_chunk_finish = max(prev_chunk_finish, chunk_DP_start) + DP_comm_per_chunk
                dev_total_time_with_DP_comm.append(max(total_comp_time, prev_chunk_finish))
        return max(dev_total_time_with_DP_comm)
    
    def get_num_gpus(self):
        return self.pp * self.dp * self.cp * self.tp
    
    def get_global_batch_size_in_tokens(self):
        return self.seq_len * self.mbs * self.num_mb_per_pp_stage * self.pp * self.dp

@dataclass
class BFSPPSimConfig(SimConfig):
    layer_recompute: bool = False

class BFSPPSimCfgGen(SimCfgGen):
    def __init__(
        self,
        bfspp_config: BFSPPSimConfig,
        layer_recompute: bool = False,
    ):
        self.simcfg = bfspp_config
        self.cfg = bfspp_config.model_cfg
        self.seq_len = bfspp_config.seq_len
        self.mbs = bfspp_config.mbs
        self.tp = bfspp_config.tp
        self.pp = bfspp_config.pp
        self.dp = bfspp_config.dp
        self.cp = bfspp_config.cp
        self.num_chunks = bfspp_config.num_chunks
        self.gpu_mem_bytes = bfspp_config.gpu_mem_bytes
        self.gpu_avg_perf_flops = bfspp_config.gpu_avg_perf_flops
        self.num_DC = bfspp_config.num_DC
        self.DC_comm_latency = bfspp_config.DC_comm_latency
        self.DC_comm_bandwidth = bfspp_config.DC_comm_bandwidth
        self.DC_intra_comm_bandwidth = bfspp_config.intra_DC_bandwidth
        # assume zero2, bf16 training

        self.layer_recompute = layer_recompute
        self.num_mb, _ = self.get_num_mb(self.layer_recompute)

    def get_inter_DC_comm_time(self):
        param_per_layer = self.get_num_param_per_layer()
        param_per_block = (
            param_per_layer * self.cfg.num_hidden_layers / self.pp / self.num_chunks
        )

        # perform reduce/scatter based RS,AG.
        comm_vol = 2 * param_per_block / self.num_DC
        return comm_vol / self.DC_comm_bandwidth + self.DC_comm_latency

    def get_intra_DC_comm_time(self):
        param_per_layer = self.get_num_param_per_layer()
        param_per_block = (
            param_per_layer * self.cfg.num_hidden_layers / self.pp / self.num_chunks
        )
        assert self.dp % self.num_DC == 0
        dev_per_DC = self.cp * self.dp // self.num_DC

        comm_vol = 2 * param_per_block
        return (dev_per_DC - 1) / dev_per_DC * comm_vol / self.DC_intra_comm_bandwidth

    def get_num_mb(self, layer_recompute=False):
        # use up the available GPU memory
        avail_gpu_mem = self.get_avail_gpu_mem_for_activation()
        if layer_recompute:
            M_F = self.get_forward_block_mem_incr(layer_recompute)
            act_per_layer = (
                self.get_forward_block_mem_incr(False) / self.get_num_layers_per_block()
            )
            num_mb_max = math.floor(
                (avail_gpu_mem - act_per_layer) / M_F / self.num_chunks
            )
        else:
            M_F = self.get_forward_block_mem_incr(layer_recompute)
            num_mb_max = math.floor(avail_gpu_mem / M_F / self.num_chunks)
        used_gpu_mem = num_mb_max * M_F * self.num_chunks + self.simcfg.gpu_mem_bytes - avail_gpu_mem
        return num_mb_max, used_gpu_mem

    def get_tokens_per_iteration(self):
        return self.seq_len * self.mbs * self.num_mb * self.dp

    def get_total_computation_time_per_device(self):
        T_F = self.get_forward_block_runtime()
        T_B = self.get_dgrad_block_runtime() + self.get_wgrad_block_runtime()
        if self.layer_recompute:
            T_B += T_F

        return (T_F + T_B) * self.num_chunks * self.num_mb

    def get_simulated_runtime(self):
        # inaccurate
        T_C = self.get_inter_DC_comm_time() + self.get_intra_DC_comm_time()
        T_F = self.get_forward_block_runtime()
        T_B = self.get_dgrad_block_runtime() + self.get_wgrad_block_runtime()
        if self.layer_recompute:
            T_B += T_F
        return (
            3 * self.pp * T_C
            + (self.num_chunks - 1) * max(self.num_mb * T_F, self.pp * T_C)
            + (self.num_chunks - 1) * max(self.num_mb * T_B, self.pp * 2 * T_C)
            + self.num_mb * T_F
            + self.num_mb * T_B
        )

    def get_milp_sol_runtime(self, time_limit: int = 20):
        """
        Assume the following routine: intra-DC RS, inter-DC RS, inter-DC AG, intra-DC AG.
        Inter-DC comm should be serialized to mimic the limited inter-DC link. Can be overlapped with intra-DC comm.
        Intra-DC comm should not overlap with other intra-DC comm, but can overlap with inter-DC comm.
        Build this scheduling problem as a MILP.
        """
        n_pp = self.pp
        n_c = self.num_chunks
        n_mb = self.num_mb

        T_F_block = self.get_forward_block_runtime()
        T_B_block = self.get_dgrad_block_runtime() + self.get_wgrad_block_runtime()
        if self.layer_recompute:
            T_B_block += T_F_block
        T_intra = self.get_intra_DC_comm_time()
        T_inter = self.get_inter_DC_comm_time()

        (T_F_block, T_B_block, T_intra, T_inter), time_scale_factor = scale_to_int(
            [T_F_block, T_B_block, T_intra, T_inter]
        )
        prob = LpProblem("BFSPP", LpMinimize)

        n_nodes = n_pp * n_c * 2

        def get_pp_dev(node_id):
            return node_id // (n_c * 2)

        def get_chunk(node_id):
            return (node_id // 2) % n_c

        def get_block_type(node_id):
            return node_id % 2

        # compute block end time
        E: Dict[Tuple, LpVariable] = {}
        # completion time of intra-DC comm
        C_intra: Dict[Tuple, LpVariable] = {}
        # completion time of inter-DC comm
        C_inter: Dict[Tuple, LpVariable] = {}

        for node_id in range(n_nodes):
            p = get_pp_dev(node_id)
            c = get_chunk(node_id)
            block_type = get_block_type(node_id)
            # 0: forward / AG
            # 1: backward / RS

            E[(p, c, block_type)] = LpVariable(
                f"E_{p}_{c}_{block_type}", 0, None, cat="Continuous"
            )
            C_intra[(p, c, block_type)] = LpVariable(
                f"C_intra_{p}_{c}_{block_type}", 0, None, cat="Continuous"
            )

            C_inter[(p, c, block_type)] = LpVariable(
                f"C_inter_{p}_{c}_{block_type}", 0, None, cat="Continuous"
            )

        # ordering of intra-DC comm
        O_intra: List[Dict[Tuple, LpVariable]] = [{} for _ in range(n_pp)]
        for node_l in range(n_nodes):
            for node_r in range(node_l):
                p_l = get_pp_dev(node_l)
                p_r = get_pp_dev(node_r)
                c_l = get_chunk(node_l)
                c_r = get_chunk(node_r)
                if p_l != p_r:
                    continue
                t_l = get_block_type(node_l)
                t_r = get_block_type(node_r)
                O_intra[p_l][(c_l, t_l, c_r, t_r)] = LpVariable(
                    f"O_intra_{p_l}_{c_l}_{t_l}_{c_r}_{t_r}", 0, 1, cat="Binary"
                )
                O_intra[p_r][(c_r, t_r, c_l, t_l)] = (
                    1 - O_intra[p_l][(c_l, t_l, c_r, t_r)]
                )

        # ordering of inter-DC comm
        O_inter: Dict[Tuple, LpVariable] = {}
        for node_l in range(n_nodes):
            for node_r in range(node_l):
                p_l = get_pp_dev(node_l)
                p_r = get_pp_dev(node_r)
                c_l = get_chunk(node_l)
                c_r = get_chunk(node_r)
                t_l = get_block_type(node_l)
                t_r = get_block_type(node_r)
                O_inter[(p_l, c_l, t_l, p_r, c_r, t_r)] = LpVariable(
                    f"O_inter_{p_l}_{c_l}_{t_l}_{p_r}_{c_r}_{t_r}", 0, 1, cat="Binary"
                )
                O_inter[(p_r, c_r, t_r, p_l, c_l, t_l)] = (
                    1 - O_inter[(p_l, c_l, t_l, p_r, c_r, t_r)]
                )

        pseudo_inf = (
            n_c * n_mb * (T_F_block + T_B_block) + 2 * (T_intra + T_inter) * n_c * n_pp
        )

        # constraints

        # first block (first backward block)
        prob += E[(n_pp - 1, n_c - 1, 1)] >= n_mb * T_B_block

        for node_id in range(n_nodes):
            p = get_pp_dev(node_id)
            c = get_chunk(node_id)
            block_type = get_block_type(node_id)

            prob += C_inter[(p, c, 0)] >= C_inter[(p, c, 1)] + T_inter

            if block_type == 0:
                # comm deps
                prob += C_intra[(p, c, 0)] >= C_inter[(p, c, 0)] + T_intra
                prob += E[(p, c, 0)] >= C_intra[(p, c, 0)] + T_F_block * n_mb

                if p > 0:
                    # cross device deps
                    prob += E[(p, c, 0)] >= E[(p - 1, c, 0)] + T_F_block
                if c > 0:
                    # on device deps
                    prob += E[(p, c, 0)] >= E[(p, c - 1, 0)] + T_F_block * n_mb

                    if p == 0:
                        # cross chunk deps
                        prob += E[(0, c, 0)] >= T_F_block + E[(n_pp - 1, c - 1, 0)]
            else:
                assert block_type == 1
                # comm deps
                prob += C_intra[(p, c, 1)] >= E[(p, c, 1)] + T_intra
                prob += C_inter[(p, c, 1)] >= C_intra[(p, c, 1)] + T_inter

                if p < n_pp - 1:
                    # cross device deps
                    prob += E[(p, c, 1)] >= E[(p + 1, c, 1)] + T_B_block
                if c < n_c - 1:
                    # on device deps
                    prob += E[(p, c, 1)] >= E[(p, c + 1, 1)] + T_B_block * n_mb

                    # cross chunk deps
                    if p == n_pp - 1:
                        prob += E[(n_pp - 1, c, 1)] >= T_B_block + E[(0, c + 1, 1)]

        # ordering constraints
        for node_l in range(n_nodes):
            for node_r in range(n_nodes):
                if node_l == node_r:
                    continue

                p_l, c_l, t_l = (
                    get_pp_dev(node_l),
                    get_chunk(node_l),
                    get_block_type(node_l),
                )
                p_r, c_r, t_r = (
                    get_pp_dev(node_r),
                    get_chunk(node_r),
                    get_block_type(node_r),
                )

                # intra DC deps
                if p_l == p_r:
                    prob += (
                        C_intra[(p_l, c_l, t_l)]
                        >= C_intra[(p_r, c_r, t_r)]
                        + T_intra
                        - pseudo_inf * O_intra[p_l][(c_l, t_l, c_r, t_r)]
                    )
                # inter DC deps
                prob += (
                    C_inter[(p_l, c_l, t_l)]
                    >= C_inter[(p_r, c_r, t_r)]
                    + T_inter
                    - pseudo_inf * O_inter[(p_l, c_l, t_l, p_r, c_r, t_r)]
                )

        res = LpVariable("res")
        prob += res >= E[(n_pp - 1, n_c - 1, 0)]
        prob.setObjective(res)

        with gp.Env(params=gurobi_options) as env:
            solver = pulp.GUROBI(
                mip=True,
                msg=False,
                warmStart=False,
                gapRel=1e-6,
                timeLimit=time_limit,
                env=env,
            )
            status = prob.solve(solver)
            print(f"Status: {LpStatus[status]}")
            solver.close()

        return value(res) / time_scale_factor

    def get_tokens_per_second(self, milp=False):
        if milp:
            return self.get_tokens_per_iteration() / self.get_milp_sol_runtime()
        else:
            return self.get_tokens_per_iteration() / self.get_simulated_runtime()

    def get_tokens_per_second_per_device(self, milp=False):
        return self.get_tokens_per_second(milp=milp) / self.pp / self.dp / self.tp / self.cp


def get_best_BFSPP_schedule(
    sim_cfg: SimConfig,
) -> Tuple[int | float, int]:
    res_unit_throughput = 0
    
    assert sim_cfg.model_cfg.num_hidden_layers % sim_cfg.pp == 0
    num_layers = sim_cfg.model_cfg.num_hidden_layers
    pp = sim_cfg.pp
    for num_chunks in range(1, num_layers // pp):
        if num_layers % (num_chunks * pp) != 0:
            continue
        new_sim_cfg = deepcopy(sim_cfg)
        new_sim_cfg.num_chunks = num_chunks
        sim_cfg_gen = BFSPPSimCfgGen(new_sim_cfg)
        unit_throughput = sim_cfg_gen.get_tokens_per_second_per_device()
        res_unit_throughput = max(res_unit_throughput, unit_throughput)
        res_peak_mem = sim_cfg_gen.get_num_mb(sim_cfg_gen.layer_recompute)[1]
        
    return res_unit_throughput, res_peak_mem


class Simulator:
    def __init__(self, sim_cfg: SimConfig):
        self.sim_cfg = sim_cfg
    
    def benchmark_cross_DC_PP(self):
        def _mem_allowed(sys_cfg: SystemConfig, schedule: List[List[TaskNode]]):
            M_F, M_B, M_W, M_Limit = sys_cfg.M_F, sys_cfg.M_B, sys_cfg.M_W, sys_cfg.M_Limit
            dev_mem_usage = [[] for _ in range(sim_cfg_gen.pp)]
            for device_id, device_tasks in enumerate(schedule):
                dev_mem_usage[device_id].append(M_F[device_id])
                for task in device_tasks[1:]:
                    if task.task_type == "F":
                        dev_mem_usage[device_id].append(dev_mem_usage[device_id][-1] + M_F[device_id])
                    elif task.task_type == "B":
                        dev_mem_usage[device_id].append(dev_mem_usage[device_id][-1] + M_B[device_id])
                    elif task.task_type == "W":
                        dev_mem_usage[device_id].append(dev_mem_usage[device_id][-1] + M_W[device_id])
            max_mem_usage = max([max(mem_usage) for mem_usage in dev_mem_usage])
            if max_mem_usage > M_Limit[device_id]:
                return False, max_mem_usage
            return True, max_mem_usage
            
            
        
        # pipeline, chunks, B W split
        pipe_list = [
            (OneFOneBPipeline, 1, False),
            (GpipePipeline, 1, False),
            (Interleaved1F1BPipeline, 2, False),
            (Hanayo1F1BPipeline, 2, False),
            (ZBH1Pipeline, 1, True),
            (HeuristicWaveZBPipeline, 2, True),
        ]
        result_unit_throughput, result_mem = {}, {}
        for pipe_cls, num_chunks, is_wgrad_split in pipe_list:
            sim_cfg = deepcopy(self.sim_cfg)
            sim_cfg.num_chunks = num_chunks
            sim_cfg_gen = SimCfgGen(sim_cfg)
            sys_cfg = sim_cfg_gen.get_system_config(combine_B_W=not is_wgrad_split, round_to_int=False)
            pp = pipe_cls(sys_cfg)
            pp.schedule()
            pp.solve_dependencies()
            schedule = pp.get_device_scheduled_tasks()
            pp.print_schedule(save=True)
            name = pp.pipeline_name()
            runtime = sim_cfg_gen.get_total_runtime_with_DP_comm(schedule)
            mem_in_range, max_mem_usage = _mem_allowed(sys_cfg, schedule)
            result_mem[name] = max_mem_usage
            result_unit_throughput[name] = sim_cfg_gen.get_global_batch_size_in_tokens() / runtime / sim_cfg_gen.get_num_gpus()
            
            # if mem_in_range:
            #     print(f"{pipe_cls.__name__} runtime: {runtime}, max mem usage: {max_mem_usage}")
            # else:
            #     print(f"{pipe_cls.__name__} OOM runtime: {runtime}, max mem usage: {max_mem_usage}")
            
        # bfs_unit_throughput, bfs_peak_mem = get_best_BFSPP_schedule(sim_cfg)
        # result_unit_throughput["BFSPP(lower bound)"] = bfs_unit_throughput
        # result_mem["BFSPP(lower bound)"] = bfs_peak_mem
        
        return result_unit_throughput, result_mem
            
if __name__ == "__main__":
    sim = SimConfig(
        model_cfg=CUSTOM_SIZE_TO_CONFIG[400],
        seq_len=8192,
        mbs=1,
        tp=8,
        pp=16,
        dp=128,
        cp=1,
        num_mb_per_pp_stage=2,
        num_chunks=2,
        gpu_mem_bytes=96 * 1024**3,  # around 96GB
        gpu_avg_perf_flops=350 * 10**12,
        num_DC=2,
        intra_DC_bandwidth=320 * 10**9,
        DC_comm_latency=0.01,
        DC_comm_bandwidth=32 * 10**9,
    )
    simulate = Simulator(sim)
    simulate.benchmark_cross_DC_PP()
    