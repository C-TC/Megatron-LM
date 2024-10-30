from collections import defaultdict
import contextlib
from copy import deepcopy
import json
import os
import time
from typing import Dict, Iterator, List, Tuple, Union
from datetime import timedelta

import numpy as np

from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
    HeuristicWaveZBPipeline,
    HeuristicWaveZBPipelineV2,
    HeuristicZBUDPipeline,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline_config import (
    SystemConfig,
)
from megatron.core.pipeline_parallel.cdc_scheduler.wgrad_store import WGradStore
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import (
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_step,
    backward_step,
)
from megatron.core.utils import get_model_config, get_model_type
from megatron.training import get_args
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator import (
    Pipeline,
    get_default_static_schedule,
)
from megatron.core.pipeline_parallel.cdc_scheduler.execution_planner import (
    CommEvent,
    CommEventType,
    ComputeTask,
    ExecutionPlanner,
    TaskEvent,
)


_CDC_PP_SCHEDULER = None


def get_cdc_pp_scheduler():
    global _CDC_PP_SCHEDULER
    if _CDC_PP_SCHEDULER is None:
        args = get_args()
        _CDC_PP_SCHEDULER = CDCPPScheduler(args)
    return _CDC_PP_SCHEDULER


def tuple_keys_to_str(d):
    """Recursively converts tuple keys to strings."""
    return {
        str(k): (tuple_keys_to_str(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def str_keys_to_tuple(d):
    """Recursively converts string keys that represent tuples back to tuples."""

    def try_convert_key(k):
        try:
            return eval(k) if k.startswith("(") and k.endswith(")") else k
        except:
            return k

    return {
        try_convert_key(k): (str_keys_to_tuple(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def process_pp_stages_per_dc(pp_stages_per_dc, pp_size, num_dc):
    if len(pp_stages_per_dc) == 0:
        # naive split
        ret = [pp_size // num_dc] * num_dc
        for i in range(pp_size % num_dc):
            ret[i] += 1
    elif len(pp_stages_per_dc) == 1:
        ret = [pp_stages_per_dc[0]] * num_dc
    assert (
        sum(ret) == pp_size
    ), f"pp_stages_per_dc {ret} does not sum to pp_size {pp_size}"
    return ret


def get_or_set_pp_io_tensor(tensor_dict: Dict, key, config, tensor_shape):
    return tensor_dict.setdefault(
        key,
        torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        ),
    )


class CDCDynamicScheduleGenerator:
    def __init__(
        self,
        args,
        schedule_type: str,
        num_microbatch: int,
        profile_result_path: str = None,
    ) -> None:
        self.args = args
        self.schedule_type = schedule_type
        # TODO: cdc: support more schedule types
        assert self.schedule_type in [
            "wave",
            "ud",
        ], "Currently only support wave schedule"
        self.num_chunks = 1 if self.schedule_type == "ud" else 2
        self.num_microbatch = num_microbatch
        # self.profile_result_path = profile_result_path
        self.profile_result_path = profile_result_path

        with open(os.path.join(self.profile_result_path, "total.json"), "r") as f:
            profile_result = json.load(f)
        self.T_F_list = profile_result["T_F"]
        self.T_B_list = profile_result["T_B"]
        self.T_W_list = profile_result["T_W"]
        self.T_C_matrix = np.array(profile_result["T_C"])
        self.M_F_list = profile_result["M_F"]
        self.M_B_list = profile_result["M_B"]
        self.M_W_list = profile_result["M_W"]
        self.M_Limit_list = profile_result["M_Limit"]

        self.pp_size = args.pipeline_model_parallel_size
        assert len(self.T_F_list) == self.pp_size

        self.pipeline: Pipeline | None = None

        if args.cdc_latency_as_F_blocks > 0.0:
            T_F_block = np.mean(self.T_F_list)
            new_latency = T_F_block * args.cdc_latency_as_F_blocks * self.num_chunks
            self.override_T_C(new_latency)

        if args.dynamic_mem_factor > 0.0:
            self.override_M_Limit(args.dynamic_mem_factor)

        if dist.get_rank() == 0:
            self.rank_zero = True
        else:
            self.rank_zero = False
        self.dump_profile()

    def dump_profile(self) -> None:
        if not self.rank_zero:
            return
        cur_time = time.time()
        with open(
            os.path.join(self.profile_result_path, f"override_{cur_time}.json"), "w"
        ) as f:
            json.dump(
                {
                    "T_F": self.T_F_list,
                    "T_B": self.T_B_list,
                    "T_C": self.T_C_matrix.tolist(),
                    "T_W": self.T_W_list,
                    "M_F": self.M_F_list,
                    "M_B": self.M_B_list,
                    "M_W": self.M_W_list,
                    "M_Limit": self.M_Limit_list,
                    "num_mb": self.num_microbatch,
                },
                f,
            )

    def override_T_C(self, latency_sec) -> None:
        new_T_C = np.zeros((self.pp_size, self.pp_size))

        pp_stages_per_dc = process_pp_stages_per_dc(
            self.args.pp_stages_per_dc, self.pp_size, self.args.num_dc
        )
        dc_boundaries = [
            sum(pp_stages_per_dc[:i]) for i in range(1, self.args.num_dc + 1)
        ]
        for boundary in dc_boundaries:
            src = (boundary - 1) % self.pp_size
            dst = boundary % self.pp_size
            new_T_C[src, dst] = latency_sec
            new_T_C[dst, src] = latency_sec

        self.T_C_matrix = new_T_C

    def override_M_Limit(self, mem_factor: float) -> None:
        for i in range(self.pp_size):
            new_mem_limit = int(
                self.pp_size * self.num_chunks * mem_factor * self.M_F_list[i] * 1.05
            )
            self.M_Limit_list[i] = min(new_mem_limit, self.M_Limit_list[i])

    def generate_schedule_from_profile(self) -> None:
        if self.schedule_type == "wave":
            num_chunks = 2
            sys_cfg = SystemConfig(
                T_F=self.T_F_list,
                T_B=self.T_B_list,
                T_C=self.T_C_matrix,
                T_W=self.T_W_list,
                M_F=self.M_F_list,
                M_B=self.M_B_list,
                M_W=self.M_W_list,
                M_Limit=self.M_Limit_list,
                num_devices=self.pp_size,
                num_microbatches=self.num_microbatch,
                num_chunks=num_chunks,
            )
            candidates: List[Pipeline] = []
            try:
                wave_v1 = HeuristicWaveZBPipeline(sys_cfg)
                wave_v1.schedule()
                wave_v1.solve_dependencies()
            except Exception as e:
                if self.rank_zero:
                    print(f"Error when generating HeuristicWaveZBPipeline: {sys_cfg}")
                    print(e)
                wave_v1 = None
            if wave_v1 is not None:
                candidates.append(wave_v1)

            for bootstrap_soft_bound in [True, False]:
                for aux_tear_down_opt in [True, False]:
                    for aux_w_if_b_mem_limited in [True, False]:
                        cfg = deepcopy(sys_cfg)
                        cfg.bootstrap_soft_bound = bootstrap_soft_bound
                        cfg.aux_tear_down_opt = aux_tear_down_opt
                        cfg.aux_w_if_b_mem_limited = aux_w_if_b_mem_limited
                        try:
                            cand_pipe = HeuristicWaveZBPipelineV2(cfg)
                            cand_pipe.schedule()
                            cand_pipe.solve_dependencies()
                        except Exception as e:
                            if self.rank_zero:
                                print(
                                    f"Error when generating HeuristicWaveZBPipelineV2: {cfg}"
                                )
                                print(e)
                            cand_pipe = None
                        if cand_pipe is not None:
                            candidates.append(cand_pipe)
            # select the best pipeline
            best_pipeline = None
            best_time = float("inf")
            for cur_pipe in candidates:
                pp_runtime = cur_pipe.get_schedule_time(device_wise=True)
                if pp_runtime < best_time:
                    best_time = pp_runtime
                    best_pipeline = cur_pipe
            assert best_pipeline is not None
            self.pipeline = best_pipeline

        elif self.schedule_type == "ud":
            num_chunks = 1
            sys_cfg = SystemConfig(
                T_F=self.T_F_list,
                T_B=self.T_B_list,
                T_C=self.T_C_matrix,
                T_W=self.T_W_list,
                M_F=self.M_F_list,
                M_B=self.M_B_list,
                M_W=self.M_W_list,
                M_Limit=self.M_Limit_list,
                num_devices=self.pp_size,
                num_microbatches=self.num_microbatch,
                num_chunks=num_chunks,
            )
            candidates: List[Pipeline] = []
            for aux_interleave_priority in [True, False]:
                cfg = deepcopy(sys_cfg)
                cfg.aux_interleave_priority = aux_interleave_priority
                try:
                    cand_pipe = HeuristicZBUDPipeline(cfg)
                    cand_pipe.schedule()
                    cand_pipe.solve_dependencies()
                except Exception as e:
                    if self.rank_zero:
                        print(f"Error when generating HeuristicZBUDPipeline: {cfg}")
                        print(e)
                    cand_pipe = None
                if cand_pipe is not None:
                    candidates.append(cand_pipe)
            # select the best pipeline
            best_pipeline = None
            best_time = float("inf")
            for cur_pipe in candidates:
                pp_runtime = cur_pipe.get_schedule_time(device_wise=True)
                if pp_runtime < best_time:
                    best_time = pp_runtime
                    best_pipeline = cur_pipe
            assert best_pipeline is not None
            self.pipeline = best_pipeline

        if dist.get_rank() == 0:
            self.pipeline.print_schedule(
                name=str(time.time()), save=True, save_path=self.profile_result_path
            )

    def get_schedule(self) -> Pipeline:
        self.generate_schedule_from_profile()
        return self.pipeline

    def update_latency_ms(self, latency_ms):
        self.override_T_C(latency_ms / 1000)
        self.dump_profile()


class CDCPPScheduler:
    """

    Notice that chunk_id == virtual_pipeline_model_parallel_rank
    TODO: support checkpointing

    """

    def __init__(self, args) -> None:
        self.args = args
        self.config = None
        self.use_static_schedule = False
        self.use_dynamic_schedule = False
        self.pp_schedule: Pipeline = None

        self.cdc_verbose_print = args.cdc_verbose_print
        self.cdc_print_rank = args.cdc_print_rank

        pp_size = args.pipeline_model_parallel_size
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_rank = pp_rank

        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.cdc_log_profile = (
            True if self.tp_rank == 0 and self.dp_rank == 0 else False
        )

        num_microbatch = get_num_microbatches()

        self.profile_result_path = args.tensorboard_dir
        self.profile_result_file = os.path.join(args.tensorboard_dir, f"{pp_rank}.json")
        self.enable_cdc_profile = args.enable_cdc_profile
        if self.enable_cdc_profile:
            assert (
                args.cdc_profile_iter < args.train_iters
            ), "Profile iteration should be less than total iterations"
        # (mb, chunk, type) -> [time, memory before, memory after]
        self.cdc_compute_profile_dict = {}
        # four lists of [alpha, beta] x [to_next, to_prev]
        self.cdc_comm_profiles = None
        # basic memory: parameter, grad, optimizer state
        self.cdc_base_memory = -1
        # parameters in chunk
        self.cdc_chunk_parameters: Dict[int, int] = {}
        # other info to log: chunk -> (vocabembedding, lmhead, numlayers)
        self.cdc_layer_info = {}

        # static schedule
        if args.static_schedule is not None:
            assert args.dynamic_schedule is None
            self.use_static_schedule = True
            self.pp_schedule = get_default_static_schedule(
                args.static_schedule, pp_size, num_microbatch
            )
        else:
            assert args.dynamic_schedule is not None
            assert (
                self.enable_cdc_profile is False
            ), "CDC profile should be enabled with static schedule"
            self.use_dynamic_schedule = True
            self.dynamic_schedule_type = args.dynamic_schedule
            # profile_result_path should exist
            assert os.path.exists(
                self.profile_result_path
            ), f"Profile result path {self.profile_result_path} does not exist"
            self.pp_schedule_generator = CDCDynamicScheduleGenerator(
                args,
                self.dynamic_schedule_type,
                num_microbatch,
                self.profile_result_path,
            )
            self.pp_schedule = self.pp_schedule_generator.get_schedule()

        if args.cdc_latency_as_F_blocks:
            # for both static and dynamic, adjust the cdc latency from profile

            # profile_result_path should exist
            assert os.path.exists(
                self.profile_result_path
            ), f"Profile result path {self.profile_result_path} does not exist"

            with open(os.path.join(self.profile_result_path, "total.json"), "r") as f:
                profile_result = json.load(f)
            T_F_list = profile_result["T_F"]
            mean_T_F = np.mean(T_F_list)
            new_cdc_latency = (
                pp_size
                * self.pp_schedule.sys_config.num_chunks
                * mean_T_F
                * args.cdc_latency_as_F_blocks
            )
            # s -> ms
            args.cdc_latency = new_cdc_latency * 1000
            self.cdc_print(
                f"Adjusted cdc latency: {args.cdc_latency} second, T_F:{mean_T_F}, pp_size:{pp_size}, num_chunks:{self.pp_schedule.sys_config.num_chunks}, cdc_latency_as_F_blocks:{args.cdc_latency_as_F_blocks}"
            )

        self.pp_execution_planner = ExecutionPlanner(self.pp_schedule)
        self.pp_execution_planner.generate_execution_plan()
        self.pp_execution_plan: List[List[ComputeTask]] = (
            self.pp_execution_planner.execution_plan
        )

        self.pp_execution_plan_cur_device: List[ComputeTask] = self.pp_execution_plan[
            pp_rank
        ]

        self.cdc_print(
            f"execution_plan: \n {self.pp_execution_planner.print_execution_plan()}",
            rank=0,
            verbose=2,
        )

        self.cdc_print(f"layer distribution: {self.get_num_layers_in_chunk()}")

        # cross-DC
        self.num_dc = args.num_dc
        self.cdc_latency = args.cdc_latency

        # decide whether to insert latency.
        self.cdc_recv_prev = False
        self.cdc_recv_next = False

        if self.num_dc > 1 and self.cdc_latency > 0:
            self.pp_stages_per_dc = process_pp_stages_per_dc(
                args.pp_stages_per_dc, pp_size, self.num_dc
            )

            # check if ocurrent rank on the boundary of DCs
            dc_boundaries = [
                sum(self.pp_stages_per_dc[:i]) for i in range(1, self.num_dc + 1)
            ]
            if pp_rank + 1 in dc_boundaries:
                # check if any recv next events in the plan
                for task in self.pp_execution_plan_cur_device:
                    for event in task.pre_events + task.post_events:
                        if (
                            isinstance(event, CommEvent)
                            and event.type == CommEventType.POST_RECV_NEXT
                        ):
                            self.cdc_recv_next = True
                            break
            if pp_rank in [x % pp_size for x in dc_boundaries]:
                # check if any recv prev events in the plan
                for task in self.pp_execution_plan_cur_device:
                    for event in task.pre_events + task.post_events:
                        if (
                            isinstance(event, CommEvent)
                            and event.type == CommEventType.POST_RECV_PREV
                        ):
                            self.cdc_recv_prev = True
                            break
            self.cdc_print(
                f"latency injection: cdc_recv_prev: {self.cdc_recv_prev}, cdc_recv_next: {self.cdc_recv_next}"
            )

        if self.enable_cdc_profile:
            self.cdc_comm_profiles = self.pp_benchmark()

        self.wgrad_split = any(
            [task.task_desc.type == "W" for task in self.pp_execution_plan_cur_device]
        )

        self.wgrad_store = WGradStore() if self.wgrad_split else None

        # p2p handles, (mb, chunk, type)
        self.send_next_reqs: Dict[Tuple, dist.Work] = {}
        self.recv_next_reqs: Dict[Tuple, dist.Work] = {}
        self.send_prev_reqs: Dict[Tuple, dist.Work] = {}
        self.recv_prev_reqs: Dict[Tuple, dist.Work] = {}

        # cached tensors (mb, chunk)
        self.input_tensors: Dict[Tuple, torch.Tensor] = {}
        self.output_tensors: Dict[Tuple, torch.Tensor] = {}
        self.output_tensor_grads: Dict[Tuple, torch.Tensor] = {}
        self.input_tensor_grads: Dict[Tuple, torch.Tensor] = {}

        # token count
        self.total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

        # grad sync
        self.no_sync_func = None
        self.no_sync_context = None

        self.validate_args()

        self.exp_logging = args.cdc_exp_logging
        if self.tp_rank != 0 or self.dp_rank != 0 or self.pp_rank != 0:
            self.exp_logging_my_rank = False
        else:
            self.exp_logging_my_rank = True
        self.exp_logging_path = args.tensorboard_dir
        self.exp_logging_start_iter = 2
        self.exp_logging_end_iter = args.exit_interval - 1
        self.exp_logging_iter_time = defaultdict(list)
        self.exp_logging_max_allocated_mem = defaultdict(list)
        self.cdc_exp_override_latency = False
        self.cdc_exp_override_latency_ms = args.cdc_exp_override_latency_ms
        self.cdc_exp_override_latency_test_iters = (
            args.cdc_exp_override_latency_test_iters
        )
        # if self.exp_logging:
        #     assert self.exp_logging_start_iter + 10 < self.exp_logging_end_iter

        if len(self.cdc_exp_override_latency_ms) > 0:
            self.cdc_exp_override_latency = True
            assert not args.cdc_latency_as_F_blocks
            # if iter = <> then change latency.
            self.cdc_exp_override_iter = [
                2 + i * self.cdc_exp_override_latency_test_iters
                for i in range(len(self.cdc_exp_override_latency_ms))
            ]
            self.exp_logging_end_iter = (
                self.cdc_exp_override_iter[-1]
                + self.cdc_exp_override_latency_test_iters
            )
            args.exit_interval = self.exp_logging_end_iter + 1

    def update_schedule_with_latency(self, latency_ms):
        if self.use_static_schedule:
            self.cdc_latency = latency_ms
        else:
            # dynamic schedule
            self.pp_schedule_generator.update_latency_ms(latency_ms)
            self.pp_schedule = self.pp_schedule_generator.get_schedule()
            self.pp_execution_planner = ExecutionPlanner(self.pp_schedule)
            self.pp_execution_planner.generate_execution_plan()
            self.pp_execution_plan: List[List[ComputeTask]] = (
                self.pp_execution_planner.execution_plan
            )

            self.pp_execution_plan_cur_device: List[ComputeTask] = (
                self.pp_execution_plan[self.pp_rank]
            )

            self.cdc_print(
                f"updated execution_plan: \n {self.pp_execution_planner.print_execution_plan()}",
                rank=0,
                verbose=2,
            )
            self.cdc_latency = latency_ms

    def clean_up(self):
        # get ready for the next iteration
        self.send_next_reqs.clear()
        self.recv_next_reqs.clear()
        self.send_prev_reqs.clear()
        self.recv_prev_reqs.clear()

        self.input_tensors.clear()
        self.output_tensors.clear()
        self.output_tensor_grads.clear()
        self.input_tensor_grads.clear()

        self.total_num_tokens.zero_()

        self.no_sync_func = None
        self.no_sync_context = None

    def validate_args(self):
        args = self.args
        # validate args
        assert args.enable_cdcpp_scheduler, "CDCPPScheduler is not enabled"
        assert (
            args.pipeline_model_parallel_size > 1
        ), "CDCPPScheduler is only for pipeline parallelism"
        assert (
            args.pipeline_model_parallel_size % 2 == 0
        ), "Check _initialize_pp_extra_groups_communicators()"
        assert not args.defer_embedding_wgrad_compute
        assert not args.variable_seq_lengths

        if self.use_static_schedule:
            pass

        assert (
            not args.align_grad_reduce
        ), "align_grad_reduce is not supported, therefore grad_sync_func must be None"
        if hasattr(args, "grad_sync_func"):
            # grad_sync_func may not be set now. No align_grad_reduce should be enough.
            assert args.grad_sync_func is None

        assert (
            not args.align_param_gather
        ), "align_param_gather is not supported, therefore param_sync_func must be None"
        if hasattr(args, "param_sync_func"):
            # param_sync_func may not be set now. No align_param_gather should be enough.
            assert args.param_sync_func is None

        assert (
            not args.defer_embedding_wgrad_compute
        ), "defer_embedding_wgrad_compute is not supported"

        assert (
            args.virtual_pipeline_model_parallel_size
            == self.pp_schedule.sys_config.num_chunks
        ), "For compatibility, virtual_pipeline_model_parallel_size is equivalent to number of chunks"

        if self.wgrad_split:
            assert (
                args.gradient_accumulation_fusion
            ), "W-grad split requires gradient accumulation fusion"

    def update_args_and_config(self, args, config):
        # Megatron may add/modify attributes in args and config after initialization.
        self.args = args
        self.config = config
        # crossdc: TODO: enable this
        self.config.deallocate_pipeline_outputs = False

    def get_wgrad_store(self):
        return self.wgrad_store

    def get_layer_offset(self, dev_id, chunk_id):
        layers_list = self.get_num_layers_in_chunk(dev_id=None, chunk_id=None)
        execution_order = self.pp_schedule.get_pipeline_execution_order()
        idx_in_order = execution_order.index((dev_id, chunk_id))
        return sum(layers_list[:idx_in_order])

    def get_num_layers_in_chunk(self, dev_id=None, chunk_id=None):
        # TODO: better model splitting
        # now we treat vocab and lm head as one layer each. and add unbalanced layers to last several chunks
        num_layer = self.args.num_layers
        if self.args.head_tail_as_one_layer:
            num_layer = num_layer - 2
        execution_order = self.pp_schedule.get_pipeline_execution_order()
        total_chunks = len(execution_order)

        assert total_chunks >= 2, "CDC scheduler should be enabled with pp >= 2"
        if total_chunks == 2:
            layer_list = [num_layer // 2, num_layer - num_layer // 2]
        else:
            head_tail_layers = (num_layer + 2) // total_chunks - 1
            rest_layers = num_layer - head_tail_layers * 2
            rest_chunks = total_chunks - 2
            remainder = rest_layers % rest_chunks
            layer_list = [rest_layers // rest_chunks] * rest_chunks
            for i in range(remainder):
                # add to last several chunks
                layer_list[-(i + 1)] += 1
            layer_list = [head_tail_layers] + layer_list + [head_tail_layers]

        assert (dev_id is not None or chunk_id is not None) or (
            dev_id is None and chunk_id is None
        )
        if dev_id is not None and chunk_id is not None:
            return layer_list[execution_order.index((dev_id, chunk_id))]
        else:
            return layer_list

    def _is_last_microbatch_for_model_chunk(
        self, compute_task: ComputeTask, number_of_microbatches
    ):
        # To enable grad sync in backward.
        has_w_blocks = self.wgrad_split

        if has_w_blocks:
            return (
                compute_task.task_desc.mb_id == number_of_microbatches - 1
                and compute_task.task_desc.type == "W"
            )
        else:
            return (
                compute_task.task_desc.mb_id == number_of_microbatches - 1
                and compute_task.task_desc.type == "B"
            )

    def schedule_comm_event(self, event: CommEvent, config, tensor_shape):
        send_next_group = parallel_state.get_pipeline_extra_send_next_group()
        recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        send_prev_group = parallel_state.get_pipeline_extra_send_prev_group()
        recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()

        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()

        assert event.task_type in ["F", "B"]

        if event.task_type == "F":
            send_buffer = get_or_set_pp_io_tensor(
                self.output_tensors, (event.mb_id, event.chunk_id), config, tensor_shape
            )
            recv_buffer = get_or_set_pp_io_tensor(
                self.input_tensors, (event.mb_id, event.chunk_id), config, tensor_shape
            )
        else:
            send_buffer = get_or_set_pp_io_tensor(
                self.input_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )
            recv_buffer = get_or_set_pp_io_tensor(
                self.output_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )

        # block only when benchmarking runtime
        block_host_till_comm_finish = (
            self.enable_cdc_profile
            and self.args.curr_iteration == self.args.cdc_profile_iter
        )

        if event.type == CommEventType.LOCAL_COPY:
            if event.task_type == "F":
                with torch.no_grad():
                    recv_buffer.copy_(
                        self.output_tensors[
                            (event.mb_id, event.prev_task_chunk_id)
                        ].detach()
                    )
            else:
                with torch.no_grad():
                    recv_buffer.copy_(
                        self.input_tensor_grads[
                            (event.mb_id, event.prev_task_chunk_id)
                        ].detach()
                    )
        elif event.type == CommEventType.POST_SEND_NEXT:
            self.send_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self.isend(send_buffer, next_rank, group=send_next_group)
            )
        elif event.type == CommEventType.POST_RECV_NEXT:
            self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self.irecv(recv_buffer, next_rank, group=recv_next_group)
            )
        elif event.type == CommEventType.POST_SEND_PREV:
            self.send_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self.isend(send_buffer, prev_rank, group=send_prev_group)
            )
        elif event.type == CommEventType.POST_RECV_PREV:
            self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                self.irecv(recv_buffer, prev_rank, group=recv_prev_group)
            )
        elif event.type == CommEventType.WAIT_SEND_NEXT:
            handle = self.send_next_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            handle.wait()
        elif event.type == CommEventType.WAIT_RECV_NEXT:
            handle = self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            assert hasattr(
                handle, "wait_with_delay_in_ms"
            ), "Latency injection requires custom pytorch build for wait_with_delay_in_ms"
            if self.cdc_recv_next:
                handle.wait_with_delay_in_ms(timedelta(milliseconds=self.cdc_latency))
            else:
                if block_host_till_comm_finish:
                    handle.wait_with_delay_in_ms(timedelta(milliseconds=0))
                else:
                    handle.wait()
        elif event.type == CommEventType.WAIT_SEND_PREV:
            handle = self.send_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            handle.wait()
        elif event.type == CommEventType.WAIT_RECV_PREV:
            handle = self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            assert hasattr(
                handle, "wait_with_delay_in_ms"
            ), "Latency injection requires custom pytorch build for wait_with_delay_in_ms"
            if self.cdc_recv_prev:
                handle.wait_with_delay_in_ms(timedelta(milliseconds=self.cdc_latency))
            else:
                if block_host_till_comm_finish:
                    handle.wait_with_delay_in_ms(timedelta(milliseconds=0))
                else:
                    handle.wait()
        else:
            raise NotImplementedError()

    def schedule_event(
        self, event: TaskEvent, config, tensor_shape, forward_only, num_microbatches
    ):
        if isinstance(event, CommEvent):
            if forward_only:
                mb_id = event.mb_id
                task_type = event.task_type
                if mb_id >= num_microbatches or task_type != "F":
                    return
            self.schedule_comm_event(event, config, tensor_shape)
        else:
            raise NotImplementedError()

    def schedule_compute_task(
        self,
        compute_task: ComputeTask,
        model,
        data_iterator,
        forward_step_func,
        tensor_shape,
        forward_data_store,
        collect_non_loss_data,
        first_val_step,
        forward_only,
        num_microbatches,
    ):
        config = get_model_config(model[0])

        for pre_event in compute_task.pre_events:
            self.cdc_print(f"pre_event: {pre_event}", verbose=2)
            self.schedule_event(
                pre_event, config, tensor_shape, forward_only, num_microbatches
            )

        task_type = compute_task.task_desc.type
        chunk_id = compute_task.task_desc.chunk_id
        mb_id = compute_task.task_desc.mb_id
        # Important
        parallel_state.set_virtual_pipeline_model_parallel_rank(chunk_id)

        is_first_stage = parallel_state.is_pipeline_first_stage()
        is_last_stage = parallel_state.is_pipeline_last_stage()

        task_type_to_int = {"F": 0, "B": 1, "W": 2}

        if (
            self.enable_cdc_profile
            and self.args.curr_iteration == self.args.cdc_profile_iter
        ):
            if self.cdc_base_memory < 0:
                self.cdc_base_memory = torch.cuda.memory_allocated(
                    device=torch.cuda.current_device()
                )
            # sync default stream
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            mem_before = torch.cuda.memory_allocated(device=torch.cuda.current_device())
            # high precision timer on cpu
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            time_before = time.perf_counter()

        if (
            self.exp_logging_my_rank
            and self.exp_logging_first_mb
            and self.args.curr_iteration >= self.exp_logging_start_iter
        ):
            # start timing before first compute task
            torch.cuda.synchronize()
            self.exp_logging_iter_time[self.cdc_latency].append(time.perf_counter())

        if task_type == "F" and (not forward_only or mb_id < num_microbatches):
            self.cdc_print(
                f"forward_step mb_id: {mb_id}, chunk_id: {chunk_id}", verbose=2
            )
            self.output_tensors[(mb_id, chunk_id)], num_tokens = forward_step(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator[chunk_id],
                model=model[chunk_id],
                num_microbatches=num_microbatches,
                input_tensor=self.input_tensors[(mb_id, chunk_id)]
                if not is_first_stage
                else None,
                forward_data_store=forward_data_store,
                config=config,
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,  # max_outstanding_backprops, num_microbatches_with_partial_activation_checkpoints
                is_first_microbatch=check_first_val_step(
                    first_val_step, forward_only, mb_id == 0
                ),
                current_microbatch=mb_id,
                encoder_decoder_xattn=False,
            )
            self.total_num_tokens += num_tokens.item()
            # The following is buggy. crossdc: TODO: another way to deallocate?
            # if is_last_stage:
            #     # no need to cache output tensor at last stage
            #     self.output_tensors[(mb_id, chunk_id)] = None

        elif task_type == "B" and not forward_only:
            # Only training. In eval, we skip backward.

            self.cdc_print(
                f"backward_step mb_id: {mb_id}, chunk_id: {chunk_id}", verbose=2
            )
            # enable grad sync for the last microbatch
            if self._is_last_microbatch_for_model_chunk(compute_task, num_microbatches):
                self.enable_grad_sync()

            output_tensor_grad = (
                self.output_tensor_grads[(mb_id, chunk_id)]
                if not is_last_stage
                else None
            )
            self.input_tensor_grads[(mb_id, chunk_id)] = backward_step(
                input_tensor=self.input_tensors[(mb_id, chunk_id)],
                output_tensor=self.output_tensors[(mb_id, chunk_id)],
                output_tensor_grad=output_tensor_grad,
                model_type=get_model_type(model[chunk_id]),
                config=config,
            )
            # release tensors
            self.input_tensors[(mb_id, chunk_id)] = None
            self.output_tensors[(mb_id, chunk_id)] = None
            self.output_tensor_grads[(mb_id, chunk_id)] = None

            if is_first_stage:
                self.input_tensor_grads[(mb_id, chunk_id)] = None

            if self.wgrad_store is not None:
                self.wgrad_store.finish_collection_wgrad_block()

            # disable grad sync for other microbatches
            if self._is_last_microbatch_for_model_chunk(compute_task, num_microbatches):
                self.disable_grad_sync()

        elif task_type == "W" and not forward_only:
            assert self.wgrad_store is not None

            self.cdc_print(
                f"wgrad_step mb_id: {mb_id}, chunk_id: {chunk_id}", verbose=2
            )

            self.wgrad_store.compute_wgrad_block()

            if self._is_last_microbatch_for_model_chunk(compute_task, num_microbatches):
                # TODO: I guess enable_grad_sync would not enable DP comm here.
                # So, we need to do it manually.
                model_chunk = model[chunk_id]
                assert hasattr(model_chunk, "finish_grad_sync")
                model_chunk.finish_grad_sync()

        if (
            self.enable_cdc_profile
            and self.args.curr_iteration == self.args.cdc_profile_iter
        ):
            # sync default stream
            torch.cuda.default_stream(torch.cuda.current_device()).synchronize()
            time_after = time.perf_counter()
            mem_after = torch.cuda.memory_allocated(device=torch.cuda.current_device())
            self.cdc_compute_profile_dict[
                (mb_id, chunk_id, task_type_to_int[task_type])
            ] = [
                time_after - time_before,
                mem_before,
                mem_after,
            ]

        for post_event in compute_task.post_events:
            self.cdc_print(f"post_event: {post_event}", verbose=2)
            self.schedule_event(
                post_event, config, tensor_shape, forward_only, num_microbatches
            )

    def deallocate_tensor_in_dicts(self):
        # TODO: any better ways?

        # output tensors and input tensor grads
        for (mb_id, chunk_id, task_type), handle in list(
            self.send_next_reqs.items()
        ) + list(self.send_prev_reqs.items()):
            if task_type == "F":
                if (
                    self.output_tensors[(mb_id, chunk_id)] is not None
                    and handle is not None
                    and handle.is_completed()
                ):
                    self.cdc_print(
                        f"deallocate_output_tensor: {mb_id}, {chunk_id}", verbose=2
                    )
                    deallocate_output_tensor(
                        self.output_tensors[(mb_id, chunk_id)],
                        self.config.deallocate_pipeline_outputs,
                    )
            elif task_type == "B":
                if (
                    self.input_tensor_grads[(mb_id, chunk_id)] is not None
                    and handle is not None
                    and handle.is_completed()
                ):
                    self.cdc_print(
                        f"releasing input grad ref: {mb_id}, {chunk_id}", verbose=2
                    )
                    self.input_tensor_grads[(mb_id, chunk_id)] = None

    def setup_grad_sync(self):
        # grad sync
        # Disable async grad reductions
        assert self.config is not None
        self.no_sync_func = self.config.no_sync_func
        # crossdc: chunk based no sync
        if isinstance(self.no_sync_func, list):

            def multi_no_sync():
                stack = contextlib.ExitStack()
                for model_chunk_no_sync_func in self.args.no_sync_func:
                    stack.enter_context(model_chunk_no_sync_func())
                return stack

            self.no_sync_func = multi_no_sync
        if self.no_sync_func is None:
            self.no_sync_func = contextlib.nullcontext
        self.no_sync_context = None

    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            no_sync_context = self.no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None

    def forward_backward_func(
        self,
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,  # unused
        decoder_seq_length: int = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
    ):
        # self.cdc_print(f'first_stage (virtual): {parallel_state.is_pipeline_first_stage(ignore_virtual=True)} ({parallel_state.is_pipeline_first_stage()}), last_stage (virtual): {parallel_state.is_pipeline_last_stage(ignore_virtual=True)} ({parallel_state.is_pipeline_last_stage()})')

        if not forward_only:
            assert num_microbatches == self.pp_schedule.sys_config.num_microbatches
        else:
            assert num_microbatches <= self.pp_schedule.sys_config.num_microbatches

        if self.pp_schedule.has_multiple_chunks():
            assert isinstance(model, list), "Model has multiple chunks"
            assert [isinstance(chunk, torch.nn.Module) for chunk in model]
            assert isinstance(
                data_iterator, list
            ), "Expect each chunk to have its own data iterator"
            config = get_model_config(model[0])
        else:
            if isinstance(model, list):
                assert len(model) == 1, "Model should only have one chunk"
            else:
                model = [model]
            assert isinstance(model[0], torch.nn.Module)
            if isinstance(data_iterator, list):
                assert (
                    len(data_iterator) == 1
                ), "Data iterator should only have one chunk"
            else:
                data_iterator = [data_iterator]
            config = get_model_config(model[0])

        assert all(
            [get_model_type(chunk) != ModelType.encoder_and_decoder for chunk in model]
        )

        if (
            self.enable_cdc_profile
            and self.args.curr_iteration == self.args.cdc_profile_iter
        ):
            if len(self.cdc_chunk_parameters) == 0:
                for chunk_id, chunk in enumerate(model):
                    assert len(model) == self.pp_schedule.sys_config.num_chunks
                    # get number
                    num_params = sum(p.numel() for p in chunk.parameters())
                    self.cdc_chunk_parameters[chunk_id] = num_params
                    self.cdc_print(
                        f"model info: chunk {chunk_id} has {num_params} parameters"
                    )
                    self.cdc_print(
                        f"model info :chunk {chunk_id} model: {chunk.__repr__()} "
                    )

            for chunk_id in range(self.pp_schedule.sys_config.num_chunks):
                if chunk_id not in self.cdc_layer_info:
                    first_stage_rank = self.pp_schedule.get_pipeline_first_stage_rank()
                    last_stage_rank = self.pp_schedule.get_pipeline_last_stage_rank()
                    cur_chunk_has_vocab_embedding = (
                        self.pp_rank == first_stage_rank and chunk_id == 0
                    )
                    cur_chunk_has_lm_head = (
                        self.pp_rank == last_stage_rank
                        and chunk_id == self.pp_schedule.sys_config.num_chunks - 1
                    )
                    num_layers = self.get_num_layers_in_chunk(
                        dev_id=self.pp_rank, chunk_id=chunk_id
                    )
                    self.cdc_layer_info[chunk_id] = (
                        cur_chunk_has_vocab_embedding,
                        cur_chunk_has_lm_head,
                        num_layers,
                    )
                    self.cdc_print(
                        f"model info: chunk {chunk_id} has {num_layers} layers, vocab embedding: {cur_chunk_has_vocab_embedding}, lm head: {cur_chunk_has_lm_head}"
                    )

        forward_data_store = []

        # TODO: need this?
        # # Needed only when gradients are finalized in M-Core
        if config.finalize_model_grads_func is not None and not forward_only:
            embedding_module = clear_embedding_activation_buffer(config, model)

        # TODO: timer

        self.setup_grad_sync()

        self.disable_grad_sync()

        tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
        tensor_shape[0] = (
            tensor_shape[0] // parallel_state.get_context_parallel_world_size()
        )
        if config.sequence_parallel:
            tensor_shape[0] = (
                tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
            )

        # multiply tensor shape dims
        self.pp_comm_size_bytes = (
            tensor_shape[0]
            * tensor_shape[1]
            * tensor_shape[2]
            * torch.tensor([], dtype=config.pipeline_dtype).element_size()
        )

        if self.cdc_exp_override_latency and len(self.cdc_exp_override_iter) > 0:
            if self.args.curr_iteration == self.cdc_exp_override_iter[0]:
                self.update_schedule_with_latency(self.cdc_exp_override_latency_ms[0])
                self.cdc_exp_override_iter.pop(0)
                self.cdc_exp_override_latency_ms.pop(0)

        if self.exp_logging:
            dist.barrier()

        for idx, compute_task in enumerate(self.pp_execution_plan_cur_device):
            # self.cdc_print(f"compute_task: {compute_task}")
            self.exp_logging_first_mb = False
            if self.exp_logging_my_rank:
                if idx == 0:
                    self.exp_logging_first_mb = True
            self.schedule_compute_task(
                compute_task=compute_task,
                model=model,
                data_iterator=data_iterator,
                forward_step_func=forward_step_func,
                tensor_shape=tensor_shape,
                forward_data_store=forward_data_store,
                collect_non_loss_data=collect_non_loss_data,
                first_val_step=first_val_step,
                forward_only=forward_only,
                num_microbatches=num_microbatches,
            )
            self.deallocate_tensor_in_dicts()

        assert self.wgrad_store is None or self.wgrad_store.is_empty()

        self.enable_grad_sync()

        if config.finalize_model_grads_func is not None and not forward_only:
            # If defer_embedding_wgrad_compute is enabled we need to do the
            # weight gradient GEMM's here.
            finish_embedding_wgrad_compute(config, embedding_module)

            # Finalize model grads (perform full grad all-reduce / reduce-scatter for
            # data parallelism, layernorm all-reduce for sequence parallelism, and
            # embedding all-reduce for pipeline parallelism).
            config.finalize_model_grads_func(
                model,
                self.total_num_tokens if config.calculate_per_token_loss else None,
            )

        if (
            self.enable_cdc_profile
            and self.args.curr_iteration == self.args.cdc_profile_iter
        ):
            # write profile result
            if self.cdc_log_profile:
                self.cdc_print(
                    f"cdc_compute_profile_dict: {self.cdc_compute_profile_dict}"
                )
                if self.pp_rank == 0:
                    self.cdc_print(f"cdc_comm_profiles: {self.cdc_comm_profiles}")
                self.cdc_print(f"cdc_base_memory: {self.cdc_base_memory}")
                self.cdc_print(f"cdc_chunk_parameters: {self.cdc_chunk_parameters}")
                self.cdc_print(f"cdc_layer_info: {self.cdc_layer_info}")

                with open(self.profile_result_file, "w") as f:
                    json.dump(
                        {
                            "compute": tuple_keys_to_str(self.cdc_compute_profile_dict),
                            "comm": self.cdc_comm_profiles,
                            "base_mem": self.cdc_base_memory,
                            "params": self.cdc_chunk_parameters,
                            "layer_info": self.cdc_layer_info,
                        },
                        f,
                    )

            dist.barrier()

            # rank 0 concludes the profile result
            if dist.get_rank() == 0:
                json_file_path = self.profile_result_path
                json_results = []
                pp_size = parallel_state.get_pipeline_model_parallel_world_size()
                for i in range(pp_size):
                    with open(os.path.join(json_file_path, f"{i}.json"), "r") as f:
                        json_results.append(json.load(f))

                # crossdc: TODO: rebalance based on profile, heterogeneity
                T_F_list = []
                T_B_list = []
                T_W_list = []
                T_C_matrix = np.zeros((pp_size, pp_size))
                M_F_list = []
                M_B_list = []
                M_W_list = []
                M_Limit_list = []
                base_mem_list = []
                max_gpu_mem = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                for i in range(pp_size):
                    compute_profile = str_keys_to_tuple(json_results[i]["compute"])
                    base_mem_list.append(json_results[i]["base_mem"])
                    T_cur_dev = [[] for _ in range(3)]
                    M_cur_dev = [[] for _ in range(3)]
                    for key, value in compute_profile.items():
                        cur_mb, cur_chunk, cur_type = key
                        compute_time, mem_before, mem_after = value
                        T_cur_dev[cur_type].append(compute_time)
                        M_cur_dev[cur_type].append(mem_after - mem_before)
                    # use min val, since gpu is highy async.
                    # crossdc: TODO: currently we do not differentiate chunks in ppsim
                    T_F_list.append(np.min(T_cur_dev[0]))
                    T_B_list.append(np.min(T_cur_dev[1]))
                    if len(T_cur_dev[2]) > 0:
                        T_W_list.append(np.min(T_cur_dev[2]))
                    else:
                        T_W_list.append(0)
                    M_F_list.append(np.percentile(M_cur_dev[0], 75))
                    if len(M_cur_dev[2]) > 0:
                        M_W_list.append(np.percentile(M_cur_dev[2], 75))
                    else:
                        M_W_list.append(0)
                    M_B_list.append(-M_F_list[-1] - M_W_list[-1])
                    M_Limit_list.append(int((max_gpu_mem - base_mem_list[-1]) * 0.98))
                # T_C
                alpha_to_next, alpha_to_prev, beta_to_next, beta_to_prev = json_results[
                    0
                ]["comm"]
                message_size = self.pp_comm_size_bytes
                for i in range(pp_size):
                    T_C_matrix[i, (i + 1) % pp_size] = (
                        alpha_to_next[i] + beta_to_next[i] * message_size
                    )
                    T_C_matrix[i, (i - 1) % pp_size] = (
                        alpha_to_prev[i] + beta_to_prev[i] * message_size
                    )

                # make T_C symmetric
                T_C_matrix = (T_C_matrix + T_C_matrix.T) / 2
                with open(os.path.join(json_file_path, "total.json"), "w") as f:
                    json.dump(
                        {
                            "T_F": T_F_list,
                            "T_B": T_B_list,
                            "T_W": T_W_list,
                            "T_C": T_C_matrix.tolist(),
                            "M_F": M_F_list,
                            "M_B": M_B_list,
                            "M_W": M_W_list,
                            "M_Limit": M_Limit_list,
                        },
                        f,
                    )

        if (
            self.exp_logging_my_rank
            and self.args.curr_iteration >= self.exp_logging_start_iter
        ):
            torch.cuda.synchronize()
            self.exp_logging_iter_time[self.cdc_latency][-1] = (
                time.perf_counter() - self.exp_logging_iter_time[self.cdc_latency][-1]
            )
            self.exp_logging_max_allocated_mem[self.cdc_latency].append(
                torch.cuda.max_memory_allocated()
            )
            torch.cuda.reset_max_memory_allocated()

        if (
            self.exp_logging
            and self.exp_logging_my_rank
            and self.args.curr_iteration == self.exp_logging_end_iter
        ):
            # write to json
            timpstamp = int(time.time())
            schedule = (
                self.args.static_schedule
                if self.use_static_schedule
                else self.args.dynamic_schedule
            )
            with open(
                os.path.join(self.exp_logging_path, f"exp_{timpstamp}.json"), "w"
            ) as f:
                json.dump(
                    {
                        "iter_time": self.exp_logging_iter_time,
                        "max_mem": self.exp_logging_max_allocated_mem,
                        "config": {
                            "schedule": schedule,
                            "TP": self.args.tensor_model_parallel_size,
                            "PP": self.args.pipeline_model_parallel_size,
                            "DP": self.args.data_parallel_size,
                            "seq_len": self.args.seq_length,
                            "GBS": self.args.global_batch_size,
                            "n_DC": self.args.num_dc,
                            "cdc_latency": self.args.cdc_latency,
                            "cdc_latency_F_blocks": self.args.cdc_latency_as_F_blocks,
                            "cdc_actual_latency": self.cdc_latency,
                            "dyn_mem_factor": self.args.dynamic_mem_factor,
                            "num_layers": self.args.num_layers,
                            "cdc_exp_tf_block_size": self.args.cdc_exp_tf_block_size,
                        },
                    },
                    f,
                )

        self.clean_up()

        self.cdc_print(f"forward_data_store: {forward_data_store}", verbose=2)
        return forward_data_store

    def get_forward_backward_func(self):
        return self.forward_backward_func

    def get_cdc_recv_delay(self) -> int | float:
        return self.cdc_latency

    def send(
        self, tensor: torch.Tensor, dst: int, group: dist.ProcessGroupNCCL | None = None
    ):
        dist.send(tensor, dst, group=group)

    def isend(
        self, tensor: torch.Tensor, dst: int, group: dist.ProcessGroupNCCL | None = None
    ) -> dist.Work:
        return dist.isend(tensor, dst, group=group)

    def recv(
        self, tensor: torch.Tensor, src: int, group: dist.ProcessGroupNCCL | None = None
    ):
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()
        recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        if (src == prev_rank and self.cdc_recv_prev and group is recv_prev_group) or (
            src == next_rank and self.cdc_recv_next and group is recv_next_group
        ):
            work = dist.irecv(tensor, src, group=group)
            assert hasattr(
                work, "wait_with_delay_in_ms"
            ), "Latency injection requires custom pytorch build for wait_with_delay_in_ms"
            work.wait_with_delay_in_ms(timedelta(milliseconds=self.cdc_latency))
        else:
            dist.recv(tensor, src, group=group)

    def irecv(
        self, tensor: torch.Tensor, src: int, group: dist.ProcessGroupNCCL | None = None
    ):
        return dist.irecv(tensor, src, group=group)

    def cdc_print(self, msg: str, rank=None, verbose=1):
        if verbose > self.cdc_verbose_print:
            return

        my_rank = dist.get_rank()
        if rank is not None and my_rank != rank:
            return
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        if self.cdc_print_rank == -1 or my_rank == self.cdc_print_rank:
            print(
                f"[CDC] Global[{my_rank}] TP[{tp_rank}] PP[{pp_rank}] DP[{dp_rank}]:    {msg}"
            )

    def pp_benchmark(self):
        """
        Return:
            alpha_to_next: [pp_size]
            alpha_to_prev: [pp_size]
            beta_to_next: [pp_size]
            beta_to_prev: [pp_size]
        """
        my_rank = parallel_state.get_pipeline_model_parallel_rank()
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        assert pp_size % 2 == 0

        warmup = 2
        num_iters = 10
        message_size_bytes = 2**30

        tensor_small = torch.ones(1, dtype=torch.float32).cuda(
            torch.cuda.current_device()
        )
        tensor_large = torch.ones(message_size_bytes // 4, dtype=torch.float32).cuda(
            torch.cuda.current_device()
        )

        send_next_group = parallel_state.get_pipeline_extra_send_next_group()
        recv_next_group = parallel_state.get_pipeline_extra_recv_next_group()
        send_prev_group = parallel_state.get_pipeline_extra_send_prev_group()
        recv_prev_group = parallel_state.get_pipeline_extra_recv_prev_group()

        torch.cuda.synchronize()
        for _ in range(warmup):
            if my_rank % 2 == 0:
                self.send(tensor_large, next_rank, group=send_next_group)
                self.recv(tensor_large, next_rank, group=recv_next_group)
                self.send(tensor_large, prev_rank, group=send_prev_group)
                self.recv(tensor_large, prev_rank, group=recv_prev_group)
            else:
                self.recv(tensor_large, prev_rank, group=recv_prev_group)
                self.send(tensor_large, prev_rank, group=send_prev_group)
                self.recv(tensor_large, next_rank, group=recv_next_group)
                self.send(tensor_large, next_rank, group=send_next_group)
        torch.cuda.synchronize()

        # [prev, next] x [small, large]
        t_recv_prev = [0.0, 0.0]
        t_recv_next = [0.0, 0.0]
        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.send(tensor, next_rank, group=send_next_group)
                else:
                    self.recv(tensor, prev_rank, group=recv_prev_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 != 0:
                    t_recv_prev[idx] += (end - start) / num_iters

        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.send(tensor, prev_rank, group=send_prev_group)
                else:
                    self.recv(tensor, next_rank, group=recv_next_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 != 0:
                    t_recv_next[idx] += (end - start) / num_iters

        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.recv(tensor, next_rank, group=recv_next_group)
                else:
                    self.send(tensor, prev_rank, group=send_prev_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 == 0:
                    t_recv_next[idx] += (end - start) / num_iters

        for idx, tensor in enumerate([tensor_small, tensor_large]):
            for _ in range(num_iters):
                dist.barrier(group=pp_group)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if my_rank % 2 == 0:
                    self.recv(tensor, prev_rank, group=recv_prev_group)
                else:
                    self.send(tensor, next_rank, group=send_next_group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                if my_rank % 2 == 0:
                    t_recv_prev[idx] += (end - start) / num_iters

        # idx -> idx + 1
        alpha_to_next = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )
        beta_to_next = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )
        # idx -> idx - 1
        alpha_to_prev = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )
        beta_to_prev = torch.zeros(
            pp_size, dtype=torch.float32, device=torch.cuda.current_device()
        )

        # IMPORTANT: only collect on receiver, since latency is only injected on receiver.
        alpha_to_next[(my_rank - 1) % pp_size] = t_recv_prev[0]
        alpha_to_prev[(my_rank + 1) % pp_size] = t_recv_next[0]

        beta_to_next[(my_rank - 1) % pp_size] = (
            t_recv_prev[1] - t_recv_prev[0]
        ) / message_size_bytes
        beta_to_prev[(my_rank + 1) % pp_size] = (
            t_recv_next[1] - t_recv_next[0]
        ) / message_size_bytes

        dist.all_reduce(alpha_to_next, op=dist.ReduceOp.SUM, group=pp_group)
        dist.all_reduce(beta_to_next, op=dist.ReduceOp.SUM, group=pp_group)
        dist.all_reduce(alpha_to_prev, op=dist.ReduceOp.SUM, group=pp_group)
        dist.all_reduce(beta_to_prev, op=dist.ReduceOp.SUM, group=pp_group)

        # average globally
        dist.all_reduce(alpha_to_next, op=dist.ReduceOp.AVG)
        dist.all_reduce(beta_to_next, op=dist.ReduceOp.AVG)
        dist.all_reduce(alpha_to_prev, op=dist.ReduceOp.AVG)
        dist.all_reduce(beta_to_prev, op=dist.ReduceOp.AVG)

        return (
            alpha_to_next.tolist(),
            alpha_to_prev.tolist(),
            beta_to_next.tolist(),
            beta_to_prev.tolist(),
        )
