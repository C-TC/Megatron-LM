import contextlib
from typing import Dict, Iterator, List, Tuple, Union

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
import megatron.core.pipeline_parallel.cdc_scheduler.cdc_comm as cdc_comm


_CDC_PP_SCHEDULER = None


def get_cdc_pp_scheduler():
    global _CDC_PP_SCHEDULER
    if _CDC_PP_SCHEDULER is None:
        args = get_args()
        _CDC_PP_SCHEDULER = CDCPPScheduler(args)
    return _CDC_PP_SCHEDULER


class CDCPPScheduler:
    """

    Notice that chunk_id == virtual_pipeline_model_parallel_rank
    TODO: support checkpointing

    """

    def __init__(self, args) -> None:
        self.args = args
        self.config = None
        self.use_static_schedule = False
        self.pp_schedule: Pipeline = None

        self.cdc_verbose_print = args.cdc_verbose_print
        self.cdc_print_rank = args.cdc_print_rank

        pp_size = args.pipeline_model_parallel_size
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        num_microbatch = get_num_microbatches()
        # static schedule
        if args.static_schedule is not None:
            self.use_static_schedule = True
            self.pp_schedule = get_default_static_schedule(
                args.static_schedule, pp_size, num_microbatch
            )
        else:
            raise NotImplementedError()

        # cross-DC
        self.num_dc = args.num_dc
        self.cdc_latency = args.cdc_latency

        # decide whether to insert hooks in PGs.
        self.cdc_recv_prev = False
        self.cdc_recv_next = False

        if self.num_dc > 1 and self.cdc_latency > 0:
            self.pp_stages_per_dc = args.pp_stages_per_dc
            if len(self.pp_stages_per_dc) == 0:
                # naive split
                self.pp_stages_per_dc = [pp_size // self.num_dc] * self.num_dc
                for i in range(pp_size % self.num_dc):
                    self.pp_stages_per_dc[i] += 1
            elif len(self.pp_stages_per_dc) == 1:
                self.pp_stages_per_dc = [self.pp_stages_per_dc[0]] * self.num_dc
            assert sum(self.pp_stages_per_dc) == pp_size
            assert any(
                [stages > 1 for stages in self.pp_stages_per_dc]
            ), "CDC requires at least one stage per DC due to limitation in cdc comm"

            # check if ocurrent rank on the boundary of DCs
            dc_boundaries = [
                sum(self.pp_stages_per_dc[:i]) for i in range(1, self.num_dc)
            ]
            if pp_rank + 1 in dc_boundaries:
                self.cdc_recv_next = True
            if pp_rank in [(x + 1) % pp_size for x in dc_boundaries]:
                self.cdc_recv_prev = True
            assert not (
                self.cdc_recv_prev and self.cdc_recv_next
            ), "CDC recv cannot be both prev and next"

        self.pp_execution_planner = ExecutionPlanner(self.pp_schedule)
        self.pp_execution_planner.generate_execution_plan()
        self.pp_execution_plan: List[List[ComputeTask]] = (
            self.pp_execution_planner.execution_plan
        )

        self.pp_execution_plan_cur_device: List[ComputeTask] = self.pp_execution_plan[
            pp_rank
        ]

        self.cdc_print(f"execution_plan: \n {self.pp_execution_planner.print_execution_plan()}", rank=0)

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
        # now we treat vocab and lm head as one layer each. and add unbalanced layers to first several chunks
        num_layer = self.args.num_layers
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
                layer_list[i] += 1
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

        def get_or_set_tensor(tensor_dict: Dict, key, config, tensor_shape):
            return tensor_dict.setdefault(
                key,
                torch.empty(
                    tensor_shape,
                    requires_grad=True,
                    device=torch.cuda.current_device(),
                    dtype=config.pipeline_dtype,
                ),
            )

        if event.task_type == "F":
            send_buffer = get_or_set_tensor(
                self.output_tensors, (event.mb_id, event.chunk_id), config, tensor_shape
            )
            recv_buffer = get_or_set_tensor(
                self.input_tensors, (event.mb_id, event.chunk_id), config, tensor_shape
            )
        else:
            send_buffer = get_or_set_tensor(
                self.input_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )
            recv_buffer = get_or_set_tensor(
                self.output_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )

        if event.type == CommEventType.POST_SEND_NEXT:
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
            handle.wait()
        elif event.type == CommEventType.WAIT_SEND_PREV:
            handle = self.send_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
            handle.wait()
        elif event.type == CommEventType.WAIT_RECV_PREV:
            handle = self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)]
            assert handle is not None
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
            self.cdc_print(f"pre_event: {pre_event}")
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

        if task_type == "F" and (not forward_only or mb_id < num_microbatches):
            self.cdc_print(f"forward_step mb_id: {mb_id}, chunk_id: {chunk_id}")
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

            self.cdc_print(f"backward_step mb_id: {mb_id}, chunk_id: {chunk_id}")
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

            self.cdc_print(f"wgrad_step mb_id: {mb_id}, chunk_id: {chunk_id}")

            self.wgrad_store.compute_wgrad_block()

            if self._is_last_microbatch_for_model_chunk(compute_task, num_microbatches):
                # TODO: I guess enable_grad_sync would not enable DP comm here.
                # So, we need to do it manually.
                model_chunk = model[chunk_id]
                assert hasattr(model_chunk, "finish_grad_sync")
                model_chunk.finish_grad_sync()

        for post_event in compute_task.post_events:
            self.cdc_print(f"post_event: {post_event}")
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
                    self.cdc_print(f"deallocate_output_tensor: {mb_id}, {chunk_id}")
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
                    self.cdc_print(f"releasing input grad ref: {mb_id}, {chunk_id}")
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

        for compute_task in self.pp_execution_plan_cur_device:
            # self.cdc_print(f"compute_task: {compute_task}")
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

        self.clean_up()

        self.cdc_print(f"forward_data_store: {forward_data_store}")
        return forward_data_store

    def get_forward_backward_func(self):
        return self.forward_backward_func

    def get_cdc_recv_delay(self) -> int | float:
        return self.cdc_latency

    def need_hook_recv_prev(self) -> bool:
        return self.cdc_recv_prev

    def need_hook_recv_next(self) -> bool:
        return self.cdc_recv_next

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
        if (src == prev_rank and self.cdc_recv_prev) or (
            src == next_rank and self.cdc_recv_next
        ):
            return cdc_comm.irecv(tensor, src, group=group).wait()
        else:
            dist.recv(tensor, src, group=group)

    def irecv(
        self, tensor: torch.Tensor, src: int, group: dist.ProcessGroupNCCL | None = None
    ):
        prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
        next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
        if (src == prev_rank and self.cdc_recv_prev) or (
            src == next_rank and self.cdc_recv_next
        ):
            return cdc_comm.irecv(tensor, src, group=group)
        else:
            return dist.irecv(tensor, src, group=group)

    def cdc_print(self, msg: str, rank=None):
        if not self.cdc_verbose_print:
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
