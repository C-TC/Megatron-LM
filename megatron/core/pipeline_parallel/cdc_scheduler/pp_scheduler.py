import contextlib
from typing import Dict, Iterator, List, Tuple, Union

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


class CDCPPScheduler:
    """

    Notice that chunk_id == virtual_pipeline_model_parallel_rank
    TODO: support checkpointing

    """

    def __init__(self, args) -> None:
        self.args = args
        self.use_static_schedule = False
        self.pp_schedule: Pipeline = None

        # static schedule
        if args.static_schedule is not None:
            self.use_static_schedule = True
            pp_size = args.pipeline_model_parallel_size
            num_microbatch = get_num_microbatches()
            self.pp_schedule = get_default_static_schedule(args.static_schedule, pp_size, num_microbatch)
        else:
            raise NotImplementedError()

        self.validate_args()

        self.pp_execution_planner = ExecutionPlanner(self.pp_schedule)
        self.pp_execution_planner.generate_execution_plan()
        self.pp_execution_plan: List[List[ComputeTask]] = (
            self.pp_execution_planner.execution_plan
        )

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_execution_plan_cur_device: List[ComputeTask] = self.pp_execution_plan[
            pp_rank
        ]

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
            not args.align_grad_reduce and args.grad_sync_func is None
        ), "align_grad_reduce is not supported, therefore grad_sync_func must be None"
        assert (
            not args.align_param_gather and args.param_sync_func is None
        ), "align_param_gather is not supported, therefore param_sync_func must be None"

        assert (
            not args.defer_embedding_wgrad_compute
        ), "defer_embedding_wgrad_compute is not supported"

        assert (
            args.virtual_pipeline_model_parallel_size
            == self.pp_schedule.sys_config.num_chunks
        ), "For compatibility, virtual_pipeline_model_parallel_size is equivalent to number of chunks"

    def get_layer_offset(self, dev_id, chunk_id):
        layers_list = self.get_num_layers_in_chunk(dev_id=None, chunk_id=None)
        execution_order = self.pp_schedule.get_pipeline_execution_order()
        idx_in_order = execution_order.index((dev_id, chunk_id))
        return sum(layers_list[:idx_in_order])        
    
    def get_num_layers_in_chunk(self, *, dev_id=None, chunk_id=None):
        # TODO: better model splitting
        # now we treat vocab and lm head as one layer each. and add unbalanced layers to first several chunks
        num_layer = self.args.num_layers
        execution_order = self.pp_schedule.get_pipeline_execution_order()
        total_chunks = len(execution_order)
        
        head_tail_layers = (num_layer + 2) // total_chunks
        rest_layers = num_layer - head_tail_layers * 2
        rest_chunks = total_chunks - 2
        remainder = rest_layers % rest_chunks
        layer_list = [rest_layers // rest_chunks] * rest_chunks
        for i in range(remainder):
            layer_list[i] += 1
        layer_list = [head_tail_layers] + layer_list + [head_tail_layers]
        
        assert (dev_id is not None or chunk_id is not None) or (dev_id is None and chunk_id is None)
        if dev_id is not None and chunk_id is not None:
            return layer_list[execution_order.index((dev_id, chunk_id))]
        else:
            return layer_list
        
    def _is_last_microbatch_for_model_chunk(
        self, compute_task: ComputeTask, number_of_microbatches
    ):
        # To enable grad sync in backward.
        # TODO: W block
        has_w_blocks = any(
            [task.task_desc.type == "W" for task in self.pp_execution_plan_cur_device]
        )
        assert not has_w_blocks, "W blocks are not supported yet"

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
                self.output_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )
            recv_buffer = get_or_set_tensor(
                self.input_tensor_grads,
                (event.mb_id, event.chunk_id),
                config,
                tensor_shape,
            )

        if event.type == CommEventType.POST_SEND_NEXT:
            self.send_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                dist.isend(send_buffer, next_rank, group=send_next_group)
            )
        elif event.type == CommEventType.POST_RECV_NEXT:
            self.recv_next_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                dist.irecv(recv_buffer, next_rank, group=recv_next_group)
            )
        elif event.type == CommEventType.POST_SEND_PREV:
            self.send_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                dist.isend(send_buffer, prev_rank, group=send_prev_group)
            )
        elif event.type == CommEventType.POST_RECV_PREV:
            self.recv_prev_reqs[(event.mb_id, event.chunk_id, event.task_type)] = (
                dist.irecv(recv_buffer, prev_rank, group=recv_prev_group)
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

        # TODO: set_virtual_pipeline_model_parallel_rank

        for pre_event in compute_task.pre_events:
            self.schedule_event(
                pre_event, config, tensor_shape, forward_only, num_microbatches
            )

        task_type = compute_task.task_desc.type
        chunk_id = compute_task.task_desc.chunk_id
        mb_id = compute_task.task_desc.mb_id
        parallel_state.set_virtual_pipeline_model_parallel_rank(chunk_id)
        is_first_stage = parallel_state.is_pipeline_first_stage()
        is_last_stage = parallel_state.is_pipeline_last_stage()

        if task_type == "F" and (not forward_only or mb_id < num_microbatches):
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
            if is_last_stage:
                # no need to cache output tensor at last stage
                self.output_tensors[(mb_id, chunk_id)] = None

        elif task_type == "B" and not forward_only:
            # Only training. In eval, we skip backward.

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

            # disable grad sync for other microbatches
            if self._is_last_microbatch_for_model_chunk(compute_task, num_microbatches):
                self.disable_grad_sync()

        elif task_type == "W" and not forward_only:
            # TODO: W block
            raise NotImplementedError()

        for post_event in compute_task.post_events:
            self.schedule_event(
                post_event, config, tensor_shape, forward_only, num_microbatches
            )

    def deallocate_tensor_in_dicts(self):
        # TODO: any better ways?

        # output tensors and input tensor grads
        for (mb_id, chunk_id, task_type), handle in (
            self.send_next_reqs.items() + self.send_prev_reqs.items()
        ):
            if task_type == "F":
                if (
                    self.output_tensors[(mb_id, chunk_id)] is not None
                    and handle is not None
                    and handle.is_completed()
                ):
                    deallocate_output_tensor(
                        self.output_tensors[(mb_id, chunk_id)],
                        self.args.deallocate_pipeline_outputs,
                    )
            elif task_type == "B":
                if (
                    self.input_tensor_grads[(mb_id, chunk_id)] is not None
                    and handle is not None
                    and handle.is_completed()
                ):
                    self.input_tensor_grads[(mb_id, chunk_id)] = None

    def setup_grad_sync(self):
        # grad sync
        # Disable async grad reductions
        self.no_sync_func = self.args.no_sync_func
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

        return forward_data_store

    def get_forward_backward_func(self):
        return self.forward_backward_func
