from dataclasses import dataclass
from typing import List, Optional, Tuple
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import prepare_input_tensors_for_wgrad_compute
import torch
import torch.distributed as dist
from torch.distributed import Work
from torch import Tensor


@dataclass
class WGradUnit:
    is_CPL: bool  # True if CPL, False if RPL
    weight: Tensor
    input_tensor: Tensor
    input_tensor_AG: bool
    grad_output: Tensor
    grad_output_AG: bool

    def __post_init__(self):
        # assert not (self.input_tensor_AG and self.grad_output_AG), "Only 1 AG flag can be True"
        pass


def _gather_along_first_dim_async(tensor: Tensor) -> Tuple[Tensor, Optional[Work]]:
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return tensor, None
    dim_size = list(tensor.size())
    dim_size[0] *= world_size
    output = torch.empty(
        dim_size, dtype=tensor.dtype, device=torch.cuda.current_device()
    )
    handle = dist.all_gather_into_tensor(
        output,
        tensor.contiguous(),
        group=get_tensor_model_parallel_group(),
        async_op=True,
    )
    return output, handle


def prepare_input_output_grad(
    unit: WGradUnit,
) -> Tuple[List[Tensor], List[Optional[Work]]]:
    # TODO: do we need global buffers to reduce memory usage?
    if unit.grad_output_AG:
        grad_output, handle_out = _gather_along_first_dim_async(unit.grad_output)
    else:
        grad_output = unit.grad_output
        handle_out = None
    if unit.input_tensor_AG:
        input_tensor, handle_in = _gather_along_first_dim_async(unit.input_tensor)
    else:
        input_tensor = unit.input_tensor
        handle_in = None

    grad_output, input_tensor = prepare_input_tensors_for_wgrad_compute(
        grad_output, input_tensor
    )

    return [input_tensor, grad_output], [handle_in, handle_out]


def compute_wgrad(input_tensor: Tensor, grad_output: Tensor, weight: Tensor) -> None:
    import fused_weight_gradient_mlp_cuda

    if weight.main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            input_tensor, grad_output, weight.main_grad
        )
    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
            input_tensor, grad_output, weight.main_grad
        )
    else:
        raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")


class WGradStore:
    def __init__(self) -> None:
        self.cur_wgrad_block: List[WGradUnit] = []
        self.wgrad_blocks: List[List[WGradUnit]] = []

    def add_unit_to_block(self, unit: WGradUnit) -> None:
        self.cur_wgrad_block.append(unit)

    def finish_collection_wgrad_block(self) -> None:
        self.wgrad_blocks.append(self.cur_wgrad_block)
        self.cur_wgrad_block = []

    def compute_wgrad_block(self) -> None:
        # overlap computation with next unit's communication
        # get the first block of units
        assert len(self.wgrad_blocks) > 0, "No wgrad block available at this point."
        wgrad_unit_list = self.wgrad_blocks.pop(0)
        if len(wgrad_unit_list) == 0:
            return

        prev_inout, prev_handles = prepare_input_output_grad(wgrad_unit_list[0])
        prev_weight = wgrad_unit_list[0].weight
        for unit in wgrad_unit_list[1:]:
            cur_inout, cur_handles = prepare_input_output_grad(unit)
            cur_weight = unit.weight
            for handle in prev_handles:
                if handle is not None:
                    handle.wait()
            compute_wgrad(*prev_inout, prev_weight)
            prev_inout = cur_inout
            prev_handles = cur_handles
            prev_weight = cur_weight
        # last unit
        for handle in prev_handles:
            if handle is not None:
                handle.wait()
        compute_wgrad(*prev_inout, prev_weight)
        del prev_inout, cur_inout

    def is_empty(self) -> bool:
        return len(self.wgrad_blocks) == 0
