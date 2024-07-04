from dataclasses import dataclass
from typing import Tuple
import torch
import os

class AutogradEventTimer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, timer: "WrapEventTimer"):
        timer.record_forward()
        ctx.timer = timer
        # print(f'forward: {timer.module_name}')
        return input

    @staticmethod
    def backward(ctx, grad):
        timer: "WrapEventTimer" = ctx.timer
        # print(f'backward: {timer.module_name}')
        timer.record_backward()
        return grad, None

class WrapEventTimer:

    def __init__(self, module_name: str, is_begin_timer: bool = False, enable_nvtx: bool = False):
        super(WrapEventTimer, self).__init__()
        self.module_name = module_name
        self.timer_forward = torch.cuda.Event(enable_timing=True)
        self.timer_backward = torch.cuda.Event(enable_timing=True)
        self.enable_nvtx = enable_nvtx
        self.is_begin_timer = is_begin_timer

    def record_forward(self):
        if self.enable_nvtx:
            if self.is_begin_timer:
                torch.cuda.nvtx.range_push(self.module_name + "_forward")
            else:
                torch.cuda.nvtx.range_pop()
        self.timer_forward.record()

    def record_backward(self):
        if self.enable_nvtx:
            if self.is_begin_timer:
                torch.cuda.nvtx.range_pop()
            else:
                torch.cuda.nvtx.range_push(self.module_name + "_backward")
        self.timer_backward.record()

class ModuleTimerPair:

    def __init__(self, module_name: str) -> None:
        enable_nvtx = os.environ.get("TP_ENABLE_NVTX", "0") == "1"
        self.begin_timer = WrapEventTimer(module_name, is_begin_timer=True, enable_nvtx=enable_nvtx)
        self.end_timer = WrapEventTimer(module_name, is_begin_timer=False, enable_nvtx=enable_nvtx)
        self.module_name = module_name

        GlobalTimerCollection.add_module_timer(self)

    def get_forward_backward_time(self) -> Tuple[float, float]:
        try:
            forward_time = self.begin_timer.timer_forward.elapsed_time(
                self.end_timer.timer_forward
            )
        except RuntimeError:
            forward_time = 0.0
            print(f'error: module_name: {self.module_name}, get_forward_time')
        
        try:
            backward_time = self.end_timer.timer_backward.elapsed_time(
                self.begin_timer.timer_backward
            )
        except RuntimeError:
            backward_time = 0.0
            print(f'error: module_name: {self.module_name}, get_backward_time')
        return forward_time, backward_time

    def begin_timers(self, input):
        return AutogradEventTimer.apply(input, self.begin_timer)
    
    def end_timers(self, input):
        return AutogradEventTimer.apply(input, self.end_timer)

class GlobalTimerCollection:
    module_timer_list: list["ModuleTimerPair"] = []
    iteration: int = -1

    @staticmethod
    def add_module_timer(module_timer_pair: "ModuleTimerPair"):
        GlobalTimerCollection.module_timer_list.append(module_timer_pair)

    @staticmethod
    def get_device_info():
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        return rank, local_rank

    @staticmethod
    def set_iteration(iteration: int):
        GlobalTimerCollection.iteration = iteration
        if iteration == 0:
            GlobalTimerCollection.delete_previous_logs()
        
    @staticmethod
    def delete_previous_logs():
        rank, local_rank = GlobalTimerCollection.get_device_info()
        path = os.environ.get("TP_LOG_PATH", "")
        if path == "":
            raise ValueError("TP_LOG_PATH is not set")
        file_path = f"{path}/TP_{rank}_{local_rank}.txt"
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def log_module_timers():
        rank, local_rank = GlobalTimerCollection.get_device_info()
        path = os.environ.get("TP_LOG_PATH", "")
        if path == "":
            raise ValueError("TP_LOG_PATH is not set")
        file_path = f"{path}/TP_{rank}_{local_rank}.txt"
        with open(os.path.abspath(file_path), "a") as f:
            for module_timer in GlobalTimerCollection.module_timer_list:
                forward_time, backward_time = module_timer.get_forward_backward_time(
                )
                f.write(
                    f"{GlobalTimerCollection.iteration} {module_timer.module_name} {forward_time} {backward_time}\n"
                )

