from dataclasses import dataclass
from typing import Tuple
import torch
import os

class WrapEventTimer(torch.nn.module):

    def __init__(self, name: str):
        super(WrapEventTimer, self).__init__()
        self.name = name
        self.timer_forward = torch.cuda.Event(enable_timing=True)
        self.timer_backward = torch.cuda.Event(enable_timing=True)

    def forward(self, input):
        self.timer_forward.record()
        return input

    def backward(self, grad):
        self.timer_backward.record()
        return grad


class ModuleTimerPair:

    def __init__(self, module_name: str) -> None:
        self.begin_timers = WrapEventTimer()
        self.end_timers = WrapEventTimer()
        self.module_name = module_name

        GlobalTimerCollection.add_module_timer(self)

    def get_forward_backward_time(self) -> Tuple[float, float]:
        return self.begin_timers.timer_forward.elapsed_time(
            self.end_timers.timer_forward
        ), self.end_timers.timer_backward.elapsed_time(
            self.begin_timers.timer_backward)


class GlobalTimerCollection:
    module_timer_list: list["ModuleTimerPair"]
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
        iteration = iteration

    @staticmethod
    def log_module_timers():
        rank, local_rank = GlobalTimerCollection.get_device_info()
        path = os.environ.get("TP_LOG_PATH", "")
        file_path = f"{path}/TP_{rank}_{local_rank}.txt"
        with open(os.path(file_path), "w+") as f:
            for module_timer in GlobalTimerCollection.module_timer_list:
                forward_time, backward_time = module_timer.get_forward_backward_time(
                )
                f.write(
                    f"{GlobalTimerCollection.iteration} {module_timer.module_name} {forward_time} {backward_time}\n"
                )

