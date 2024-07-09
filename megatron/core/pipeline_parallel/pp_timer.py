

from dataclasses import dataclass
import os
from typing import List
import torch

class PPTimer:
    def __init__(self, mb: int, chunk: int = 0, is_forward: bool = True) -> None:
        self.mb = mb
        self.chunk = chunk
        self.is_forward = is_forward
        self.start_timer = torch.cuda.Event(enable_timing=True)
        self.end_timer = torch.cuda.Event(enable_timing=True)
        self.end_recorded = False

    def record_start(self):
        self.start_timer.record()
    
    def record_end(self):
        assert not self.end_recorded
        self.end_timer.record()
        self.end_recorded = True

class PPTimeRecordType:
    FORWARD = 0
    BACKWARD = 1
    IDLE = 2

@dataclass
class PPTimeRecord:
    time: float
    mb: int
    chunk: int
    record_type: PPTimeRecordType

class PPTimerCollection:
    timer_list_cur_iter: List[PPTimer] = []
    timer_collections: List[List[PPTimeRecord]] = []
    iteration: int = 0

    @staticmethod
    def record_timer_start(mb: int, chunk: int = 0, is_forward: bool = True):
        timer = PPTimer(mb, chunk, is_forward)
        timer.record_start()
        PPTimerCollection.timer_list_cur_iter.append(timer)

    @staticmethod
    def record_timer_end():
        PPTimerCollection.timer_list_cur_iter[-1].record_end()
        
    @staticmethod
    def collect_timers():
        
        record_list: List[PPTimeRecord] = []
        for i, timer in enumerate(PPTimerCollection.timer_list_cur_iter):
            if i > 0:
                # idle time
                idle_time = PPTimerCollection.timer_list_cur_iter[i - 1].end_timer.elapsed_time(timer.start_timer)
                record_list.append(PPTimeRecord(idle_time, -1, -1, PPTimeRecordType.IDLE))

            elapsed_time = timer.start_timer.elapsed_time(timer.end_timer)
            record_type = PPTimeRecordType.FORWARD if timer.is_forward else PPTimeRecordType.BACKWARD
            record_list.append(PPTimeRecord(elapsed_time, timer.mb, timer.chunk, record_type))
        
        PPTimerCollection.timer_collections.append(record_list)
        
    @staticmethod
    def step():
        PPTimerCollection.collect_timers()
        PPTimerCollection.timer_list_cur_iter = []
        PPTimerCollection.iteration += 1

    @staticmethod
    def get_device_info():
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        return rank, local_rank

    @staticmethod
    def delete_previous_logs():
        rank, local_rank = PPTimerCollection.get_device_info()
        path = os.environ.get("PP_LOG_PATH", "")
        if path == "":
            raise ValueError("PP_LOG_PATH is not set")
        file_path = f"{path}/PP_{rank}_{local_rank}.txt"
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def log_module_timers():
        PPTimerCollection.delete_previous_logs()
        rank, local_rank = PPTimerCollection.get_device_info()
        path = os.environ.get("PP_LOG_PATH", "")
        if path == "":
            raise ValueError("PP_LOG_PATH is not set")
        file_path = f"{path}/PP_{rank}_{local_rank}.txt"
        with open(os.path.abspath(file_path), "a") as f:
            for i, time_records in enumerate(PPTimerCollection.timer_collections):
                f.write(f"iter {i} ")
                for time_record in time_records:
                    type_str_dict = {
                        PPTimeRecordType.FORWARD: "F",
                        PPTimeRecordType.BACKWARD: "B",
                        PPTimeRecordType.IDLE: "I"
                    }
                    if time_record.record_type == PPTimeRecordType.IDLE:
                        f.write(f" {type_str_dict[time_record.record_type]} {time_record.time} ")
                    else:
                        f.write(f"{type_str_dict[time_record.record_type]}_{time_record.mb}_{time_record.chunk} {time_record.time}")
                f.write("\n")
