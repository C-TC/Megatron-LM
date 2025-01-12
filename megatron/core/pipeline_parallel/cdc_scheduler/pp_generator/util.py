from collections import defaultdict
import math
from typing import List, Tuple
import numpy as np


def generate_comm_mat(num_DC, num_dev_per_DC, T_intra, T_inter) -> np.ndarray:
    assert isinstance(T_intra, int) and isinstance(T_inter, int)
    num_dev = num_DC * num_dev_per_DC
    ret = (np.ones((num_dev, num_dev), dtype=int) - np.eye(num_dev, dtype=int)) * T_intra
    for i in range(num_DC):
        for j in range(num_DC):
            if i != j:
                ret[
                    i * num_dev_per_DC : (i + 1) * num_dev_per_DC,
                    j * num_dev_per_DC : (j + 1) * num_dev_per_DC,
                ] = T_inter
    return ret

def device_to_dc(dev_id, num_dev_per_DC) -> int:
    # now we assume each DC has the same number of devices, and devices are allocated sequentially to DCs
    return dev_id // num_dev_per_DC

def scale_to_int(val_list: List[float], min_interval=8) -> Tuple[List[int], float]:
    min_gap = math.inf
    for i, val_x in enumerate(val_list):
        min_gap = min(min_gap, abs(val_x))
        for j in range(i + 1, len(val_list)):
            min_gap = min(min_gap, abs(abs(val_list[j]) - abs(val_x)))
    scale_factor = min_interval / min_gap
    return [int(round(x * scale_factor)) for x in val_list], scale_factor

class BandwidthDelayModel:
    def __init__(self, sys_config):
        self.sys_config = sys_config
        # (sender, receiver) -> [(start_time, end_time),...]
        self.link_occupancy = defaultdict(list)

    def get_bandwidth_time_with_delay(
        self, prev_dev, cur_dev, send_start_time, bandwidth_time
    ):
        # return the earliest start time for the current task
        # model the bw delay effect
        if bandwidth_time == 0:
            # infinite bandwidth, no bw delay
            return bandwidth_time
        assert (
            prev_dev != cur_dev
        ), "Same device scenario should be handled outside"

        # basic case: first task
        if len(self.link_occupancy[(prev_dev, cur_dev)]) == 0:
            self.link_occupancy[(prev_dev, cur_dev)].append(
                (send_start_time, send_start_time + bandwidth_time)
            )
            return bandwidth_time

        # general task: check bw delay and place the task
        # assumption: the time range in link_occupancy is sorted
        last_send_task = self.link_occupancy[(prev_dev, cur_dev)][-1]
        if send_start_time >= last_send_task[1]:
            # no bw delay
            self.link_occupancy[(prev_dev, cur_dev)].append(
                (send_start_time, send_start_time + bandwidth_time)
            )
            return bandwidth_time
        else:
            # find the gap for bw delay
            next_range_index = 0
            while next_range_index < len(self.link_occupancy[(prev_dev, cur_dev)]):
                next_range = self.link_occupancy[(prev_dev, cur_dev)][next_range_index]
                prev_range_end = 0 if next_range_index == 0 else self.link_occupancy[(prev_dev, cur_dev)][next_range_index - 1][1]
                if send_start_time >= prev_range_end and send_start_time + bandwidth_time <= next_range[0]:
                    break
                next_range_index += 1
            if next_range_index == len(self.link_occupancy[(prev_dev, cur_dev)]):
                # no available gap, wait for the last task to finish
                new_start_time = last_send_task[1]
                self.link_occupancy[(prev_dev, cur_dev)].append(
                    (new_start_time, new_start_time + bandwidth_time)
                )
                return new_start_time + bandwidth_time - send_start_time
            else:
                # find the gap, place the task
                new_start_time = send_start_time
                self.link_occupancy[(prev_dev, cur_dev)].insert(
                    next_range_index,
                    (new_start_time, new_start_time + bandwidth_time),
                )
                return bandwidth_time
        
    def clear(self):
        self.link_occupancy = defaultdict(list)
