from dataclasses import dataclass
from functools import cmp_to_key
import math
from typing import List, Tuple
from .pipeline_config import SystemConfig


@dataclass
class ScheduleNode:
    mb_id: int
    device_id: int
    chunk_id: int
    type: int  # 0: Forward, 1: Backward, 2: Weight
    available_time: float | int
    time_cost: float | int
    mem_incr: float | int
    start_time: float | int = -1
    end_time: float | int = -1

    def __post_init__(self):
        assert self.type in [0, 1, 2]


class ScheduleDevice:
    def __init__(self, dev_id: int, sys_cfg: SystemConfig) -> None:
        self.dev_id = dev_id
        self.sys_cfg = sys_cfg
        self.T_F = sys_cfg.T_F[dev_id]
        self.T_B = sys_cfg.T_B[dev_id]
        self.T_W = sys_cfg.T_W[dev_id]
        self.M_F = sys_cfg.M_F[dev_id]
        self.M_B = sys_cfg.M_B[dev_id]
        self.M_W = sys_cfg.M_W[dev_id]

        self.mem_limit = sys_cfg.M_Limit[dev_id]

        # assume scheduled_nodes are sorted by end_time
        self.scheduled_nodes: List[ScheduleNode] = []

        self.schedulable_nodes: List[ScheduleNode] = []

        self._next_schedulable_time = 0 if dev_id == 0 else math.inf

        self._next_node = None

        self.cur_mem_usage = 0

    def add_schedulable_node(self, node_list: List[ScheduleNode]):
        self.schedulable_nodes.extend(node_list)
        self.update_next()

    def get_current_end_time(self):
        if len(self.scheduled_nodes) == 0:
            return 0
        return self.scheduled_nodes[-1].end_time

    def next_schedulable_time(self):
        return self._next_schedulable_time

    def next_node_to_schedule(self):
        return self._next_node

    def _next_to_schedule(self) -> Tuple[None | ScheduleNode, int | float]:
        """
        Return the next node to schedule and the time to schedule it
        """
        if len(self.schedulable_nodes) == 0:
            return None, math.inf

        # preference: B0 > B1 > F1 > F0 > W if mem allows
        # Otherwise, B0 > B1 > W > F1 > F0
        # but always prefer no bubble
        earliest_start_time = min(
            [node.available_time for node in self.schedulable_nodes]
        )
        cur_end_time = self.get_current_end_time()
        if earliest_start_time <= cur_end_time:
            # probably multiple nodes can be scheduled
            cur_schedulable_nodes = [
                node
                for node in self.schedulable_nodes
                if node.available_time <= cur_end_time
            ]
        else:
            # only earliest node can be scheduled to avoid bubble
            cur_schedulable_nodes = [
                node
                for node in self.schedulable_nodes
                if node.available_time <= earliest_start_time
            ]

        def node_priority(node: ScheduleNode, mem_allowed: bool):
            if mem_allowed:
                return [2, 1, 0][node.type] * 2 + (node.chunk_id if node.type == 0 else (1 - node.chunk_id))
            else:
                return [0, 2, 1][node.type] * 2 + (node.chunk_id if node.type == 0 else (1 - node.chunk_id))

        mem_allowed_forward = self.cur_mem_usage + self.M_F <= self.mem_limit

        def node_cmp(node_l: ScheduleNode, node_r: ScheduleNode):
            priority_l = node_priority(node_l, mem_allowed_forward)
            priority_r = node_priority(node_r, mem_allowed_forward)
            if priority_l == priority_r:
                # prefer lower num mb
                return node_l.mb_id - node_r.mb_id
            return priority_r - priority_l
            # if mem_allowed_forward:
            #      return node_l.mb_id - node_r.mb_id
            # early_node = min(node_l, node_r, key=lambda x: x.mb_id)
            # if early_node.type != 0:
            #     # mem sufficient
            #     return node_l.mb_id - node_r.mb_id
            # else:
            #     priority_l = node_priority(node_l, mem_allowed_forward)
            #     priority_r = node_priority(node_r, mem_allowed_forward)
            #     return priority_r - priority_l

        cur_schedulable_nodes.sort(key=cmp_to_key(node_cmp))

        # always prefer the first node in the list
        # unless it causes deadlock
        # case 1: on dev 0: no schedulable B/W node,
        #   try to schedule a F_0 (chunk 0 forward), which may
        #   block future F_1 (no mem) and therefore blocks B/W
        # case 2: on other devs: no schedulable B/W node,
        #   try to schedule a F node but no mem, move clock forward
        # case 1
        if (
            self.dev_id == 0
            and all(
                [
                    node.type == 0 and node.chunk_id == 0
                    for node in cur_schedulable_nodes
                ]
            )
            and self.cur_mem_usage + 2 * self.M_F > self.mem_limit
        ):
            # no other schedulable nodes at this time, to avoid deadlock
            # move clock forward to first schedulable F1/B/W node
            alive_nodes = [
                node
                for node in self.schedulable_nodes
                if node.type in [1, 2] or node.chunk_id == 1
            ]
            if len(alive_nodes) == 0:
                # failed to find a schedulable node even in the future
                # in this case, wait for other devices to proceed
                return None, math.inf
            alive_nodes.sort(key=lambda x: x.available_time)
            return alive_nodes[0], alive_nodes[0].available_time

        # case 2
        if (
            self.dev_id != 0
            and all([node.type == 0 for node in cur_schedulable_nodes])
            and self.cur_mem_usage + self.M_F > self.mem_limit
        ):
            alive_nodes = [
                node for node in self.schedulable_nodes if node.type in [1, 2]
            ]
            if len(alive_nodes) == 0:
                return None, math.inf
            alive_nodes.sort(key=lambda x: x.available_time)
            return alive_nodes[0], alive_nodes[0].available_time

        # general case
        for node in cur_schedulable_nodes:
            if node.type in [1, 2]:
                return node, node.available_time
            # F block
            if self.cur_mem_usage + self.M_F <= self.mem_limit:
                return node, node.available_time
        return None, math.inf

    def update_next(self):
        next_node, next_avail_time = self._next_to_schedule()
        self._next_node = next_node
        self._next_schedulable_time = next_avail_time
        if next_node is None:
            return
        if len(self.scheduled_nodes) > 0:
            self._next_schedulable_time = max(
                self._next_schedulable_time, self.scheduled_nodes[-1].end_time
            )

    def schedule_node(self):
        assert self._next_node is not None
        node = self._next_node
        self.schedulable_nodes.remove(node)
        self.scheduled_nodes.append(node)
        self.cur_mem_usage += (
            self.M_F if node.type == 0 else self.M_B if node.type == 1 else self.M_W
        )
        node.start_time = self._next_schedulable_time
        node.end_time = node.start_time + node.time_cost
        self.update_next()


class ZBVHeuristicSchedule:
    def __init__(self, system_cfg: SystemConfig):
        self.system_cfg = system_cfg
        assert self.system_cfg.num_chunks == 2

        self.devices: List[ScheduleDevice] = [
            ScheduleDevice(i, self.system_cfg)
            for i in range(self.system_cfg.num_devices)
        ]

        self._schedule = None

    @property
    def num_devices(self):
        return self.system_cfg.num_devices

    @property
    def num_mb(self):
        return self.system_cfg.num_microbatches

    @property
    def num_chunks(self):
        return self.system_cfg.num_chunks

    def get_T_F(self, device_id):
        return self.system_cfg.T_F[device_id]

    def get_T_B(self, device_id):
        return self.system_cfg.T_B[device_id]

    def get_T_W(self, device_id):
        return self.system_cfg.T_W[device_id]

    def get_T_comm(self, src_dev, dst_dev):
        return self.system_cfg.T_C[src_dev, dst_dev]

    def get_M_F(self, device_id):
        return self.system_cfg.M_F[device_id]

    def get_M_B(self, device_id):
        return self.system_cfg.M_B[device_id]

    def get_M_W(self, device_id):
        return self.system_cfg.M_W[device_id]

    def get_M_lim(self, device_id):
        return self.system_cfg.M_Limit[device_id]

    def next_device_to_schedule(self) -> int:
        def schedule_comparator(dev_l: ScheduleDevice, dev_r: ScheduleDevice):
            next_time_l = dev_l.next_schedulable_time()
            next_time_r = dev_r.next_schedulable_time()
            if next_time_l == next_time_r:
                return dev_l.dev_id - dev_r.dev_id
            return next_time_l - next_time_r

        # return dev id
        return min(self.devices, key=cmp_to_key(schedule_comparator)).dev_id

    def _get_next_block(self, cur_node: ScheduleNode):
        """Return next block of this microbatch (only F/B)"""
        cur_dev = cur_node.device_id
        cur_chunk = cur_node.chunk_id
        cur_type = cur_node.type
        if cur_type == 0 and cur_chunk == 0:
            if cur_dev != self.num_devices - 1:
                return cur_dev + 1, 0, 0
            else:
                return cur_dev, 1, 0
        if cur_type == 0 and cur_chunk == 1:
            if cur_dev != 0:
                return cur_dev - 1, 1, 0
            else:
                return cur_dev, 1, 1
        if cur_type == 1 and cur_chunk == 1:
            if cur_dev != self.num_devices - 1:
                return cur_dev + 1, 1, 1
            else:
                return cur_dev, 0, 1
        if cur_type == 1 and cur_chunk == 0:
            if cur_dev != 0:
                return cur_dev - 1, 0, 1
            else:
                return None, None, None

        assert cur_type == 2
        return None, None, None

    def schedule(self):
        next_schedule_dev = self.next_device_to_schedule()
        assert next_schedule_dev == 0

        self.devices[next_schedule_dev].add_schedulable_node(
            [
                ScheduleNode(
                    mb_id,
                    next_schedule_dev,
                    0,
                    0,
                    0,
                    self.get_T_F(next_schedule_dev),
                    self.get_M_F(next_schedule_dev),
                )
                for mb_id in range(self.num_mb)
            ]
        )

        while True:
            cur_schedule_dev = self.next_device_to_schedule()
            if self.devices[cur_schedule_dev].next_schedulable_time() == math.inf:
                break
            cur_node = self.devices[cur_schedule_dev].next_node_to_schedule()
            if cur_node is None:
                break
            self.devices[cur_schedule_dev].schedule_node()
            cur_node_end_time = self.devices[cur_schedule_dev].get_current_end_time()
            # if B, shcedule W
            if cur_node.type == 1:
                self.devices[cur_schedule_dev].add_schedulable_node(
                    [
                        ScheduleNode(
                            cur_node.mb_id,
                            cur_schedule_dev,
                            cur_node.chunk_id,
                            2,
                            cur_node_end_time,
                            self.get_T_W(cur_schedule_dev),
                            self.get_M_W(cur_schedule_dev),
                        )
                    ]
                )

            next_dev, next_chunk, next_type = self._get_next_block(cur_node)
            if next_dev is not None:
                comm_time = self.get_T_comm(cur_schedule_dev, next_dev)
                next_time_cost = (
                    self.get_T_F(next_dev) if next_type == 0 else self.get_T_B(next_dev)
                )
                next_mem_incr = (
                    self.get_M_F(next_dev) if next_type == 0 else self.get_M_B(next_dev)
                )
                self.devices[next_dev].add_schedulable_node(
                    [
                        ScheduleNode(
                            cur_node.mb_id,
                            next_dev,
                            next_chunk,
                            next_type,
                            cur_node_end_time + comm_time,
                            next_time_cost,
                            next_mem_incr,
                        )
                    ]
                )

        # check if all nodes are scheduled
        if not all([len(dev.schedulable_nodes) == 0 for dev in self.devices]) and all(
            [len(dev.scheduled_nodes) < 6 * self.num_mb for dev in self.devices]
        ):
            print("Warning: some nodes are not scheduled")

    def get_schedule(self) -> List[List[Tuple[int, int, int, str, int]]]:
        schedule = [[] for _ in range(self.num_devices)]
        tasktype_to_str = {0: "F", 1: "B", 2: "W"}
        for dev in self.devices:
            for node in dev.scheduled_nodes:
                schedule[dev.dev_id].append(
                    (
                        node.device_id,
                        node.mb_id,
                        node.chunk_id,
                        tasktype_to_str[node.type],
                        node.end_time,
                    )
                )
        return schedule
