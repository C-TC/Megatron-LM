from dataclasses import dataclass
from functools import cmp_to_key
import math
from typing import List, Tuple
from .pipeline_config import PipelineBlockDesc, SystemConfig
from .util import BandwidthDelayModel


@dataclass
class UDSubScheduleNode:
    mb_id: int
    device_id: int
    type: int  # 0: Forward, 1: Backward, 2: Weight
    available_time: int
    time_cost: int
    mem_incr: int
    start_time: int = -1
    end_time: int = -1
    subpart_start: int = 0  # inclusive
    subpart_end: int = 1  # exclusive
    num_subparts: int = 1

    def __post_init__(self):
        assert self.type in [0, 1, 2]


class UDScheduleDevice:
    def __init__(self, dev_id: int, sys_cfg: SystemConfig, num_subparts: int) -> None:
        self.dev_id = dev_id
        self.sys_cfg = sys_cfg
        self.num_devices = sys_cfg.num_devices
        self.num_subparts = num_subparts
        self.T_F = sys_cfg.T_F[0][dev_id]
        assert (
            self.T_F % num_subparts == 0
        ), f"{dev_id=}:{self.T_F=} must be divisible by {num_subparts=}"
        self.T_B = sys_cfg.T_B[0][dev_id]
        assert (
            self.T_B % num_subparts == 0
        ), f"{dev_id=}:{self.T_B=} must be divisible by {num_subparts=}"
        self.T_W = sys_cfg.T_W[0][dev_id]
        assert (
            self.T_W % num_subparts == 0
        ), f"{dev_id=}:{self.T_W=} must be divisible by {num_subparts=}"
        self.M_F = sys_cfg.M_F[0][dev_id]
        assert (
            self.M_F % num_subparts == 0
        ), f"{dev_id=}:{self.M_F=} must be divisible by {num_subparts=}"
        self.M_B = sys_cfg.M_B[0][dev_id]
        assert (
            self.M_B % num_subparts == 0
        ), f"{dev_id=}:{self.M_B=} must be divisible by {num_subparts=}"
        self.M_W = sys_cfg.M_W[0][dev_id]
        assert (
            self.M_W % num_subparts == 0
        ), f"{dev_id=}:{self.M_W=} must be divisible by {num_subparts=}"

        self.mem_limit = sys_cfg.M_Limit[dev_id]

        # assume scheduled_nodes are sorted by end_time
        self.scheduled_nodes: List[UDSubScheduleNode] = []

        self.schedulable_nodes: List[UDSubScheduleNode] = []

        self._next_schedulable_time = 0 if dev_id == 0 else math.inf

        self._next_node = None

        self.cur_mem_usage = 0
        self.last_scheduled_full_block_type = 0  # either F/B, by default, F block

        self.tear_down_phase = False

    def add_schedulable_node(self, node_list: List[UDSubScheduleNode]):
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

    def _next_to_schedule(self) -> Tuple[None | UDSubScheduleNode, int]:
        """
        Return the next node to schedule and the time to schedule it
        """
        if len(self.schedulable_nodes) == 0:
            return None, math.inf

        # preference: B > F > W if mem allows
        # Otherwise, B > W > F
        # but always prefer no bubble
        F_mem_allowed = (
            self.cur_mem_usage + self.M_F // self.num_subparts <= self.mem_limit
        )
        B_mem_allowed = (
            self.cur_mem_usage + self.M_B // self.num_subparts <= self.mem_limit
        )

        cur_schedulable_nodes = []
        cur_end_time = self.get_current_end_time()
        available_nodes = [
            node
            for node in self.schedulable_nodes
            if node.available_time <= cur_end_time
        ]
        available_nodes_schedulable = [
            node
            for node in available_nodes
            if node.mem_incr // node.num_subparts + self.cur_mem_usage <= self.mem_limit
        ]
        if len(available_nodes_schedulable) > 0:
            cur_schedulable_nodes = available_nodes_schedulable
        else:
            # go forward in time to find schedulable nodes
            self.schedulable_nodes.sort(key=lambda x: x.available_time)
            for schedulable_node in self.schedulable_nodes:
                if (
                    schedulable_node.mem_incr // schedulable_node.num_subparts
                    + self.cur_mem_usage
                    <= self.mem_limit
                ):
                    cur_schedulable_nodes = [
                        schedulable_node,
                    ]
                    break

        def node_priority(
            node: UDSubScheduleNode, F_mem_allowed: bool, B_mem_allowed: bool
        ):
            if not F_mem_allowed and not B_mem_allowed:
                # W is the only choice
                return [0, 1, 2][node.type]
            if self.tear_down_phase:
                return [0, 2, 1][node.type]
            
            if self.last_scheduled_full_block_type == 0:
                if B_mem_allowed:
                    return [1, 2, 0][node.type]
                else:
                    # case when B increase memory
                    # return [2, 0, 1][node.type]
                    return [1, 0, 2][node.type]
            if self.last_scheduled_full_block_type == 1:
                if F_mem_allowed:
                    return [2, 1, 0][node.type]
                else:
                    # choose the one reduce memory the fastest
                    return [0, -self.M_B / self.T_B, -self.M_W / self.T_W][node.type]

        def node_cmp(node_l: UDSubScheduleNode, node_r: UDSubScheduleNode):
            priority_l = node_priority(
                node_l, F_mem_allowed=F_mem_allowed, B_mem_allowed=B_mem_allowed
            )
            priority_r = node_priority(
                node_r, F_mem_allowed=F_mem_allowed, B_mem_allowed=B_mem_allowed
            )
            if priority_l == priority_r:
                # prefer lower num mb
                return node_l.mb_id - node_r.mb_id
            return priority_r - priority_l

        cur_schedulable_nodes.sort(key=cmp_to_key(node_cmp))

        # always prefer the first node in the list

        # general case
        for node in cur_schedulable_nodes:
            if node.type == 2:
                return node, node.available_time

            if (node.type == 0 and F_mem_allowed) or (node.type == 1 and B_mem_allowed):
                return node, node.available_time
        return None, math.inf

    def update_next(self):
        num_mb = self.sys_cfg.num_microbatches
        last_scheduled_node = (
            self.scheduled_nodes[-1] if len(self.scheduled_nodes) > 0 else None
        )
        if last_scheduled_node is not None:
            if (
                last_scheduled_node.mb_id == num_mb - 1
                and last_scheduled_node.type == 0
                and last_scheduled_node.subpart_end == self.num_subparts
            ):
                self.tear_down_phase = True

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
        if node.subpart_end != self.num_subparts:
            # Add remaining subparts
            next_subpart_node = UDSubScheduleNode(
                mb_id=node.mb_id,
                device_id=node.device_id,
                type=node.type,
                available_time=node.available_time,
                time_cost=node.time_cost,
                mem_incr=node.mem_incr,
                subpart_start=node.subpart_end,
                subpart_end=node.subpart_end + 1,
                num_subparts=node.num_subparts,
            )
            self.schedulable_nodes.append(next_subpart_node)
        else:
            # update last scheduled block type
            if node.type in [0, 1]:
                self.last_scheduled_full_block_type = node.type
        self.scheduled_nodes.append(node)
        self.cur_mem_usage += (
            self.M_F if node.type == 0 else self.M_B if node.type == 1 else self.M_W
        ) // self.num_subparts
        node.start_time = max(self._next_schedulable_time, node.available_time)
        node.end_time = node.start_time + node.time_cost // self.num_subparts
        self.update_next()


class DynZBUDSubScheduler:
    def __init__(self, system_cfg: SystemConfig, num_subparts: int = 1):
        self.system_cfg = system_cfg
        assert self.system_cfg.num_chunks == 1

        self.devices: List[UDScheduleDevice] = [
            UDScheduleDevice(i, self.system_cfg, num_subparts=num_subparts)
            for i in range(self.system_cfg.num_devices)
        ]

        self._schedule = None
        self.num_subparts = num_subparts

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
        return self.system_cfg.T_F[0][device_id]

    def get_T_B(self, device_id):
        return self.system_cfg.T_B[0][device_id]

    def get_T_W(self, device_id):
        return self.system_cfg.T_W[0][device_id]

    def get_T_lat(self, src_dev, dst_dev):
        return self.system_cfg.T_alpha[src_dev, dst_dev]

    def get_T_bw(self, src_dev, dst_dev):
        return self.system_cfg.T_beta[src_dev, dst_dev]

    def get_M_F(self, device_id):
        return self.system_cfg.M_F[0][device_id]

    def get_M_B(self, device_id):
        return self.system_cfg.M_B[0][device_id]

    def get_M_W(self, device_id):
        return self.system_cfg.M_W[0][device_id]

    def get_M_lim(self, device_id):
        return self.system_cfg.M_Limit[device_id]

    def next_device_to_schedule(self) -> int:
        def schedule_comparator(dev_l: UDScheduleDevice, dev_r: UDScheduleDevice):
            next_time_l = dev_l.next_schedulable_time()
            next_time_r = dev_r.next_schedulable_time()
            if next_time_l == next_time_r:
                return dev_l.dev_id - dev_r.dev_id
            return next_time_l - next_time_r

        # return dev id
        return min(self.devices, key=cmp_to_key(schedule_comparator)).dev_id

    def _get_next_block(self, cur_node: UDSubScheduleNode):
        """Return next block of this microbatch (only F/B)"""
        cur_dev = cur_node.device_id
        cur_type = cur_node.type
        if cur_type == 0:
            if cur_dev != self.num_devices - 1:
                return cur_dev + 1, 0
            else:
                return cur_dev, 1
        if cur_type == 1:
            if cur_dev != 0:
                return cur_dev - 1, 1
            else:
                return None, None

        return None, None

    def schedule(self):
        bw_delay_model = BandwidthDelayModel(self.system_cfg)

        next_schedule_dev = self.next_device_to_schedule()
        assert next_schedule_dev == 0

        self.devices[next_schedule_dev].add_schedulable_node(
            [
                UDSubScheduleNode(
                    mb_id,
                    next_schedule_dev,
                    0,
                    0,
                    self.get_T_F(next_schedule_dev),
                    self.get_M_F(next_schedule_dev),
                    subpart_start=0,
                    subpart_end=1,
                    num_subparts=self.num_subparts,
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
            if cur_node.type == 1 and cur_node.subpart_end == self.num_subparts:
                self.devices[cur_schedule_dev].add_schedulable_node(
                    [
                        UDSubScheduleNode(
                            cur_node.mb_id,
                            cur_schedule_dev,
                            2,
                            0,  # W can be scheduled immediately
                            self.get_T_W(cur_schedule_dev),
                            self.get_M_W(cur_schedule_dev),
                            subpart_start=0,
                            subpart_end=1,
                            num_subparts=self.num_subparts,
                        )
                    ]
                )

            if cur_node.subpart_end != self.num_subparts:
                continue

            # last subpart of the node has scheduled
            next_dev, next_type = self._get_next_block(cur_node)
            if next_dev is not None:
                lat_time = self.get_T_lat(cur_schedule_dev, next_dev)
                bw_time = self.get_T_bw(cur_schedule_dev, next_dev)
                bw_time_with_delay = bw_delay_model.get_bandwidth_time_with_delay(
                    cur_schedule_dev, next_dev, cur_node_end_time, bw_time
                )
                next_time_cost = (
                    self.get_T_F(next_dev) if next_type == 0 else self.get_T_B(next_dev)
                )
                next_mem_incr = (
                    self.get_M_F(next_dev) if next_type == 0 else self.get_M_B(next_dev)
                )
                self.devices[next_dev].add_schedulable_node(
                    [
                        UDSubScheduleNode(
                            cur_node.mb_id,
                            next_dev,
                            next_type,
                            cur_node_end_time + bw_time_with_delay + lat_time
                            if next_dev != cur_schedule_dev
                            else 0,
                            next_time_cost,
                            next_mem_incr,
                            subpart_start=0,
                            subpart_end=1,
                            num_subparts=self.num_subparts,
                        )
                    ]
                )

        # check if all nodes are scheduled
        if not all([len(dev.schedulable_nodes) == 0 for dev in self.devices]) or any(
            [
                len(dev.scheduled_nodes) < 3 * self.num_subparts * self.num_mb
                for dev in self.devices
            ]
        ):
            print("Warning: some nodes are not scheduled")

        self.merge_subparts()

    def merge_subparts(self):
        # first: reorder patterns like WFWFWF
        # by moving W to front, until end of a full block or another W
        for dev in self.devices:
            w_idx = 0
            while True:
                # advance to the first W
                while True:
                    if w_idx >= len(dev.scheduled_nodes):
                        break
                    if dev.scheduled_nodes[w_idx].type == 2:
                        break
                    w_idx += 1

                if w_idx >= len(dev.scheduled_nodes):
                    break
                # find the target to exchange
                exchange_idx = w_idx
                while True:
                    assert exchange_idx >= 1
                    if dev.scheduled_nodes[exchange_idx - 1].type == 2:
                        break
                    if (
                        dev.scheduled_nodes[exchange_idx - 1].subpart_end
                        == self.num_subparts
                    ):
                        break
                    exchange_idx -= 1

                if exchange_idx == w_idx:
                    # nothing to exchange
                    w_idx += 1
                    continue
                else:
                    # extract and insert
                    w_node = dev.scheduled_nodes.pop(w_idx)
                    dev.scheduled_nodes.insert(exchange_idx, w_node)
                    continue

        # second: merge subparts
        for dev in self.devices:
            node_idx = 0
            while node_idx < len(dev.scheduled_nodes):
                node = dev.scheduled_nodes[node_idx]
                if node.subpart_end == self.num_subparts:
                    node_idx += 1
                    continue
                if node_idx + 1 >= len(dev.scheduled_nodes):
                    break
                next_node = dev.scheduled_nodes[node_idx + 1]
                if node.mb_id != next_node.mb_id or node.type != next_node.type:
                    node_idx += 1
                    continue

                # merge subparts
                node.subpart_end = next_node.subpart_end
                node.end_time = next_node.end_time
                # remove next_node
                del dev.scheduled_nodes[node_idx + 1]

    def get_schedule(self) -> List[List[PipelineBlockDesc]]:
        schedule = [[] for _ in range(self.num_devices)]
        tasktype_to_str = {0: "F", 1: "B", 2: "W"}
        for dev in self.devices:
            for node in dev.scheduled_nodes:
                schedule[dev.dev_id].append(
                    PipelineBlockDesc(
                        device_id=node.device_id,
                        mb_id=node.mb_id,
                        task_type=tasktype_to_str[node.type],
                        end_time=node.end_time,
                        subpart_start=node.subpart_start,
                        subpart_end=node.subpart_end,
                        num_subparts=node.num_subparts,
                    )
                )
        return schedule
