from dataclasses import dataclass
from functools import cmp_to_key
import math
from typing import List, Optional, Tuple
from .pipeline_config import PipelineBlockDesc, SystemConfig
from .heuristic_ud_subschedule import (
    DynZBUDSubScheduler,
    UDScheduleDevice,
)
from .util import BandwidthDelayModel


@dataclass
class WaveSubScheduleNode:
    mb_id: int
    device_id: int
    type: int  # 0: Forward, 1: Backward, 2: Weight
    available_time: int
    time_cost: int
    mem_incr: int
    chunk_id: int
    start_time: int = -1
    end_time: int = -1
    subpart_start: int = 0  # inclusive
    subpart_end: int = 1  # exclusive
    num_subparts: int = 1

    def __post_init__(self):
        assert self.type in [0, 1, 2]


class WaveBootStrapSubSchedule:
    def __init__(self, system_cfg: SystemConfig, num_subparts: int = 1):
        self.system_cfg = system_cfg
        assert self.system_cfg.num_chunks == 2
        self.num_subparts = num_subparts
        self.device_tasks: List[List[WaveSubScheduleNode]] = [
            [] for _ in range(self.system_cfg.num_devices)
        ]
        # the start time of first B block on each device
        self.first_b_start = []
        self.bw_delay_model = BandwidthDelayModel(self.system_cfg)

    def print_schedule(self):
        for i in range(self.system_cfg.num_devices):
            print(f"Device {i}:")
            for node in self.device_tasks[i]:
                task_type = ["F", "B", "W"][node.type]
                subpart = (
                    f"({node.subpart_start}-{node.subpart_end}/{node.num_subparts})"
                    if node.num_subparts > 1
                    else ""
                )
                print(
                    f"{task_type}{node.mb_id}c{node.chunk_id}{subpart}, [{node.start_time}, {node.end_time}]"
                )

    def bootstrap(self):
        M_F = self.system_cfg.M_F
        M_Limit = self.system_cfg.M_Limit
        T_F = self.system_cfg.T_F
        T_B = self.system_cfg.T_B
        T_C = self.system_cfg.T_C
        T_beta = self.system_cfg.T_beta
        assert all(
            [2 * M_F[i] <= M_Limit[i] for i in range(self.system_cfg.num_devices)]
        ), "Not enough memory for 1 V"

        start_of_backward = T_F[0]
        for dev in range(1, self.system_cfg.num_devices):
            lat_time = T_C[dev - 1, dev]
            bw_time = T_beta[dev - 1, dev]
            bw_time_with_delay = self.bw_delay_model.get_bandwidth_time_with_delay(
                dev - 1, dev, start_of_backward, bw_time
            )
            start_of_backward = (
                start_of_backward + lat_time + bw_time_with_delay + T_F[dev]
            )

        start_of_backward += T_F[-1]
        for dev in range(self.system_cfg.num_devices - 2, -1, -1):
            lat_time = T_C[dev + 1, dev]
            bw_time = T_beta[dev + 1, dev]
            bw_time_with_delay = self.bw_delay_model.get_bandwidth_time_with_delay(
                dev + 1, dev, start_of_backward, bw_time
            )
            start_of_backward = (
                start_of_backward + lat_time + bw_time_with_delay + T_F[dev]
            )

        self.first_b_start.append(start_of_backward)
        start_of_backward += T_B[0]
        for i in range(1, self.system_cfg.num_devices):
            lat_time = T_C[i - 1, i]
            bw_time = T_beta[i - 1, i]
            bw_time_with_delay = self.bw_delay_model.get_bandwidth_time_with_delay(
                i - 1, i, start_of_backward, bw_time
            )
            self.first_b_start.append(start_of_backward + lat_time + bw_time_with_delay)
            start_of_backward = self.first_b_start[-1] + T_B[i]

        self.bw_delay_model.clear()
        for mb_id in range(self.system_cfg.num_microbatches):
            self.schedule_mb(mb_id)

        # schedule first b blocks
        for i in range(self.system_cfg.num_devices):
            for subpart_start in range(self.num_subparts):
                self.device_tasks[i].append(
                    WaveSubScheduleNode(
                        mb_id=0,
                        device_id=i,
                        type=1,
                        available_time=self.first_b_start[i],
                        time_cost=T_B[i],
                        mem_incr=self.system_cfg.M_B[i],
                        start_time=self.first_b_start[i] + subpart_start * T_B[i] // self.num_subparts,
                        end_time=self.first_b_start[i] + (subpart_start + 1) * T_B[i] // self.num_subparts,
                        subpart_start=subpart_start,
                        subpart_end=subpart_start + 1,
                        num_subparts=self.num_subparts,
                        chunk_id=1,
                    )
            )

    def get_avail_ranges(self, device_id, avail_time) -> List[Tuple[int, int]]:
        T_F = self.system_cfg.T_F[device_id]
        avail_ranges = []
        cur_start_time = 0
        if len(self.device_tasks[device_id]) == 0:
            if avail_time >= self.first_b_start[device_id]:
                return []
            return [(max(avail_time, cur_start_time), self.first_b_start[device_id])]

        for node in self.device_tasks[device_id]:
            valid_start_time = max(cur_start_time, avail_time)
            if node.start_time >= valid_start_time + T_F // self.num_subparts:
                avail_ranges.append((valid_start_time, node.start_time))
            cur_start_time = node.end_time

        if cur_start_time < self.first_b_start[device_id]:
            avail_ranges.append(
                (max(cur_start_time, avail_time), self.first_b_start[device_id])
            )

        return avail_ranges

    def insert(
        self, avail_time, mb_id, device_id, chunk_id
    ) -> Optional[WaveSubScheduleNode]:
        # return the last inserted subpart node if inserted
        M_Limit = self.system_cfg.M_Limit[device_id]
        M_F = self.system_cfg.M_F[device_id]
        T_F = self.system_cfg.T_F[device_id]
        last_inserted_node = None

        for subpart_start in range(self.num_subparts):
            if (
                1 + len(self.device_tasks[device_id])
            ) * M_F // self.num_subparts > M_Limit:
                # no available memory
                return None
            avail_time_ranges = self.get_avail_ranges(device_id, avail_time)
            if len(avail_time_ranges) == 0:
                # no available time slot
                return None
            for start_time, end_time in avail_time_ranges:
                if end_time - start_time >= T_F // self.num_subparts:
                    node_to_insert = WaveSubScheduleNode(
                        mb_id=mb_id,
                        device_id=device_id,
                        type=0,
                        available_time=avail_time,
                        time_cost=T_F // self.num_subparts,
                        mem_incr=M_F,
                        start_time=start_time,
                        end_time=start_time + T_F // self.num_subparts,
                        subpart_start=subpart_start,
                        subpart_end=subpart_start + 1,
                        num_subparts=self.num_subparts,
                        chunk_id=chunk_id,
                    )

                    self.device_tasks[device_id].append(node_to_insert)
                    last_inserted_node = node_to_insert
                    # sort by start time
                    self.device_tasks[device_id].sort(key=lambda x: x.start_time)
                    break

        return last_inserted_node

    def schedule_mb(self, mb_id):
        # forward pass chunk 0
        prev_end_time = 0
        last_inserted_node = self.insert(
            avail_time=prev_end_time, mb_id=mb_id, device_id=0, chunk_id=0
        )

        for i in range(1, self.system_cfg.num_devices):
            # if last inserted node is not full, break
            if (
                last_inserted_node is None
                or last_inserted_node.subpart_end != self.num_subparts
            ):
                return False
            lat_time = self.system_cfg.T_C[i - 1, i]
            bw_time = self.system_cfg.T_beta[i - 1, i]
            bw_time_with_delay = self.bw_delay_model.get_bandwidth_time_with_delay(
                i - 1, i, last_inserted_node.end_time, bw_time
            )
            last_inserted_node = self.insert(
                avail_time=last_inserted_node.end_time + lat_time + bw_time_with_delay,
                mb_id=mb_id,
                device_id=i,
                chunk_id=0,
            )

        # forward pass chunk 1
        if (
            last_inserted_node is None
            or last_inserted_node.subpart_end != self.num_subparts
        ):
            return False
        prev_end_time = last_inserted_node.end_time
        last_inserted_node = self.insert(
            avail_time=prev_end_time,
            mb_id=mb_id,
            device_id=self.system_cfg.num_devices - 1,
            chunk_id=1,
        )
        for i in range(self.system_cfg.num_devices - 2, -1, -1):
            if (
                last_inserted_node is None
                or last_inserted_node.subpart_end != self.num_subparts
            ):
                return False
            lat_time = self.system_cfg.T_C[i + 1, i]
            bw_time = self.system_cfg.T_beta[i + 1, i]
            bw_time_with_delay = self.bw_delay_model.get_bandwidth_time_with_delay(
                i + 1, i, last_inserted_node.end_time, bw_time
            )
            last_inserted_node = self.insert(
                avail_time=last_inserted_node.end_time + lat_time + bw_time_with_delay,
                mb_id=mb_id,
                device_id=i,
                chunk_id=1,
            )
        return True

    def get_bootstrap_schedule(self):
        return self.device_tasks


class WaveScheduleDevice(UDScheduleDevice):
    def __init__(self, dev_id: int, sys_cfg: SystemConfig, num_subparts: int) -> None:
        super().__init__(dev_id, sys_cfg, num_subparts)
        # assume scheduled_nodes are sorted by end_time
        self.scheduled_nodes: List[WaveSubScheduleNode] = []

        self.schedulable_nodes: List[WaveSubScheduleNode] = []

        self.last_scheduled_full_block_chunk_id = [
            0,
            0,
            0,
        ]  # idx->type  0: F, 1: B, 2: W
        self.last_b_chunk1_block_mb_id = -1

    def add_schedulable_node(self, node_list: List[WaveSubScheduleNode]):
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

    def _next_to_schedule(self) -> Tuple[None | WaveSubScheduleNode, int]:
        """
        Return the next node to schedule and the time to schedule it
        """
        if len(self.schedulable_nodes) == 0:
            return None, math.inf

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
            node: WaveSubScheduleNode, F_mem_allowed: bool, B_mem_allowed: bool
        ):
            if self.tear_down_phase:
                return [0, 2, 1][node.type] * 2 + node.chunk_id
            
            
            extra_priority_for_current_chunk = (
                1
                if (node.type == 0 and node.chunk_id == 1)
                or (node.type == 1 and node.chunk_id == 0)
                or (node.type == 2 and node.chunk_id == 1)
                else 0
            )
            # if not F_mem_allowed and not B_mem_allowed:
            #     # W is the only choice
            #     return [0, 1, 2][node.type] * 2 + node.chunk_id
            # if F_mem_allowed and not B_mem_allowed:
            #     # recompute
            #     return [1, -math.inf, 2][node.type] * 2 + extra_priority_for_current_chunk
            # if B_mem_allowed and not F_mem_allowed:
            #     return [-math.inf, 2, 1][node.type] * 2 + extra_priority_for_current_chunk
            # # both F and B are allowed
            # if node.type == 0 and node.chunk_id == 0 \
            #     and self.last_b_chunk1_block_mb_id + self.num_devices < node.mb_id:
            #         return -math.inf
            # return [2, 1, 0][node.type] * 2 + extra_priority_for_current_chunk
            
            
            
            if self.last_scheduled_full_block_type == 0:
                if B_mem_allowed:
                    return [1, 2, 0][node.type] * 2 + extra_priority_for_current_chunk
                else:
                    # case when B increase memory
                    # return [2, 0, 1][node.type]
                    return [1, -math.inf, 2][node.type] * 2 + extra_priority_for_current_chunk
            if self.last_scheduled_full_block_type == 1:
                if F_mem_allowed:
                    return [2, 1, 0][node.type] * 2 + extra_priority_for_current_chunk
                else:
                    # choose the one reduce memory the fastest
                    return [0, -self.M_B / self.T_B, -self.M_W / self.T_W][
                        node.type
                    ] * 2 / max(
                        abs(self.M_B / self.T_B), abs(self.M_W / self.T_W)
                    ) + extra_priority_for_current_chunk

        def node_cmp(node_l: WaveSubScheduleNode, node_r: WaveSubScheduleNode):
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
                and last_scheduled_node.chunk_id == 1
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

    def _schedule_node_core(self, node: WaveSubScheduleNode):
        self.schedulable_nodes.remove(node)
        if node.subpart_end != self.num_subparts:
            # Add remaining subparts
            next_subpart_node = WaveSubScheduleNode(
                mb_id=node.mb_id,
                device_id=node.device_id,
                type=node.type,
                available_time=node.available_time,
                time_cost=node.time_cost,
                mem_incr=node.mem_incr,
                subpart_start=node.subpart_end,
                subpart_end=node.subpart_end + 1,
                num_subparts=node.num_subparts,
                chunk_id=node.chunk_id,
            )
            self.schedulable_nodes.append(next_subpart_node)
        else:
            # update last scheduled block type
            if node.type in [0, 1]:
                self.last_scheduled_full_block_type = node.type
            self.last_scheduled_full_block_chunk_id[node.type] = node.chunk_id
            if node.type == 1 and node.chunk_id == 1:
                self.last_b_chunk1_block_mb_id = node.mb_id
        self.scheduled_nodes.append(node)
        self.cur_mem_usage += (
            self.M_F if node.type == 0 else self.M_B if node.type == 1 else self.M_W
        ) // self.num_subparts
        node.start_time = max(self._next_schedulable_time, node.available_time)
        node.end_time = node.start_time + node.time_cost // self.num_subparts
        self.update_next()
        return node

    def schedule_node(self):
        assert self._next_node is not None
        node = self._next_node
        return self._schedule_node_core(node)

    def schedule_node_force(self, mb_id, chunk_id, type, subpart_start, subpart_end):
        target_node = None
        for node in self.schedulable_nodes:
            if (
                node.mb_id == mb_id
                and node.chunk_id == chunk_id
                and node.type == type
                and node.subpart_start == subpart_start
                and node.subpart_end == subpart_end
            ):
                target_node = node
                break
        if target_node is None:
            raise ValueError("Force scheduling a node not in schedulable nodes")
        return self._schedule_node_core(target_node)


class DynZBWaveSubScheduler(DynZBUDSubScheduler):
    def __init__(self, system_cfg: SystemConfig, num_subparts: int = 1):
        self.system_cfg = system_cfg
        assert self.system_cfg.num_chunks == 2

        self.devices: List[WaveScheduleDevice] = [
            WaveScheduleDevice(i, self.system_cfg, num_subparts=num_subparts)
            for i in range(self.system_cfg.num_devices)
        ]

        self._schedule = None
        self.num_subparts = num_subparts

    def _get_next_block(self, cur_node: WaveSubScheduleNode):
        """Return next block of this microbatch (only F/B)
        dev, chunk, type
        """
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
        bw_delay_model = BandwidthDelayModel(self.system_cfg)
        bootstrap_schedule = WaveBootStrapSubSchedule(
            self.system_cfg, num_subparts=self.num_subparts
        )
        bootstrap_schedule.bootstrap()
        bs_schedule = bootstrap_schedule.get_bootstrap_schedule()
        # bootstrap_schedule.print_schedule()

        next_schedule_dev = self.next_device_to_schedule()
        assert next_schedule_dev == 0

        self.devices[next_schedule_dev].add_schedulable_node(
            [
                WaveSubScheduleNode(
                    mb_id=mb_id,
                    device_id=next_schedule_dev,
                    type=0,
                    available_time=0,
                    time_cost=self.get_T_F(next_schedule_dev),
                    mem_incr=self.get_M_F(next_schedule_dev),
                    subpart_start=0,
                    subpart_end=1,
                    num_subparts=self.num_subparts,
                    chunk_id=0,
                )
                for mb_id in range(self.num_mb)
            ]
        )

        while True:
            if any([len(tasks) != 0 for tasks in bs_schedule]):
                # force schedule the nodes
                # schedule the node with the earliest start time
                cur_schedule_dev: int = min(
                    range(len(bs_schedule)),
                    key=lambda x: bs_schedule[x][0].start_time
                    if len(bs_schedule[x]) > 0
                    else math.inf,
                )
                cur_simple_node: WaveSubScheduleNode = bs_schedule[
                    cur_schedule_dev
                ].pop(0)
                cur_node = self.devices[cur_schedule_dev].schedule_node_force(
                    cur_simple_node.mb_id,
                    cur_simple_node.chunk_id,
                    cur_simple_node.type,
                    cur_simple_node.subpart_start,
                    cur_simple_node.subpart_end,
                )
            else:
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
                        WaveSubScheduleNode(
                            mb_id=cur_node.mb_id,
                            device_id=cur_schedule_dev,
                            type=2,
                            available_time=0,
                            time_cost=self.get_T_W(cur_schedule_dev),
                            mem_incr=self.get_M_W(cur_schedule_dev),
                            subpart_start=0,
                            subpart_end=1,
                            num_subparts=self.num_subparts,
                            chunk_id=cur_node.chunk_id,
                        )
                    ]
                )

            if cur_node.subpart_end != self.num_subparts:
                continue

            # last subpart of the node has scheduled
            next_dev, next_chunk, next_type = self._get_next_block(cur_node)
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
                        WaveSubScheduleNode(
                            mb_id=cur_node.mb_id,
                            device_id=next_dev,
                            type=next_type,
                            available_time=cur_node_end_time
                            + lat_time
                            + bw_time_with_delay,
                            time_cost=next_time_cost,
                            mem_incr=next_mem_incr,
                            subpart_start=0,
                            subpart_end=1,
                            num_subparts=self.num_subparts,
                            chunk_id=next_chunk,
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
        # first: reorder patterns like 060606 (where 0 is W, 6 is F)
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
                if (
                    node.mb_id != next_node.mb_id
                    or node.type != next_node.type
                    or node.chunk_id != next_node.chunk_id
                ):
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
                        chunk_id=node.chunk_id,
                        task_type=tasktype_to_str[node.type],
                        end_time=node.end_time,
                        subpart_start=node.subpart_start,
                        subpart_end=node.subpart_end,
                        num_subparts=node.num_subparts,
                    )
                )
        return schedule
