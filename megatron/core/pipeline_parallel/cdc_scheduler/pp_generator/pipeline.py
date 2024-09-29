from dataclasses import dataclass
import enum
from typing import Callable, List, Optional, Tuple
from .pipeline_config import SystemConfig
from .heuristic_schedule import ZBVHeuristicSchedule
from .heuristic_schedule_v2 import ZBVHeuristicScheduleV2
from .zbv_heuristic import OfficialZBVHeuristicScheduler
from .heuristic_ud_schedule import UDHeuristicSchedule, ZBUDHeuristicSchedule
from .svg_event import draw_events, TIME_PER_UNIT
import os


class TaskNode:
    def __init__(
        self,
        task_type: str,
        device_id: int,
        microbatch_id: int,
        prev_device_task: "TaskNode",
        prev_microbatch_task: "TaskNode",
        next_microbatch_task: "TaskNode" = None,
    ) -> None:
        self.task_type = task_type
        self.device_id = device_id
        self.microbatch_id = microbatch_id
        self.prev_device_task = prev_device_task
        self.prev_microbatch_task = prev_microbatch_task
        self.next_microbatch_task = next_microbatch_task

        self.start_time = None
        self.completion_time = None

    def is_calculated(self):
        return self.start_time is not None and self.completion_time is not None

    def is_dependency_solved(self):
        if self.prev_device_task is None and self.prev_microbatch_task is None:
            return True

        solved = True

        if self.prev_device_task is not None:
            solved &= self.prev_device_task.is_calculated()

        if self.prev_microbatch_task is not None:
            solved &= self.prev_microbatch_task.is_calculated()

        return solved


class InterleavedTaskNode(TaskNode):
    def __init__(
        self,
        task_type: str,
        device_id: int,
        microbatch_id: int,
        chunk_id: int,
        prev_device_task: "TaskNode",
        prev_microbatch_task: "TaskNode",
        next_microbatch_task: "TaskNode" = None,
    ) -> None:
        super().__init__(
            task_type,
            device_id,
            microbatch_id,
            prev_device_task,
            prev_microbatch_task,
            next_microbatch_task,
        )
        self.chunk_id = chunk_id


class Pipeline:
    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        self._scalar_config_to_list()

        self.device_scheduled_tasks: List[List[TaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[TaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]

    def _scalar_config_to_list(self):
        for config in [
            self.sys_config.T_F,
            self.sys_config.T_B,
            self.sys_config.T_W,
            self.sys_config.M_F,
            self.sys_config.M_B,
            self.sys_config.M_W,
        ]:
            if not isinstance(config, list):
                config = [config] * self.sys_config.num_devices

    def _get_tasknode_from_device(
        self,
        mb: int,
        dev: int,
        condition: Callable[[TaskNode, Tuple], bool],
        seq_element,
    ) -> TaskNode:
        for task in self.device_scheduled_tasks[dev]:
            if condition(task, mb, seq_element):
                return task
        raise ValueError(
            f"Task not found for device {dev}, sequence element {seq_element}"
        )

    def schedule(self) -> None:
        raise NotImplementedError("Schedule method not implemented")

    def _resolve_batch_dependency(self) -> None:
        filled = any([len(x) != 0 for x in self.microbatch_scheduled_tasks])
        assert not filled, "Batch Dependency already resolved"

        sequence, task_node_match_condition = self._get_microbatch_sequence()
        num_mb = self.sys_config.num_microbatches
        for mb in range(num_mb):
            for i, seq_ele in enumerate(sequence):
                dev = seq_ele[0]
                cur_task = self._get_tasknode_from_device(
                    mb, dev, task_node_match_condition, seq_ele
                )
                if i != 0:
                    self.microbatch_scheduled_tasks[mb][
                        -1
                    ].next_microbatch_task = cur_task
                    cur_task.prev_microbatch_task = self.microbatch_scheduled_tasks[mb][
                        -1
                    ]

                self.microbatch_scheduled_tasks[mb].append(cur_task)

    def _get_microbatch_sequence(
        self,
    ) -> Tuple[List[Tuple[int, str]], Callable[[TaskNode, Tuple], bool]]:
        raise NotImplementedError("Microbatch sequence not implemented")

    def is_send_to_next_rank(self, prev_task: TaskNode, cur_task: TaskNode):
        """Helper function to decide if the communication is using send_next & recv_prev process group
        or send_prev & recv_next process group.
        Mostly for tie breaking when PP=2.

        Return:
            1 if send_next & recv_prev
            0 if on the same device
            -1 if send_prev & recv_next
        """
        prev_dev = prev_task.device_id
        cur_dev = cur_task.device_id
        prev_type = prev_task.task_type
        cur_type = cur_task.task_type
        if prev_dev == cur_dev:
            return 0
        assert prev_type == cur_type
        if prev_type == "F":
            return 1
        elif prev_type == "B":
            return -1
        else:
            raise ValueError("Unreachable")

    def _get_execution_time(self, dev: int, task_type: str) -> int:
        if task_type == "F":
            return self.sys_config.T_F[dev]
        elif task_type == "B":
            return self.sys_config.T_B[dev]
        else:
            # W block
            assert self.sys_config.T_W[dev] > 0
            return self.sys_config.T_W[dev]

    def solve_dependencies(self):
        cur_dev_idx_list = [0] * self.sys_config.num_devices
        self.device_scheduled_tasks[0][0].start_time = 0
        self.device_scheduled_tasks[0][0].completion_time = self._get_execution_time(
            0, "F"
        )
        cur_dev_idx_list[0] += 1

        while not all(
            i == len(self.device_scheduled_tasks[0]) for i in cur_dev_idx_list
        ):
            for dev in range(self.sys_config.num_devices):
                if cur_dev_idx_list[dev] < len(self.device_scheduled_tasks[dev]):
                    cur_task = self.device_scheduled_tasks[dev][cur_dev_idx_list[dev]]
                    if not cur_task.is_dependency_solved():
                        continue

                    # Schedule this task
                    prev_device_task = cur_task.prev_device_task
                    prev_microbatch_task = cur_task.prev_microbatch_task

                    if prev_device_task is not None:
                        assert (
                            prev_device_task.device_id == dev
                        ), f"Device {dev} {cur_task.task_type} {cur_task.microbatch_id} depends on {prev_device_task.device_id} {prev_device_task.task_type} {prev_device_task.microbatch_id}"
                    if prev_microbatch_task is not None:
                        assert (
                            prev_microbatch_task.microbatch_id == cur_task.microbatch_id
                        )

                    cur_dev_id = cur_task.device_id
                    compute_time = self._get_execution_time(dev, cur_task.task_type)

                    prev_microbatch_task_dev_id = (
                        prev_microbatch_task.device_id
                        if prev_microbatch_task is not None
                        else None
                    )
                    if prev_microbatch_task_dev_id is None:
                        comm_time = 0
                    else:
                        comm_time = self.sys_config.T_C[
                            prev_microbatch_task_dev_id, cur_dev_id
                        ]

                    if prev_device_task is not None:
                        cur_task.start_time = prev_device_task.completion_time
                    else:
                        cur_task.start_time = 0

                    if prev_microbatch_task is not None:
                        if prev_microbatch_task_dev_id != cur_dev_id:
                            cur_task.start_time = max(
                                cur_task.start_time,
                                prev_microbatch_task.completion_time + comm_time,
                            )
                        else:
                            cur_task.start_time = max(
                                cur_task.start_time,
                                prev_microbatch_task.completion_time,
                            )
                    cur_task.completion_time = cur_task.start_time + compute_time
                    cur_dev_idx_list[dev] += 1

    def print_debug_schedule(self, verbose=0):
        for dev in range(self.sys_config.num_devices):
            print(f"Device {dev}: ", end="")
            for task in self.device_scheduled_tasks[dev]:
                if verbose == 0:
                    print(f"-> {task.task_type} {task.microbatch_id}", end="")
                elif verbose == 1:
                    print(
                        f"-> {task.task_type} {task.microbatch_id} ({task.start_time}, {task.completion_time})",
                        end="",
                    )

            print("\n")

    def pipeline_name(self):
        raise NotImplementedError("Pipeline name not implemented")

    def has_multiple_chunks(self):
        return self.sys_config.num_chunks > 1

    def get_pipeline_first_stage_rank(self):
        raise NotImplementedError()

    def get_pipeline_last_stage_rank(self):
        raise NotImplementedError()

    def get_pipeline_execution_order(self) -> List[Tuple[int, int]]:
        """return a list of tuples (device_id, chunk_id)"""
        return [(dev, 0) for dev in range(self.sys_config.num_devices)]

    def get_device_scheduled_tasks(self) -> List[List[TaskNode]]:
        return self.device_scheduled_tasks

    def print_schedule(
        self,
        name: str = None,
        save: bool = False,
        time_range: int = 0,
        include_info: bool = True,
    ):
        global TIME_PER_UNIT
        if time_range > 0:
            longest_time = time_range
        else:
            longest_time = max(
                [x.completion_time for x in self.device_scheduled_tasks[0]]
            )
        time_scale = 1024 / longest_time * TIME_PER_UNIT
        events = [
            [
                {
                    "type": e.task_type,
                    "start_time": e.start_time * time_scale,
                    "completion_time": e.completion_time * time_scale,
                    "minibatch": e.microbatch_id,
                    "chunk": e.chunk_id if isinstance(e, InterleavedTaskNode) else 0,
                }
                for e in dev_evs
            ]
            for dev_evs in self.device_scheduled_tasks
        ]

        pipe_name = name if name is not None else self.pipeline_name()
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"{pipe_name}.svg"
        )
        d = draw_events(
            events,
            path,
            include_w=True,
            include_o=False,
            tail=50,
            longest_time=longest_time * time_scale,
            save=save,
            include_info=include_info,
        )
        return d

    def get_schedule_time(self, device_wise: bool = False):
        num_dev = self.sys_config.num_devices
        if device_wise:
            return max(
                [
                    tasks[-1].completion_time - tasks[0].start_time
                    for tasks in self.device_scheduled_tasks
                ]
            )

        return max(
            [
                max([x.completion_time for x in self.device_scheduled_tasks[i]])
                for i in range(num_dev)
            ]
        ) - min([x.start_time for x in self.device_scheduled_tasks[0]])

    def get_bubble_ratio(self, device_wise: bool = False):
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches
        num_chunks = self.sys_config.num_chunks
        cfg = self.sys_config
        total_effective_compute = [
            num_mb * num_chunks * (cfg.T_F[i] + cfg.T_B[i] + cfg.T_W[i])
            for i in range(num_dev)
        ]
        if device_wise:
            ratio = [
                total_effective_compute[i]
                / (tasks[-1].completion_time - tasks[0].start_time)
                for i, tasks in enumerate(self.device_scheduled_tasks)
            ]
            return 1 - min(ratio)
        else:
            total_time = sum(
                [
                    tasks[-1].completion_time - tasks[0].start_time
                    for tasks in self.device_scheduled_tasks
                ]
            )
            return 1 - sum(total_effective_compute) / total_time

    def get_total_time_and_bubble_ratio(self):
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches
        num_chunks = self.sys_config.num_chunks
        total_time = self.get_schedule_time()
        total_effective_compute = (
            num_mb
            * num_chunks
            * (
                sum(self.sys_config.T_F)
                + sum(self.sys_config.T_B)
                + sum(self.sys_config.T_W)
            )
        )
        bubble_ratio = 1 - total_effective_compute / num_dev / total_time
        return total_time, bubble_ratio

    def compute_schedule_time_and_bubble(self):
        self.schedule()
        self.solve_dependencies()
        return self.get_total_time_and_bubble_ratio()


class OneFOneBPipeline(Pipeline):
    def pipeline_name(self):
        return "1F1B"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return self.sys_config.num_devices - 1

    def _get_microbatch_sequence(
        self,
    ) -> Tuple[List[Tuple[int, str]], Callable[[TaskNode, Tuple], bool]]:
        num_dev = self.sys_config.num_devices

        sequence = []
        for dev in range(num_dev):
            sequence.append((dev, "F"))

        for dev in reversed(range(num_dev)):
            sequence.append((dev, "B"))

        # print(sequence)
        def condition(task: TaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
            )

        return sequence, condition

    def schedule(self):
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches

        assert (
            num_mb % num_dev == 0
        ), "For now, number of microbatches should be divisible by number of devices"

        # Device Dependency
        for dev in range(num_dev):
            for mb in range(num_dev - dev - 1):
                # warmup
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "F",
                        dev,
                        mb,
                        self.device_scheduled_tasks[dev][-1] if mb > 0 else None,
                        None,
                    )
                )

        for dev in range(num_dev):
            for mb in range(num_dev - dev - 1, num_mb):
                # steady
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "F",
                        dev,
                        mb,
                        self.device_scheduled_tasks[dev][-1] if mb > 0 else None,
                        None,
                    )
                )
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "B",
                        dev,
                        mb - (num_dev - dev - 1),
                        self.device_scheduled_tasks[dev][-1],
                        None,
                    )
                )

        for dev in range(num_dev):
            for mb in range(num_mb - num_dev + dev + 1, num_mb):
                # teardown
                self.device_scheduled_tasks[dev].append(
                    TaskNode("B", dev, mb, self.device_scheduled_tasks[dev][-1], None)
                )

        # Batch Dependency
        self._resolve_batch_dependency()


class GpipePipeline(Pipeline):
    def pipeline_name(self):
        return "GPipe"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return self.sys_config.num_devices - 1

    def _get_microbatch_sequence(
        self,
    ) -> Tuple[List[Tuple[int, str]], Callable[[TaskNode, int, Tuple], bool]]:
        num_dev = self.sys_config.num_devices
        sequence = []
        for dev in range(num_dev):
            sequence.append((dev, "F"))

        for dev in reversed(range(num_dev)):
            sequence.append((dev, "B"))

        def condition(task: TaskNode, mb: int, seq_ele: Tuple) -> bool:
            return task.task_type == seq_ele[1] and task.microbatch_id == mb

        return sequence, condition

    def schedule(self) -> None:
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches

        # Device Dependency
        for dev in range(num_dev):
            for mb in range(num_mb):
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "F",
                        dev,
                        mb,
                        self.device_scheduled_tasks[dev][-1] if mb > 0 else None,
                        None,
                    )
                )

            for mb in range(num_mb):
                self.device_scheduled_tasks[dev].append(
                    TaskNode("B", dev, mb, self.device_scheduled_tasks[dev][-1], None)
                )

        # Batch Dependency
        self._resolve_batch_dependency()


class Interleaved1F1BPipeline(Pipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        assert sys_config.num_chunks > 0
        self.sys_config = sys_config
        # self._scalar_config_to_list()

        self.device_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]

    def pipeline_name(self):
        return "Interleaved1F1B"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return self.sys_config.num_devices - 1

    def get_pipeline_execution_order(self) -> List[Tuple[int, int]]:
        return [
            (dev, chunk)
            for chunk in range(self.sys_config.num_chunks)
            for dev in range(self.sys_config.num_devices)
        ]

    def _get_microbatch_sequence(
        self,
    ) -> Tuple[
        List[Tuple[int, str, int]], Callable[[InterleavedTaskNode, int, Tuple], bool]
    ]:
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches
        num_chunks = self.sys_config.num_chunks
        assert (
            num_mb % num_dev == 0
        ), "For now, number of microbatches should be divisible by number of devices"
        sequence = []

        for chunk in range(num_chunks):
            for dev in range(num_dev):
                sequence.append((dev, "F", chunk))

        for chunk in reversed(range(num_chunks)):
            for dev in reversed(range(num_dev)):
                sequence.append((dev, "B", chunk))

        def condition(task: InterleavedTaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
                and task.chunk_id == seq_ele[2]
            )

        return sequence, condition

    def _resolve_batch_dependency(self) -> None:
        filled = any([len(x) != 0 for x in self.microbatch_scheduled_tasks])
        assert not filled, "Batch Dependency already resolved"

        sequence, task_node_match_condition = self._get_microbatch_sequence()
        num_mb = self.sys_config.num_microbatches
        for mb in range(num_mb):
            for i, seq_ele in enumerate(sequence):
                dev = seq_ele[0]
                cur_task = self._get_tasknode_from_device(
                    mb, dev, task_node_match_condition, seq_ele
                )
                if i != 0:
                    self.microbatch_scheduled_tasks[mb][
                        -1
                    ].next_microbatch_task = cur_task
                    cur_task.prev_microbatch_task = self.microbatch_scheduled_tasks[mb][
                        -1
                    ]
                cur_task.microbatch_id = mb
                self.microbatch_scheduled_tasks[mb].append(cur_task)

    def print_debug_schedule(self, verbose=0):
        for dev in range(self.sys_config.num_devices):
            print(f"Device {dev}: ", end="")
            for task in self.device_scheduled_tasks[dev]:
                if verbose == 0:
                    print(
                        f"-> {task.task_type} {task.microbatch_id}.{task.chunk_id}",
                        end="",
                    )
                elif verbose == 1:
                    print(
                        f"-> {task.task_type} {task.microbatch_id}.{task.chunk_id} ({task.start_time}, {task.completion_time})",
                        end="",
                    )

            print("\n")

    def schedule(self):
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches
        num_chunks = self.sys_config.num_chunks

        dev_fw_mb = [0 for _ in range(num_dev)]
        dev_fw_chunk = [0 for _ in range(num_dev)]
        dev_bw_mb = [0 for _ in range(num_dev)]
        dev_bw_chunk = [num_chunks - 1 for _ in range(num_dev)]

        def fw_mb_chunk_step(dev):
            nonlocal dev_fw_mb, dev_fw_chunk
            if chunk == num_chunks - 1 and (mb + 1) % num_dev == 0:
                dev_fw_mb[dev] += 1
                dev_fw_chunk[dev] = 0
            elif (mb + 1) % num_dev == 0:
                dev_fw_mb[dev] += 1 - num_dev
                dev_fw_chunk[dev] += 1
            else:
                dev_fw_mb[dev] += 1

        def bw_mb_chunk_step(dev):
            nonlocal dev_bw_mb, dev_bw_chunk
            if chunk == 0 and (mb + 1) % num_dev == 0:
                dev_bw_mb[dev] += 1
                dev_bw_chunk[dev] = num_chunks - 1
            elif (mb + 1) % num_dev == 0:
                dev_bw_mb[dev] += 1 - num_dev
                dev_bw_chunk[dev] -= 1
            else:
                dev_bw_mb[dev] += 1

        # warmup
        for dev in range(num_dev):
            num_warmup_stages = (num_dev - dev - 1) * 2 + (num_chunks - 1) * num_dev
            for stage in range(num_warmup_stages):
                mb = dev_fw_mb[dev]
                chunk = dev_fw_chunk[dev]
                fw_mb_chunk_step(dev)
                self.device_scheduled_tasks[dev].append(
                    InterleavedTaskNode(
                        "F",
                        dev,
                        mb,
                        chunk,
                        self.device_scheduled_tasks[dev][-1] if stage > 0 else None,
                        None,
                    )
                )

        # steady
        for dev in range(num_dev):
            num_warmup_stages = (num_dev - dev - 1) * 2 + (num_chunks - 1) * num_dev
            assert num_warmup_stages <= num_chunks * num_mb
            for stage in range(num_warmup_stages, num_chunks * num_mb):
                mb = dev_fw_mb[dev]
                chunk = dev_fw_chunk[dev]
                fw_mb_chunk_step(dev)
                self.device_scheduled_tasks[dev].append(
                    InterleavedTaskNode(
                        "F",
                        dev,
                        mb,
                        chunk,
                        self.device_scheduled_tasks[dev][-1] if stage > 0 else None,
                        None,
                    )
                )
                mb = dev_bw_mb[dev]
                chunk = dev_bw_chunk[dev]
                bw_mb_chunk_step(dev)
                self.device_scheduled_tasks[dev].append(
                    InterleavedTaskNode(
                        "B", dev, mb, chunk, self.device_scheduled_tasks[dev][-1], None
                    )
                )

        # teardown
        for dev in range(num_dev):
            num_warmup_stages = (num_dev - dev - 1) * 2 + (num_chunks - 1) * num_dev
            for stage in range(num_warmup_stages):
                mb = dev_bw_mb[dev]
                chunk = dev_bw_chunk[dev]
                bw_mb_chunk_step(dev)
                self.device_scheduled_tasks[dev].append(
                    InterleavedTaskNode(
                        "B", dev, mb, chunk, self.device_scheduled_tasks[dev][-1], None
                    )
                )

        # self.print_debug_schedule(verbose=0)
        self._resolve_batch_dependency()


class Hanayo1F1BPipeline(Interleaved1F1BPipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        # self._scalar_config_to_list()

        self.device_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]

        assert sys_config.num_microbatches % sys_config.num_devices == 0
        assert sys_config.num_chunks % 2 == 0

    def pipeline_name(self):
        return "Hanayo"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        num_chunks = self.sys_config.num_chunks
        return 0 if num_chunks % 2 == 0 else self.sys_config.num_devices - 1

    def get_pipeline_execution_order(self) -> List[Tuple[int, int]]:
        ret = []
        for wave in range(self.sys_config.num_chunks // 2):
            ret += [(dev, wave * 2) for dev in range(self.sys_config.num_devices)] + [
                (dev, wave * 2 + 1)
                for dev in reversed(range(self.sys_config.num_devices))
            ]
        return ret

    def _get_microbatch_sequence(
        self,
    ) -> Tuple[
        List[Tuple[int, str, int]], Callable[[InterleavedTaskNode, int, Tuple], bool]
    ]:
        num_dev = self.sys_config.num_devices
        num_chunks = self.sys_config.num_chunks
        sequence = []

        for wave in range(num_chunks // 2):
            for dev in range(num_dev):
                sequence.append((dev, "F", 2 * wave))
            for dev in reversed(range(num_dev)):
                sequence.append((dev, "F", 2 * wave + 1))

        for wave in reversed(range(num_chunks // 2)):
            for dev in range(num_dev):
                sequence.append((dev, "B", 2 * wave + 1))
            for dev in reversed(range(num_dev)):
                sequence.append((dev, "B", 2 * wave))

        def condition(task: InterleavedTaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
                and task.chunk_id == seq_ele[2]
            )

        return sequence, condition

    def is_send_to_next_rank(
        self, prev_task: InterleavedTaskNode, cur_task: InterleavedTaskNode
    ):
        prev_dev = prev_task.device_id
        cur_dev = cur_task.device_id
        prev_type = prev_task.task_type
        cur_type = cur_task.task_type
        prev_chunk = prev_task.chunk_id
        cur_chunk = cur_task.chunk_id
        if prev_dev == cur_dev:
            return 0
        assert prev_type == cur_type
        assert prev_chunk == cur_chunk
        if prev_type == "F":
            return 1 if prev_chunk % 2 == 0 else -1
        elif prev_type == "B":
            return 1 if prev_chunk % 2 == 1 else -1
        else:
            raise ValueError("Unreachable")

    def schedule(self):
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches
        num_chunks = self.sys_config.num_chunks
        sequence = self._get_microbatch_sequence()[0]
        # print(sequence)

        # a mini scheduler

        class SchedUnitType(enum.Enum):
            F = enum.auto()
            B_0 = enum.auto()
            B_1 = enum.auto()
            X = enum.auto()  # Restricted Zone
            U = enum.auto()  # Unallocated

        @dataclass
        class SchedUnit:
            type: SchedUnitType = SchedUnitType.U
            mb: int = -1
            chunk: int = -1
            index: int = -1

        class HanayoScheduler:
            def __init__(self) -> None:
                self.dev_schedule: List[List[SchedUnit]] = [
                    [SchedUnit() for _ in range(num_chunks * num_mb * 3 * 2)]
                    for _ in range(num_dev)
                ]

            def _find_next_index(self, dev: int, task_type: str, cur_index: int) -> int:
                for i in range(cur_index + 1, len(self.dev_schedule[dev])):
                    if task_type == "F":
                        if self.dev_schedule[dev][i].type == SchedUnitType.U:
                            return i
                    else:
                        if (
                            self.dev_schedule[dev][i].type == SchedUnitType.U
                            and self.dev_schedule[dev][i + 1].type == SchedUnitType.U
                        ):
                            return i
                raise ValueError(f"Cannot find next index for device {dev}")

            def _get_schedunit_index(
                self, dev: int, mb: int, chunk: int, type: SchedUnitType, start: int = 0
            ) -> int:
                for i, unit in enumerate(self.dev_schedule[dev][start:]):
                    if unit.mb == mb and unit.chunk == chunk and unit.type == type:
                        return i
                raise ValueError(
                    f"Cannot find schedunit for device {dev} {mb} {chunk} {type}"
                )

            def debug_print(self):
                for dev in range(num_dev):
                    print(f"Device {dev}: ", end="")
                    for unit in self.dev_schedule[dev]:
                        print(" | " + unit.type.name, end="")
                    print(" |\n")

            def schedule(self) -> None:
                start_point_next_repeat = -1
                for repeat in range(num_mb // num_dev):
                    cur_index = start_point_next_repeat
                    start_of_current_repeat = start_point_next_repeat
                    mb_start = repeat * num_dev
                    # schedule mb 0
                    for dev, task_type, chunk in sequence:
                        cur_index = self._find_next_index(dev, task_type, cur_index)
                        if task_type == "F":
                            self.dev_schedule[dev][cur_index] = SchedUnit(
                                SchedUnitType.F, mb_start, chunk, cur_index
                            )
                        else:
                            self.dev_schedule[dev][cur_index] = SchedUnit(
                                SchedUnitType.B_0, mb_start, chunk, cur_index
                            )
                            self.dev_schedule[dev][cur_index + 1] = SchedUnit(
                                SchedUnitType.B_1, mb_start, chunk, cur_index + 1
                            )
                            cur_index += 1
                            if dev == 0 and chunk == 0:
                                start_point_next_repeat = cur_index

                    # set the restricted zone as in the paper
                    for dev in range(num_dev):
                        target_index = self._get_schedunit_index(
                            dev, mb_start, 1, SchedUnitType.F
                        )
                        for i in range(1, num_dev - dev):
                            if (
                                self.dev_schedule[dev][target_index - i].type
                                == SchedUnitType.U
                            ):
                                self.dev_schedule[dev][
                                    target_index - i
                                ].type = SchedUnitType.X

                    for dev in range(num_dev):
                        target_index = self._get_schedunit_index(
                            dev, mb_start, num_chunks - 1, SchedUnitType.B_0
                        )
                        for i in range(1, dev + 1):
                            if (
                                self.dev_schedule[dev][target_index - i].type
                                == SchedUnitType.U
                            ):
                                self.dev_schedule[dev][
                                    target_index - i
                                ].type = SchedUnitType.X

                    # schedule mb 1 to num_dev - 1
                    for mb in range(mb_start + 1, mb_start + num_dev):
                        cur_index = start_of_current_repeat
                        for dev, task_type, chunk in sequence:
                            cur_index = self._find_next_index(dev, task_type, cur_index)
                            if task_type == "F":
                                self.dev_schedule[dev][cur_index] = SchedUnit(
                                    SchedUnitType.F, mb, chunk, cur_index
                                )
                            else:
                                self.dev_schedule[dev][cur_index] = SchedUnit(
                                    SchedUnitType.B_0, mb, chunk, cur_index
                                )
                                self.dev_schedule[dev][cur_index + 1] = SchedUnit(
                                    SchedUnitType.B_1, mb, chunk, cur_index + 1
                                )
                                cur_index += 1

            def generate_task_nodes(
                self, device_scheduled_tasks: List[List[InterleavedTaskNode]]
            ):
                for dev in range(num_dev):
                    for unit in self.dev_schedule[dev]:
                        if unit.type == SchedUnitType.F:
                            device_scheduled_tasks[dev].append(
                                InterleavedTaskNode(
                                    "F",
                                    dev,
                                    unit.mb,
                                    unit.chunk,
                                    device_scheduled_tasks[dev][-1]
                                    if len(device_scheduled_tasks[dev]) > 0
                                    else None,
                                    None,
                                )
                            )
                        elif unit.type == SchedUnitType.B_0:
                            device_scheduled_tasks[dev].append(
                                InterleavedTaskNode(
                                    "B",
                                    dev,
                                    unit.mb,
                                    unit.chunk,
                                    device_scheduled_tasks[dev][-1],
                                    None,
                                )
                            )

        scheduler = HanayoScheduler()
        scheduler.schedule()
        scheduler.generate_task_nodes(self.device_scheduled_tasks)
        self._resolve_batch_dependency()


class ZBH1Pipeline(Pipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        # self._scalar_config_to_list()
        assert all([x > 0 for x in self.sys_config.T_W])

        self.device_scheduled_tasks: List[List[TaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[TaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]

        assert sys_config.num_microbatches % sys_config.num_devices == 0

    def pipeline_name(self):
        return "ZBH1"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return self.sys_config.num_devices - 1

    def _get_microbatch_sequence(self):
        num_dev = self.sys_config.num_devices
        sequence = []
        for dev in range(num_dev):
            sequence.append((dev, "F"))
        for dev in reversed(range(num_dev)):
            sequence.append((dev, "B"))

        def condition(task: TaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
            )

        return sequence, condition

    def schedule(self):
        num_dev = self.sys_config.num_devices
        num_mb = self.sys_config.num_microbatches

        next_w_mb = [0 for _ in range(num_dev)]

        for dev in range(num_dev):
            num_warpup = num_dev - dev - 1
            num_remaining = num_mb - num_warpup

            # warmup
            for i in range(num_warpup):
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "F",
                        dev,
                        i,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )

            # steady
            for i in range(num_remaining):
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "F",
                        dev,
                        i + num_warpup,
                        self.device_scheduled_tasks[dev][-1]
                        if i + num_warpup > 0
                        else None,
                        None,
                    )
                )
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "B",
                        dev,
                        i,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )
                if i >= dev:
                    self.device_scheduled_tasks[dev].append(
                        TaskNode(
                            "W",
                            dev,
                            next_w_mb[dev],
                            self.device_scheduled_tasks[dev][-1],
                            None,
                        )
                    )
                    next_w_mb[dev] += 1

            for i in range(num_warpup):
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "B",
                        dev,
                        i + num_remaining,
                        self.device_scheduled_tasks[dev][-1],
                        None,
                    )
                )
                if next_w_mb[dev] < num_mb:
                    self.device_scheduled_tasks[dev].append(
                        TaskNode(
                            "W",
                            dev,
                            next_w_mb[dev],
                            self.device_scheduled_tasks[dev][-1],
                            None,
                        )
                    )
                    next_w_mb[dev] += 1

            while next_w_mb[dev] < num_mb:
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        "W",
                        dev,
                        next_w_mb[dev],
                        self.device_scheduled_tasks[dev][-1],
                        None,
                    )
                )
                next_w_mb[dev] += 1

        # self.print_debug_schedule(verbose=0)
        self._resolve_batch_dependency()


class AutoUDZBPipeline(Pipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        assert all([x > 0 for x in self.sys_config.T_W])

        self.device_scheduled_tasks: List[List[TaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[TaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]

    def pipeline_name(self):
        return "AutoUDZB"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return self.sys_config.num_devices - 1

    def _get_microbatch_sequence(self):
        num_dev = self.sys_config.num_devices
        sequence = []
        for dev in range(num_dev):
            sequence.append((dev, "F"))
        for dev in reversed(range(num_dev)):
            sequence.append((dev, "B"))

        def condition(task: TaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
            )

        return sequence, condition

    def schedule(self, schedule: Optional[List[List[Tuple[int, int, str, int]]]]):
        num_dev = self.sys_config.num_devices

        if schedule is None:
            Warning("MILP solver failed to find a schedule")
            return

        for dev in range(num_dev):
            for i, (dev_id, mb_id, task_type, _) in enumerate(schedule[dev]):
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        task_type,
                        dev_id,
                        mb_id,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )

        # self.print_debug_schedule(verbose=0)
        self._resolve_batch_dependency()


class AutoWaveZBPipeline(Interleaved1F1BPipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        assert all([x > 0 for x in self.sys_config.T_W])

        self.device_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]

    def pipeline_name(self):
        return "AutoWaveZB"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return (
            0
            if self.sys_config.num_chunks % 2 == 0
            else self.sys_config.num_devices - 1
        )

    def get_pipeline_execution_order(self) -> List[Tuple[int, int]]:
        return [(dev, 0) for dev in range(self.sys_config.num_devices)] + [
            (dev, 1) for dev in reversed(range(self.sys_config.num_devices))
        ]

    def _get_microbatch_sequence(self):
        num_dev = self.sys_config.num_devices
        num_chunk = self.sys_config.num_chunks
        sequence = []
        for chunk in range(num_chunk):
            if chunk % 2 == 0:
                for dev in range(num_dev):
                    sequence.append((dev, "F", chunk))
            else:
                for dev in reversed(range(num_dev)):
                    sequence.append((dev, "F", chunk))

        for chunk in reversed(range(num_chunk)):
            if chunk % 2 == 0:
                for dev in reversed(range(num_dev)):
                    sequence.append((dev, "B", chunk))
            else:
                for dev in range(num_dev):
                    sequence.append((dev, "B", chunk))

        def condition(task: InterleavedTaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
                and task.chunk_id == seq_ele[2]
            )

        return sequence, condition

    def is_send_to_next_rank(
        self, prev_task: InterleavedTaskNode, cur_task: InterleavedTaskNode
    ):
        prev_dev = prev_task.device_id
        cur_dev = cur_task.device_id
        prev_type = prev_task.task_type
        cur_type = cur_task.task_type
        prev_chunk = prev_task.chunk_id
        cur_chunk = cur_task.chunk_id
        if prev_dev == cur_dev:
            return 0
        assert prev_type == cur_type
        assert prev_chunk == cur_chunk
        if prev_type == "F":
            return 1 if prev_chunk % 2 == 0 else -1
        elif prev_type == "B":
            return 1 if prev_chunk % 2 == 1 else -1
        else:
            raise ValueError("Unreachable")

    def schedule(self, schedule: Optional[List[List[Tuple[int, int, int, str, int]]]]):
        num_dev = self.sys_config.num_devices

        if schedule is None:
            Warning("MILP solver failed to find a schedule")
            return

        for dev in range(num_dev):
            for i, (dev_id, mb_id, chunk_id, task_type, _) in enumerate(schedule[dev]):
                self.device_scheduled_tasks[dev].append(
                    InterleavedTaskNode(
                        task_type,
                        dev_id,
                        mb_id,
                        chunk_id,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )

        # self.print_debug_schedule(verbose=0)
        self._resolve_batch_dependency()


class HeuristicWaveZBPipeline(Interleaved1F1BPipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        assert all([x > 0 for x in self.sys_config.T_W])

        self.device_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]
        self.scheduler = ZBVHeuristicSchedule(self.sys_config)

    def pipeline_name(self):
        return "HeuristicWaveZB"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return (
            0
            if self.sys_config.num_chunks % 2 == 0
            else self.sys_config.num_devices - 1
        )

    def get_pipeline_execution_order(self) -> List[Tuple[int, int]]:
        return [(dev, 0) for dev in range(self.sys_config.num_devices)] + [
            (dev, 1) for dev in reversed(range(self.sys_config.num_devices))
        ]

    def _get_microbatch_sequence(self):
        num_dev = self.sys_config.num_devices
        num_chunk = self.sys_config.num_chunks
        sequence = []
        for chunk in range(num_chunk):
            if chunk % 2 == 0:
                for dev in range(num_dev):
                    sequence.append((dev, "F", chunk))
            else:
                for dev in reversed(range(num_dev)):
                    sequence.append((dev, "F", chunk))

        for chunk in reversed(range(num_chunk)):
            if chunk % 2 == 0:
                for dev in reversed(range(num_dev)):
                    sequence.append((dev, "B", chunk))
            else:
                for dev in range(num_dev):
                    sequence.append((dev, "B", chunk))

        def condition(task: InterleavedTaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
                and task.chunk_id == seq_ele[2]
            )

        return sequence, condition

    def is_send_to_next_rank(
        self, prev_task: InterleavedTaskNode, cur_task: InterleavedTaskNode
    ):
        prev_dev = prev_task.device_id
        cur_dev = cur_task.device_id
        prev_type = prev_task.task_type
        cur_type = cur_task.task_type
        prev_chunk = prev_task.chunk_id
        cur_chunk = cur_task.chunk_id
        if prev_dev == cur_dev:
            return 0
        assert prev_type == cur_type
        assert prev_chunk == cur_chunk
        if prev_type == "F":
            return 1 if prev_chunk % 2 == 0 else -1
        elif prev_type == "B":
            return 1 if prev_chunk % 2 == 1 else -1
        else:
            raise ValueError("Unreachable")

    def schedule(self):
        self.scheduler.schedule()
        schedule = self.scheduler.get_schedule()

        num_dev = self.sys_config.num_devices

        for dev in range(num_dev):
            for i, (dev_id, mb_id, chunk_id, task_type, _) in enumerate(schedule[dev]):
                self.device_scheduled_tasks[dev].append(
                    InterleavedTaskNode(
                        task_type,
                        dev_id,
                        mb_id,
                        chunk_id,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )

        self._resolve_batch_dependency()


class HeuristicUDPipeline(OneFOneBPipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        super().__init__(sys_config)
        self.scheduler = UDHeuristicSchedule(self.sys_config)

    def pipeline_name(self):
        return "HeuristicUD"

    def schedule(self):
        self.scheduler.schedule()
        schedule = self.scheduler.get_schedule()

        num_dev = self.sys_config.num_devices

        for dev in range(num_dev):
            for i, (dev_id, mb_id, task_type, _) in enumerate(schedule[dev]):
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        task_type,
                        dev_id,
                        mb_id,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )

        self._resolve_batch_dependency()


class HeuristicZBUDPipeline(ZBH1Pipeline):
    def __init__(self, sys_config: SystemConfig) -> None:
        super().__init__(sys_config)
        self.scheduler = ZBUDHeuristicSchedule(self.sys_config)

    def pipeline_name(self):
        return "HeuristicZBUD"

    def schedule(self):
        self.scheduler.schedule()
        schedule = self.scheduler.get_schedule()

        num_dev = self.sys_config.num_devices

        for dev in range(num_dev):
            for i, (dev_id, mb_id, task_type, _) in enumerate(schedule[dev]):
                self.device_scheduled_tasks[dev].append(
                    TaskNode(
                        task_type,
                        dev_id,
                        mb_id,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )

        self._resolve_batch_dependency()


class HeuristicWaveZBPipelineV2(HeuristicWaveZBPipeline):
    def pipeline_name(self):
        return "HeuristicWaveZB(V2)"

    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        assert all([x > 0 for x in self.sys_config.T_W])

        self.device_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]
        self.scheduler = ZBVHeuristicScheduleV2(self.sys_config)


class HeuristicZBVPipeline(Interleaved1F1BPipeline):
    """
    adapt from zero-bubble repo, only support constant comm time
    """

    def __init__(self, sys_config: SystemConfig) -> None:
        self.sys_config = sys_config
        assert all([x > 0 for x in self.sys_config.T_W])

        self.device_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[InterleavedTaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]
        self.scheduler = OfficialZBVHeuristicScheduler(self.sys_config)

    def pipeline_name(self):
        return "HeuristicZBV(official)"

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return 0

    def get_pipeline_execution_order(self) -> List[Tuple[int, int]]:
        return [(dev, 0) for dev in range(self.sys_config.num_devices)] + [
            (dev, 1) for dev in reversed(range(self.sys_config.num_devices))
        ]

    def _get_microbatch_sequence(self):
        num_dev = self.sys_config.num_devices
        num_chunk = self.sys_config.num_chunks
        sequence = []
        for chunk in range(num_chunk):
            if chunk % 2 == 0:
                for dev in range(num_dev):
                    sequence.append((dev, "F", chunk))
            else:
                for dev in reversed(range(num_dev)):
                    sequence.append((dev, "F", chunk))

        for chunk in reversed(range(num_chunk)):
            if chunk % 2 == 0:
                for dev in reversed(range(num_dev)):
                    sequence.append((dev, "B", chunk))
            else:
                for dev in range(num_dev):
                    sequence.append((dev, "B", chunk))

        def condition(task: InterleavedTaskNode, mb: int, seq_ele: Tuple) -> bool:
            return (
                task.device_id == seq_ele[0]
                and task.task_type == seq_ele[1]
                and task.microbatch_id == mb
                and task.chunk_id == seq_ele[2]
            )

        return sequence, condition

    def is_send_to_next_rank(
        self, prev_task: InterleavedTaskNode, cur_task: InterleavedTaskNode
    ):
        prev_dev = prev_task.device_id
        cur_dev = cur_task.device_id
        prev_type = prev_task.task_type
        cur_type = cur_task.task_type
        prev_chunk = prev_task.chunk_id
        cur_chunk = cur_task.chunk_id
        if prev_dev == cur_dev:
            return 0
        assert prev_type == cur_type
        assert prev_chunk == cur_chunk
        if prev_type == "F":
            return 1 if prev_chunk % 2 == 0 else -1
        elif prev_type == "B":
            return 1 if prev_chunk % 2 == 1 else -1
        else:
            raise ValueError("Unreachable")

    def schedule(self):
        schedule = self.scheduler.get_schedule()

        num_dev = self.sys_config.num_devices

        for dev in range(num_dev):
            for i, (dev_id, mb_id, chunk_id, task_type, _) in enumerate(schedule[dev]):
                self.device_scheduled_tasks[dev].append(
                    InterleavedTaskNode(
                        task_type,
                        dev_id,
                        mb_id,
                        chunk_id,
                        self.device_scheduled_tasks[dev][-1] if i > 0 else None,
                        None,
                    )
                )

        self._resolve_batch_dependency()


def get_default_static_schedule(
    pipeline_name: str, num_devices: int, num_microbatches: int
):
    default_cfg = SystemConfig(
        num_devices=num_devices,
        num_microbatches=num_microbatches,
        T_F=20,
        T_B=40,
        T_W=0,
        T_C=0,
    )
    iv_1f1b_cfg = SystemConfig(
        num_devices=num_devices,
        num_microbatches=num_microbatches,
        T_F=10,
        T_B=20,
        T_W=0,
        T_C=0,
        num_chunks=2,
    )
    zbh1_cfg = SystemConfig(
        num_devices=num_devices,
        num_microbatches=num_microbatches,
        T_F=10,
        T_B=10,
        T_W=10,
        T_C=0,
    )
    zbv_cfg = SystemConfig(
        num_devices=num_devices,
        num_microbatches=num_microbatches,
        T_F=10,
        T_B=10,
        T_W=10,
        T_C=0,
        num_chunks=2,
        M_F=2,
        M_B=-1,
        M_W=-1,
        M_Limit=num_devices * 2 * 2,
    )
    if pipeline_name == "1F1B":
        pipeline = OneFOneBPipeline(default_cfg)
    elif pipeline_name == "GPipe":
        pipeline = GpipePipeline(default_cfg)
    elif pipeline_name == "Interleaved1F1B":
        pipeline = Interleaved1F1BPipeline(iv_1f1b_cfg)
    elif pipeline_name == "Hanayo":
        pipeline = Hanayo1F1BPipeline(iv_1f1b_cfg)
    elif pipeline_name == "ZBH1":
        pipeline = ZBH1Pipeline(zbh1_cfg)
    elif pipeline_name == "ZBV":
        pipeline = HeuristicZBVPipeline(zbv_cfg)
    else:
        raise ValueError(f"Pipeline {pipeline_name} not supported")

    pipeline.schedule()
    pipeline.solve_dependencies()

    return pipeline
