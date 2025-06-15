from typing import Callable, List, Optional, Tuple

from .pipeline_config import PipelineBlockDesc, SystemConfig
from .heuristic_ud_subschedule import DynZBUDSubScheduler
from .pipeline import Pipeline, TaskNode
from .util import BandwidthDelayModel

class SubTaskNode(TaskNode):
    def __init__(
        self,
        task_type: str,
        device_id: int,
        microbatch_id: int,
        prev_device_task: "SubTaskNode",
        prev_microbatch_task: "SubTaskNode",
        subpart_start: int = 0,
        subpart_end: int = 1,
        num_subparts: int = 1,
        chunk_id: int = 0,
        next_microbatch_task: "SubTaskNode" = None,
        start_time: int = None,
        completion_time: int = None,
        post_send_time: int = -1
    ) -> None:
        super().__init__(
            task_type=task_type,
            device_id=device_id,
            microbatch_id=microbatch_id,
            prev_device_task=prev_device_task,
            prev_microbatch_task=prev_microbatch_task,
            chunk_id=chunk_id,
            next_microbatch_task=next_microbatch_task,
            start_time=start_time,
            completion_time=completion_time,
            post_send_time=post_send_time,
        )
        self.subpart_start = subpart_start
        self.subpart_end = subpart_end
        self.num_subparts = num_subparts


class SubPipeline(Pipeline):
    def __init__(self, sys_config: SystemConfig, num_subparts: int = 1) -> None:
        self.sys_config = sys_config
        self.num_subparts = num_subparts

        self.device_scheduled_tasks: List[List[SubTaskNode]] = [
            [] for _ in range(self.sys_config.num_devices)
        ]
        self.microbatch_scheduled_tasks: List[List[SubTaskNode]] = [
            [] for _ in range(self.sys_config.num_microbatches)
        ]

    def _get_tasknode_from_device(
        self,
        mb: int,
        dev: int,
        chunk_id: int,
        condition: Callable[[SubTaskNode, Tuple], bool],
        seq_element,
        begin_idx: int = 0,
    ) -> List[SubTaskNode]:
        ret = []
        for task in self.device_scheduled_tasks[dev][begin_idx:]:
            if condition(task, mb, seq_element, chunk_id):
                ret.append(task)
        if len(ret) == 0:
            raise ValueError(f"Task not found for {dev=}, {seq_element=}, {chunk_id=}")
        return ret

    def _resolve_batch_dependency(self) -> None:
        filled = any([len(x) != 0 for x in self.microbatch_scheduled_tasks])
        assert not filled, "Batch Dependency already resolved"

        sequence, task_node_match_condition = self._get_microbatch_sequence()
        num_mb = self.sys_config.num_microbatches
        for mb in range(num_mb):
            for i, seq_ele in enumerate(sequence):
                dev = seq_ele[0]
                chunk_id = seq_ele[2] if len(seq_ele) == 3 else 0
                cur_tasks = self._get_tasknode_from_device(
                    mb, dev, chunk_id, task_node_match_condition, seq_ele
                )
                for j, cur_task in enumerate(cur_tasks):
                    if len(self.microbatch_scheduled_tasks[mb]) != 0:
                        cur_task.prev_microbatch_task = self.microbatch_scheduled_tasks[
                            mb
                        ][-1]
                        assert (
                            cur_task.prev_microbatch_task.microbatch_id
                            == cur_task.microbatch_id
                        )
                        self.microbatch_scheduled_tasks[mb][
                            -1
                        ].next_microbatch_task = cur_task

                    self.microbatch_scheduled_tasks[mb].append(cur_task)

    def _get_microbatch_sequence(
        self,
    ) -> Tuple[List[Tuple[int, str]], Callable[[SubTaskNode, Tuple], bool]]:
        raise NotImplementedError("Microbatch sequence not implemented")

    def is_send_to_next_rank(self, prev_task: SubTaskNode, cur_task: SubTaskNode):
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

    def _get_execution_time(self, task: SubTaskNode) -> int:
        ret = None
        task_type = task.task_type
        chunk = task.chunk_id
        dev = task.device_id
        if task_type == "F":
            ret = self.sys_config.T_F[chunk][dev]
        elif task_type == "B":
            ret = self.sys_config.T_B[chunk][dev]
        else:
            # W block
            assert self.sys_config.T_W[chunk][dev] > 0
            ret = self.sys_config.T_W[chunk][dev]

        return ret // self.num_subparts * (task.subpart_end - task.subpart_start)

    def solve_dependencies(self):
        bw_delay_model = BandwidthDelayModel(self.sys_config)

        # start of scheduling
        cur_dev_idx_list = [0] * self.sys_config.num_devices
        first_task = self.device_scheduled_tasks[0][0]
        first_task.start_time = 0
        first_task.completion_time = self._get_execution_time(first_task)
        cur_dev_idx_list[0] += 1

        while not all(
            idx == len(self.device_scheduled_tasks[dev])
            for dev, idx in enumerate(cur_dev_idx_list)
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
                    compute_time = self._get_execution_time(cur_task)

                    prev_microbatch_task_dev_id = (
                        prev_microbatch_task.device_id
                        if prev_microbatch_task is not None
                        else None
                    )
                    if prev_microbatch_task_dev_id is None:
                        lat_time = 0
                        bandwidth_time = 0
                    else:
                        lat_time = self.sys_config.T_alpha[prev_microbatch_task_dev_id][cur_dev_id]
                        bandwidth_time = self.sys_config.T_beta[prev_microbatch_task_dev_id][cur_dev_id]

                    if prev_device_task is not None:
                        cur_task.start_time = prev_device_task.completion_time
                    else:
                        cur_task.start_time = 0

                    if prev_microbatch_task is not None:
                        if prev_microbatch_task_dev_id != cur_dev_id:
                            cur_task_start_time_with_bw_delay = (
                                prev_microbatch_task.completion_time
                                + lat_time
                                + bw_delay_model.get_bandwidth_time_with_delay(
                                    prev_microbatch_task_dev_id,
                                    cur_dev_id,
                                    prev_microbatch_task.completion_time,
                                    bandwidth_time,
                                )
                            )
                            cur_task.start_time = max(
                                cur_task.start_time,
                                cur_task_start_time_with_bw_delay,
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
                subpart = (
                    f"({task.subpart_start}-{task.subpart_end}/{task.num_subparts})"
                    if task.subpart_start != 0 or task.subpart_end != task.num_subparts
                    else ""
                )
                chunk = f"c{task.chunk_id}" if self.has_multiple_chunks() else ""
                if verbose == 0:
                    print(f"-> {task.task_type} {task.microbatch_id} {chunk} {subpart}", end="")
                elif verbose == 1:
                    print(
                        f"-> {task.task_type} {task.microbatch_id} {chunk} {subpart} ({task.start_time}, {task.completion_time})",
                        end="",
                    )

            print("\n")

    def get_pipeline_execution_order(self) -> List[Tuple[int, int]]:
        """return a list of tuples (device_id, chunk_id)"""
        return [(dev, 0) for dev in range(self.sys_config.num_devices)]

    def get_device_scheduled_tasks(self) -> List[List[SubTaskNode]]:
        return self.device_scheduled_tasks


class DynZBUDSubPipeline(SubPipeline):
    def __init__(self, sys_config: SystemConfig, num_subparts: int = 1) -> None:
        super().__init__(sys_config, num_subparts)
        self.scheduler = DynZBUDSubScheduler(sys_config, num_subparts)

    def pipeline_name(self):
        return "DynZBUDSub"

    def schedule(self) -> None:
        self.scheduler.schedule()
        schedule = self.scheduler.get_schedule()
        num_dev = self.sys_config.num_devices

        for dev in range(num_dev):
            for i, subblock in enumerate(schedule[dev]):
                self.device_scheduled_tasks[dev].append(
                    SubTaskNode(
                        task_type=subblock.task_type,
                        device_id=dev,
                        microbatch_id=subblock.mb_id,
                        prev_device_task=self.device_scheduled_tasks[dev][-1]
                        if i != 0
                        else None,
                        prev_microbatch_task=None,
                        subpart_start=subblock.subpart_start,
                        subpart_end=subblock.subpart_end,
                        num_subparts=subblock.num_subparts,
                        chunk_id=subblock.chunk_id,
                        next_microbatch_task=None,
                    )
                )
        self._resolve_batch_dependency()

    def _get_microbatch_sequence(
        self,
    ) -> Tuple[List[Tuple[int, str]], Callable[[SubTaskNode, int, Tuple, int], bool]]:
        num_dev = self.sys_config.num_devices

        sequence = [(dev, "F") for dev in range(num_dev)] + [
            (dev, "B") for dev in reversed(range(num_dev))
        ]

        def condition(
            task: SubTaskNode, mb: int, seq_element: Tuple[int, str], chunk_id: int
        ):
            return (
                task.device_id == seq_element[0]
                and task.task_type == seq_element[1]
                and task.microbatch_id == mb
                and task.chunk_id == chunk_id
            )

        return sequence, condition

    def get_pipeline_first_stage_rank(self):
        return 0

    def get_pipeline_last_stage_rank(self):
        return self.sys_config.num_devices - 1
