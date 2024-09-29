from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Tuple

from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
    Pipeline,
    TaskNode,
    get_default_static_schedule,
)


@dataclass
class TaskEvent:
    pass


class CommEventType(Enum):
    POST_SEND_NEXT = auto()
    POST_RECV_NEXT = auto()
    POST_SEND_PREV = auto()
    POST_RECV_PREV = auto()
    WAIT_SEND_NEXT = auto()
    WAIT_RECV_NEXT = auto()
    WAIT_SEND_PREV = auto()
    WAIT_RECV_PREV = auto()


@dataclass
class CommEvent(TaskEvent):
    type: CommEventType
    src_dev_id: int
    dst_dev_id: int
    task_type: str
    chunk_id: int
    mb_id: int


@dataclass
class ComputeTaskDesc:
    type: str
    dev_id: int
    mb_id: int
    chunk_id: int


class ComputeTask:
    def __init__(self, task_desc: ComputeTaskDesc, start_time: int, end_time: int):
        self.task_desc = task_desc
        self.start_time = start_time
        self.end_time = end_time
        self.pre_events: List[TaskEvent] = []
        self.post_events: List[TaskEvent] = []


class ExecutionPlanner:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.execution_plan: List[List[ComputeTask]] = [
            [] for _ in range(len(pipeline.device_scheduled_tasks))
        ]

    def generate_execution_plan(self):
        device_task_lists = self.pipeline.device_scheduled_tasks
        pp_size = len(device_task_lists)

        for dev_id, dev_task_list in enumerate(device_task_lists):
            # compute tasks
            tasknode_to_computetask: Dict[TaskNode, ComputeTask] = {}
            for task in dev_task_list:
                chunk_id = getattr(task, "chunk_id", 0)

                self.execution_plan[dev_id].append(
                    ComputeTask(
                        task_desc=ComputeTaskDesc(
                            type=task.task_type,
                            dev_id=dev_id,
                            mb_id=task.microbatch_id,
                            chunk_id=chunk_id,
                        ),
                        start_time=task.start_time,
                        end_time=task.completion_time,
                    )
                )
                tasknode_to_computetask[task] = self.execution_plan[dev_id][-1]

            # (prev/next, cur task)
            recv_prev_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            recv_next_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            send_prev_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            send_next_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            next_rank = (dev_id + 1) % pp_size
            prev_rank = (dev_id - 1 + pp_size) % pp_size
            for dev_task in dev_task_list:
                prev_mb_task = dev_task.prev_microbatch_task
                next_mb_task = dev_task.next_microbatch_task
                if prev_mb_task is not None and prev_mb_task.device_id != dev_id:
                    assert prev_mb_task.device_id in [prev_rank, next_rank]
                    if self.pipeline.is_send_to_next_rank(prev_mb_task, dev_task) > 0:
                        assert prev_mb_task.device_id == prev_rank
                        recv_prev_dev_tasks.append((prev_mb_task, dev_task))
                    elif self.pipeline.is_send_to_next_rank(prev_mb_task, dev_task) < 0:
                        assert prev_mb_task.device_id == next_rank
                        recv_next_dev_tasks.append((prev_mb_task, dev_task))
                if next_mb_task is not None and next_mb_task.device_id != dev_id:
                    assert next_mb_task.device_id in [prev_rank, next_rank]
                    if self.pipeline.is_send_to_next_rank(dev_task, next_mb_task) > 0:
                        assert next_mb_task.device_id == next_rank
                        send_next_dev_tasks.append((next_mb_task, dev_task))
                    elif self.pipeline.is_send_to_next_rank(dev_task, next_mb_task) < 0:
                        assert next_mb_task.device_id == prev_rank
                        send_prev_dev_tasks.append((next_mb_task, dev_task))

            # sort the tasks by send time, to avoid deadlock in each of four channels
            # send prev/next lists are already sorted by the task start time
            recv_prev_dev_tasks.sort(key=lambda x: x[0].completion_time)
            recv_next_dev_tasks.sort(key=lambda x: x[0].completion_time)

            # insert sends
            for send_task, cur_task in send_prev_dev_tasks:
                compute_task = tasknode_to_computetask[cur_task]
                assert send_task.device_id == prev_rank
                assert send_task.task_type == cur_task.task_type
                assert send_task.microbatch_id == cur_task.microbatch_id

                compute_task.post_events.append(
                    CommEvent(
                        type=CommEventType.POST_SEND_PREV,
                        src_dev_id=dev_id,
                        dst_dev_id=prev_rank,
                        chunk_id=tasknode_to_computetask[cur_task].task_desc.chunk_id,
                        task_type=cur_task.task_type,
                        mb_id=cur_task.microbatch_id,
                    )
                )

            for send_task, cur_task in send_next_dev_tasks:
                compute_task = tasknode_to_computetask[cur_task]
                assert send_task.device_id == next_rank
                assert send_task.task_type == cur_task.task_type
                assert send_task.microbatch_id == cur_task.microbatch_id

                compute_task.post_events.append(
                    CommEvent(
                        type=CommEventType.POST_SEND_NEXT,
                        src_dev_id=dev_id,
                        dst_dev_id=next_rank,
                        task_type=cur_task.task_type,
                        chunk_id=tasknode_to_computetask[cur_task].task_desc.chunk_id,
                        mb_id=cur_task.microbatch_id,
                    )
                )

            # insert recvs(sorted) before the start of corresponding send
            for recv_task, cur_task in recv_prev_dev_tasks:
                task_to_insert_post = cur_task
                while (
                    task_to_insert_post.start_time > recv_task.completion_time
                    and task_to_insert_post.prev_device_task is not None
                ):
                    # ensure the post recv is inserted before the send if possible
                    task_to_insert_post = task_to_insert_post.prev_device_task
                compute_task = tasknode_to_computetask[task_to_insert_post]
                compute_task.pre_events.append(
                    CommEvent(
                        type=CommEventType.POST_RECV_PREV,
                        src_dev_id=prev_rank,
                        dst_dev_id=dev_id,
                        task_type=cur_task.task_type,
                        chunk_id=tasknode_to_computetask[cur_task].task_desc.chunk_id,
                        mb_id=cur_task.microbatch_id,
                    )
                )

            for recv_task, cur_task in recv_next_dev_tasks:
                task_to_insert_post = cur_task
                while (
                    task_to_insert_post.start_time > recv_task.completion_time
                    and task_to_insert_post.prev_device_task is not None
                ):
                    # ensure the post recv is inserted before the send if possible
                    task_to_insert_post = task_to_insert_post.prev_device_task
                compute_task = tasknode_to_computetask[task_to_insert_post]
                compute_task.pre_events.append(
                    CommEvent(
                        type=CommEventType.POST_RECV_NEXT,
                        src_dev_id=next_rank,
                        dst_dev_id=dev_id,
                        task_type=cur_task.task_type,
                        chunk_id=tasknode_to_computetask[cur_task].task_desc.chunk_id,
                        mb_id=cur_task.microbatch_id,
                    )
                )

            # insert wait recvs
            for recv_task, cur_task in recv_prev_dev_tasks:
                compute_task = tasknode_to_computetask[cur_task]
                compute_task.pre_events.append(
                    CommEvent(
                        type=CommEventType.WAIT_RECV_PREV,
                        src_dev_id=recv_task.device_id,
                        dst_dev_id=dev_id,
                        task_type=cur_task.task_type,
                        chunk_id=tasknode_to_computetask[cur_task].task_desc.chunk_id,
                        mb_id=cur_task.microbatch_id,
                    )
                )

            for recv_task, cur_task in recv_next_dev_tasks:
                compute_task = tasknode_to_computetask[cur_task]
                compute_task.pre_events.append(
                    CommEvent(
                        type=CommEventType.WAIT_RECV_NEXT,
                        src_dev_id=recv_task.device_id,
                        dst_dev_id=dev_id,
                        task_type=cur_task.task_type,
                        chunk_id=tasknode_to_computetask[cur_task].task_desc.chunk_id,
                        mb_id=cur_task.microbatch_id,
                    )
                )

            # no need to insert wait sends.

    def print_execution_plan(self) -> str:
        output = []
        output.append("Execution Plan:")
        output.append("=" * 120)
        for dev_id, dev_task_list in enumerate(self.execution_plan):
            output.append("")
            output.append(f"Device {dev_id}:")
            for task in dev_task_list:
                output.append("-" * 60)
                output.append("  Prev Events:")
                for event in task.pre_events:
                    output.append(f"    {event}")

                output.append("  Compute Task:")
                output.append(
                    f"    {task.task_desc.type} mb{task.task_desc.mb_id} chunk{task.task_desc.chunk_id} start{task.start_time} end{task.end_time}"
                )

                output.append("  Post Events:")
                for event in task.post_events:
                    output.append(f"    {event}")
            output.append("=" * 120)
            output.append("")

        return "\n".join(output)
