from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.pipeline import (
    Pipeline,
    TaskNode,
    get_default_static_schedule,
)
from megatron.core.pipeline_parallel.cdc_scheduler.pp_generator.subpipeline import SubPipeline


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
    LOCAL_COPY = auto()


@dataclass
class CommEvent(TaskEvent):
    type: CommEventType
    src_dev_id: int
    dst_dev_id: int
    task_type: str
    chunk_id: int
    mb_id: int
    prev_task_chunk_id: int = -1  # only used for local copy

    def __hash__(self) -> int:
        return hash(
            (
                self.type,
                self.src_dev_id,
                self.dst_dev_id,
                self.task_type,
                self.chunk_id,
                self.mb_id,
                self.prev_task_chunk_id,
            )
        )
    
    def __str__(self):
        return f'{self.__class__.__name__}.{self.type.name} {self.src_dev_id} -> {self.dst_dev_id} {self.task_type} c{self.chunk_id} mb{self.mb_id}'


@dataclass
class ComputeTaskDesc:
    type: str
    dev_id: int
    mb_id: int
    chunk_id: int
    subpart_start: int = 0
    subpart_end: int = 1
    num_subparts: int = 1


class ComputeTask:
    def __init__(self, task_desc: ComputeTaskDesc, start_time: int, end_time: int):
        self.task_desc = task_desc
        self.start_time = start_time
        self.end_time = end_time
        self.pre_events: List[TaskEvent] = []
        self.post_events: List[TaskEvent] = []
        self.send_event: CommEvent = None  # send the output of this task
        self.prev_mb_task: ComputeTask = None  # the previous task in the mb
        self.recv_event: CommEvent = None  # recv the input of this task
        self.next_mb_task: ComputeTask = None  # the next task in the mb
        self.wait_recv_event: CommEvent = None  # wait for the input of this task


class DependencyNodeType(Enum):
    PC = auto()  # post compute
    C = auto()  # compute
    PSN = auto()  # post send next
    SN = auto()  # send next
    PSP = auto()  # post send prev
    SP = auto()  # send prev
    PRN = auto()  # post recv next
    RN = auto()  # recv next
    PRP = auto()  # post recv prev
    RP = auto()  # recv prev
    PWRP = auto()  # post wait recv prev
    PWRN = auto()  # post wait recv next


@dataclass
class DependencyNode:
    node_type: DependencyNodeType
    event: Optional[CommEvent] = None
    task: Optional[ComputeTask] = None
    dependencies: List["DependencyNode"] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.node_type, self.event, self.task))


class DependencyAnalyzer:
    def __init__(self, pipeline: Pipeline) -> None:
        self.G = nx.DiGraph()
        self.pipeline = pipeline
        self.num_dev = pipeline.sys_config.num_devices

        # from execution planner
        self.comm_to_comm: Dict[CommEvent, CommEvent] = {}
        self.execution_plan: List[List[ComputeTask]] = []
        self.recv_to_wait: Dict[CommEvent, CommEvent] = {}

        self.event_to_compute: Dict[CommEvent, ComputeTask] = {}
        self.event_to_exec_node: Dict[CommEvent, DependencyNode] = {}

        # self.compute_to_events: Dict[ComputeTask, List[TaskEvent]] = {}
        self.stream_cpu: List[List[DependencyNode]] = [[] for _ in range(self.num_dev)]
        self.stream_gpu_compute: List[List[DependencyNode]] = [
            [] for _ in range(self.num_dev)
        ]
        self.stream_gpu_send_prev: List[List[DependencyNode]] = [
            [] for _ in range(self.num_dev)
        ]
        self.stream_gpu_send_next: List[List[DependencyNode]] = [
            [] for _ in range(self.num_dev)
        ]
        self.stream_gpu_recv_prev: List[List[DependencyNode]] = [
            [] for _ in range(self.num_dev)
        ]
        self.stream_gpu_recv_next: List[List[DependencyNode]] = [
            [] for _ in range(self.num_dev)
        ]

    def set_execution_plan_and_mappings(
        self,
        execution_plan: List[List[ComputeTask]],
        comm_to_comm: Dict[CommEvent, CommEvent],
        recv_to_wait: Dict[CommEvent, CommEvent],
    ):
        self.execution_plan = execution_plan
        self.comm_to_comm = comm_to_comm
        self.recv_to_wait = recv_to_wait

    def _construct_dependencies(self):
        stream_lists = [
            *self.stream_cpu,
            *self.stream_gpu_compute,
            *self.stream_gpu_send_prev,
            *self.stream_gpu_send_next,
            *self.stream_gpu_recv_prev,
            *self.stream_gpu_recv_next,
        ]
        if any([len(stream_list) > 0 for stream_list in stream_lists]):
            # already constructed
            return

        for dev_id, dev_task_list in enumerate(self.execution_plan):
            for task in dev_task_list:
                for event in task.pre_events:
                    self.event_to_compute[event] = task
                for event in task.post_events:
                    self.event_to_compute[event] = task

            def _process_events(event: TaskEvent):
                # set up launching dependencies
                assert isinstance(event, CommEvent)
                if event.type == CommEventType.POST_SEND_NEXT:
                    psn_node = DependencyNode(
                        DependencyNodeType.PSN,
                        event=event,
                    )
                    sn_node = DependencyNode(
                        DependencyNodeType.SN,
                        event=event,
                    )
                    self.event_to_exec_node[event] = sn_node
                    self.stream_cpu[dev_id].append(psn_node)
                    psn_node.dependencies.append(sn_node)
                    self.stream_gpu_send_next[dev_id].append(sn_node)
                elif event.type == CommEventType.POST_SEND_PREV:
                    psp_node = DependencyNode(
                        DependencyNodeType.PSP,
                        event=event,
                    )
                    sp_node = DependencyNode(
                        DependencyNodeType.SP,
                        event=event,
                    )
                    self.event_to_exec_node[event] = sp_node
                    self.stream_cpu[dev_id].append(psp_node)
                    psp_node.dependencies.append(sp_node)
                    self.stream_gpu_send_prev[dev_id].append(sp_node)
                elif event.type == CommEventType.POST_RECV_PREV:
                    prp_node = DependencyNode(
                        DependencyNodeType.PRP,
                        event=event,
                    )
                    rp_node = DependencyNode(
                        DependencyNodeType.RP,
                        event=event,
                    )
                    self.event_to_exec_node[event] = rp_node
                    self.stream_cpu[dev_id].append(prp_node)
                    prp_node.dependencies.append(rp_node)
                    self.stream_gpu_recv_prev[dev_id].append(rp_node)
                elif event.type == CommEventType.POST_RECV_NEXT:
                    prn_node = DependencyNode(
                        DependencyNodeType.PRN,
                        event=event,
                    )
                    rn_node = DependencyNode(
                        DependencyNodeType.RN,
                        event=event,
                    )
                    self.event_to_exec_node[event] = rn_node
                    self.stream_cpu[dev_id].append(prn_node)
                    prn_node.dependencies.append(rn_node)
                    self.stream_gpu_recv_next[dev_id].append(rn_node)
                elif event.type == CommEventType.WAIT_RECV_PREV:
                    pwrp_node = DependencyNode(
                        DependencyNodeType.PWRP,
                        event=event,
                    )
                    # todo: pwrp is not exec node, put here only to simplify the code
                    self.event_to_exec_node[event] = pwrp_node
                    self.stream_cpu[dev_id].append(pwrp_node)
                elif event.type == CommEventType.WAIT_RECV_NEXT:
                    pwrn_node = DependencyNode(
                        DependencyNodeType.PWRN,
                        event=event,
                    )
                    # todo: pwrn is not exec node, put here only to simplify the code
                    self.event_to_exec_node[event] = pwrn_node
                    self.stream_cpu[dev_id].append(pwrn_node)
                elif event.type == CommEventType.LOCAL_COPY:
                    pass
                else:
                    raise ValueError(f"Unknown event type {event.type}")

            for task in dev_task_list:
                for event in task.pre_events:
                    _process_events(event)

                # process compute task
                pc_node = DependencyNode(
                    DependencyNodeType.PC,
                    task=task,
                )
                c_node = DependencyNode(
                    DependencyNodeType.C,
                    task=task,
                )
                self.stream_cpu[dev_id].append(pc_node)
                pc_node.dependencies.append(c_node)
                self.stream_gpu_compute[dev_id].append(c_node)

                for event in task.post_events:
                    _process_events(event)

        # nodes created, now connect the dependencies
        for dev_id in range(self.num_dev):
            # set up dependencies in each stream
            def _set_stream_dependencies(stream: List[List[DependencyNode]]):
                for idx in range(len(stream[dev_id]) - 1):
                    node = stream[dev_id][idx]
                    next_node_in_dev = stream[dev_id][idx + 1]
                    node.dependencies.append(next_node_in_dev)

            _set_stream_dependencies(self.stream_cpu)
            _set_stream_dependencies(self.stream_gpu_compute)
            _set_stream_dependencies(self.stream_gpu_send_prev)
            _set_stream_dependencies(self.stream_gpu_send_next)
            _set_stream_dependencies(self.stream_gpu_recv_prev)
            _set_stream_dependencies(self.stream_gpu_recv_next)

            # send <-> recv dependencies
            for node in self.stream_gpu_send_prev[dev_id]:
                recv_event = self.comm_to_comm[node.event]
                recv_node = self.event_to_exec_node[recv_event]
                # bidirectional deps to reflect the synchronization of send/recv
                node.dependencies.append(recv_node)
                recv_node.dependencies.append(node)

            for node in self.stream_gpu_send_next[dev_id]:
                recv_event = self.comm_to_comm[node.event]
                recv_node = self.event_to_exec_node[recv_event]
                # bidirectional deps to reflect the synchronization of send/recv
                node.dependencies.append(recv_node)
                recv_node.dependencies.append(node)

            # recv -> wait dependencies
            for node in (
                self.stream_gpu_recv_prev[dev_id] + self.stream_gpu_recv_next[dev_id]
            ):
                wait_event = self.recv_to_wait[node.event]
                wait_node = self.event_to_exec_node[wait_event]
                node.dependencies.append(wait_node)

    def construct_dependency_graph(self):
        # return if already constructed
        if len(self.G.nodes) > 0:
            return

        self._construct_dependencies()
        for dev_id in range(self.num_dev):
            for node in (
                self.stream_cpu[dev_id]
                + self.stream_gpu_compute[dev_id]
                + self.stream_gpu_send_prev[dev_id]
                + self.stream_gpu_send_next[dev_id]
                + self.stream_gpu_recv_prev[dev_id]
                + self.stream_gpu_recv_next[dev_id]
            ):
                self.G.add_node(node)
                for dep in node.dependencies:
                    self.G.add_edge(node, dep)

    def _custom_layout_by_stream(self):
        pos = {}
        stream_lists = [[] for _ in range(self.num_dev)]
        for dev_id in range(self.num_dev):
            stream_lists[dev_id] += self.stream_cpu[dev_id]
            stream_lists[dev_id] += self.stream_gpu_compute[dev_id]
            stream_lists[dev_id] += self.stream_gpu_send_prev[dev_id]
            stream_lists[dev_id] += self.stream_gpu_send_next[dev_id]
            stream_lists[dev_id] += self.stream_gpu_recv_prev[dev_id]
            stream_lists[dev_id] += self.stream_gpu_recv_next[dev_id]

        total_streams = sum([len(stream_list) for stream_list in stream_lists])

        y_spacing = 1.0 / (total_streams + 1)

        stream_idx = 0
        for stream_list in stream_lists:
            for dev_stream in stream_list:
                ordered_nodes = list(nx.topological_sort(self.G.subgraph(dev_stream)))
                x_spacing = (
                    1.0 / (len(ordered_nodes) + 1) if len(ordered_nodes) > 0 else 0.5
                )

                for idx, node in enumerate(ordered_nodes):
                    pos[node] = np.array(
                        [
                            0.1 + x_spacing * (idx + 1),  # x position with left margin
                            1.0 - y_spacing * (stream_idx + 1),  # y position
                        ]
                    )
                stream_idx += 1
        return pos

    def print_dependency_graph(self, save_path=".") -> None:
        self.construct_dependency_graph()
        cycles = [
            cycle
            for cycle in nx.simple_cycles(self.G)
            if not (len(cycle) == 2 and (cycle[1], cycle[0]) in self.G.edges())
        ]

        if cycles:
            print("\nFound cycles in dependency graph:")
            for i, cycle in enumerate(cycles):
                path = " -> ".join(
                    [
                        f"{node.node_type.name}"
                        + (
                            f"({node.task.task_desc.type}{node.task.task_desc.mb_id})"
                            if node.task
                            else f"({node.event.task_type}{node.event.mb_id})"
                            if node.event
                            else ""
                        )
                        for node in cycle
                    ]
                )
                print(f"Cycle {i+1}: {path}")

        viz_graph = nx.DiGraph()

        # Group nodes by stream
        stream_groups = {}
        for dev_id in range(self.pipeline.sys_config.num_devices):
            streams = {
                "CPU": self.stream_cpu[dev_id],
                "GPU_Compute": self.stream_gpu_compute[dev_id],
                "GPU_Send_Prev": self.stream_gpu_send_prev[dev_id],
                "GPU_Send_Next": self.stream_gpu_send_next[dev_id],
                "GPU_Recv_Prev": self.stream_gpu_recv_prev[dev_id],
                "GPU_Recv_Next": self.stream_gpu_recv_next[dev_id],
            }
            for stream_name, nodes in streams.items():
                key = f"Dev{dev_id}_{stream_name}"
                stream_groups[key] = nodes

        # Calculate y-positions
        y_positions = {stream: idx for idx, stream in enumerate(stream_groups.keys())}

        # Find cycles
        cycles = list(nx.simple_cycles(self.G))
        nodes_in_cycles = set()
        for cycle in cycles:
            nodes_in_cycles.update(cycle)

        # Create node positions
        pos = {}
        node_ids = {}

        for stream_name, nodes in stream_groups.items():
            ordered_nodes = list(nx.topological_sort(self.G.subgraph(nodes)))
            x_spacing = 1.0 / (len(ordered_nodes) + 1) if ordered_nodes else 0.5

            for idx, node in enumerate(ordered_nodes):
                x_pos = idx + 1
                y_pos = y_positions[stream_name] * 2  # Multiply by 2 for better spacing
                pos[node] = (x_pos, y_pos)

                node_id = f"node_{id(node)}"
                node_ids[node] = node_id

                label = ""
                if node.node_type in [DependencyNodeType.C, DependencyNodeType.PC]:
                    task = node.task
                    label = f"{node.node_type.name}\n{task.task_desc.type[:2]}{task.task_desc.mb_id}"
                else:
                    label = (
                        f"{node.node_type.name}\n{node.event.task_type[:2]}{node.event.mb_id}"
                        if node.event
                        else node.node_type.name
                    )

                fillcolor = "pink" if node in nodes_in_cycles else "lightblue"
                viz_graph.add_node(
                    node_id,
                    label=label,
                    shape="box",
                    style="filled",
                    fillcolor=fillcolor,
                )

        # Add edges
        for u, v in self.G.edges():
            is_bidirectional = (v, u) in self.G.edges()
            edge_color = "gray" if is_bidirectional else "black"
            viz_graph.add_edge(node_ids[u], node_ids[v], color=edge_color)

        # Convert to Graphviz and set positions
        A = nx.nx_agraph.to_agraph(viz_graph)

        # Set node positions
        for node, (x, y) in pos.items():
            n = A.get_node(node_ids[node])
            n.attr["pos"] = f"{x},{y}!"

        A.graph_attr.update({"splines": "ortho", "ranksep": "1.0", "nodesep": "0.5"})

        A.layout(prog="neato")
        A.draw(f"{save_path}/dependency_graph.svg")


class ExecutionPlanner:
    def __init__(self, pipeline: Pipeline, enable_dependency_analysis=False):
        self.pipeline = pipeline
        self.is_subpipeline = isinstance(pipeline, SubPipeline)
        if self.is_subpipeline:
            for dev in range(len(pipeline.device_scheduled_tasks)):
                assert all([task.num_subparts == pipeline.num_subparts for task in pipeline.device_scheduled_tasks[dev]])
        self.execution_plan: List[List[ComputeTask]] = [
            [] for _ in range(len(pipeline.device_scheduled_tasks))
        ]
        if enable_dependency_analysis:
            self.dependency_analyzer = DependencyAnalyzer(pipeline)
        else:
            self.dependency_analyzer = None

    def generate_execution_plan(self):
        # for dependency analysis
        comm_to_comm: Dict[CommEvent, CommEvent] = {}
        recv_to_wait: Dict[CommEvent, CommEvent] = {}
        #

        device_task_lists = self.pipeline.device_scheduled_tasks
        pp_size = len(device_task_lists)

        tasknode_to_computetask: Dict[TaskNode, ComputeTask] = {}
        for dev_id, dev_task_list in enumerate(device_task_lists):
            # compute tasks
            for task in dev_task_list:
                chunk_id = getattr(task, "chunk_id", 0)

                self.execution_plan[dev_id].append(
                    ComputeTask(
                        task_desc=ComputeTaskDesc(
                            type=task.task_type,
                            dev_id=dev_id,
                            mb_id=task.microbatch_id,
                            chunk_id=chunk_id,
                            subpart_start=task.subpart_start,
                            subpart_end=task.subpart_end,
                            num_subparts=task.num_subparts,
                        ),
                        start_time=task.start_time,
                        end_time=task.completion_time,
                    )
                )
                tasknode_to_computetask[task] = self.execution_plan[dev_id][-1]

                if (
                    task.prev_microbatch_task is not None
                    and task.prev_microbatch_task in tasknode_to_computetask
                ):
                    tasknode_to_computetask[
                        task
                    ].prev_mb_task = tasknode_to_computetask[task.prev_microbatch_task]
                    tasknode_to_computetask[
                        task.prev_microbatch_task
                    ].next_mb_task = tasknode_to_computetask[task]

                if (
                    task.next_microbatch_task is not None
                    and task.next_microbatch_task in tasknode_to_computetask
                ):
                    tasknode_to_computetask[
                        task
                    ].next_mb_task = tasknode_to_computetask[task.next_microbatch_task]
                    tasknode_to_computetask[
                        task.next_microbatch_task
                    ].prev_mb_task = tasknode_to_computetask[task]

        for dev_id, dev_task_list in enumerate(device_task_lists):
            # (prev/next, cur task)
            recv_prev_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            recv_next_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            send_prev_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            send_next_dev_tasks: List[Tuple[TaskNode, TaskNode]] = []
            local_copy_tasks: List[Tuple[TaskNode, TaskNode]] = []

            next_rank = (dev_id + 1) % pp_size
            prev_rank = (dev_id - 1 + pp_size) % pp_size
            for dev_task in dev_task_list:
                prev_mb_task = dev_task.prev_microbatch_task
                next_mb_task = dev_task.next_microbatch_task
                # print(f'dev {dev_id}: found next_mb_task {next_mb_task} and prev_mb_task {prev_mb_task}')
                if prev_mb_task is not None and prev_mb_task.device_id == dev_id and prev_mb_task.subpart_end == prev_mb_task.num_subparts:
                    # local copy
                    # F chunk 0 -> F chunk 1 or B chunk 1 -> B chunk 0
                    if prev_mb_task.task_type == dev_task.task_type:
                        local_copy_tasks.append((prev_mb_task, dev_task))

                if prev_mb_task is not None and prev_mb_task.device_id != dev_id and prev_mb_task.subpart_end == prev_mb_task.num_subparts:
                    assert prev_mb_task.device_id in [prev_rank, next_rank]
                    if self.pipeline.is_send_to_next_rank(prev_mb_task, dev_task) > 0:
                        assert prev_mb_task.device_id == prev_rank
                        recv_prev_dev_tasks.append((prev_mb_task, dev_task))
                    elif self.pipeline.is_send_to_next_rank(prev_mb_task, dev_task) < 0:
                        assert prev_mb_task.device_id == next_rank
                        recv_next_dev_tasks.append((prev_mb_task, dev_task))
                if next_mb_task is not None and next_mb_task.device_id != dev_id and dev_task.subpart_end == dev_task.num_subparts:
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

            # print(f"Device {dev_id} recv_prev_dev_tasks: {recv_prev_dev_tasks}")
            # print(f"Device {dev_id} recv_next_dev_tasks: {recv_next_dev_tasks}")
            # print(f"Device {dev_id} send_prev_dev_tasks: {send_prev_dev_tasks}")
            # print(f"Device {dev_id} send_next_dev_tasks: {send_next_dev_tasks}")

            # insert local copies
            for prev_mb_task, cur_task in local_copy_tasks:
                compute_task = tasknode_to_computetask[cur_task]
                assert prev_mb_task.device_id == dev_id
                assert prev_mb_task.task_type == cur_task.task_type
                assert prev_mb_task.microbatch_id == cur_task.microbatch_id
                assert (
                    tasknode_to_computetask[prev_mb_task].task_desc.chunk_id
                    != tasknode_to_computetask[cur_task].task_desc.chunk_id
                )
                compute_task.pre_events.append(
                    CommEvent(
                        type=CommEventType.LOCAL_COPY,
                        src_dev_id=dev_id,
                        dst_dev_id=dev_id,
                        task_type=cur_task.task_type,
                        chunk_id=tasknode_to_computetask[cur_task].task_desc.chunk_id,
                        mb_id=cur_task.microbatch_id,
                        prev_task_chunk_id=tasknode_to_computetask[
                            prev_mb_task
                        ].task_desc.chunk_id,
                    )
                )

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
                compute_task.send_event = compute_task.post_events[-1]

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
                compute_task.send_event = compute_task.post_events[-1]

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
                tasknode_to_computetask[cur_task].recv_event = compute_task.pre_events[
                    -1
                ]

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
                tasknode_to_computetask[cur_task].recv_event = compute_task.pre_events[
                    -1
                ]

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
                compute_task.wait_recv_event = compute_task.pre_events[-1]

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
                compute_task.wait_recv_event = compute_task.pre_events[-1]

            # no need to insert wait sends.

        # set up mappings
        for dev_id, dev_task_list in enumerate(self.execution_plan):
            for task in dev_task_list:
                if task.send_event is not None:
                    assert (
                        task.next_mb_task is not None
                        and task.next_mb_task.recv_event is not None
                    )
                    comm_to_comm[task.send_event] = task.next_mb_task.recv_event
                    comm_to_comm[task.next_mb_task.recv_event] = task.send_event
                if task.recv_event is not None:
                    assert (
                        task.prev_mb_task is not None
                        and task.prev_mb_task.send_event is not None
                    )
                    comm_to_comm[task.recv_event] = task.prev_mb_task.send_event
                    comm_to_comm[task.prev_mb_task.send_event] = task.recv_event
                    assert task.wait_recv_event is not None
                    recv_to_wait[task.recv_event] = task.wait_recv_event

        if self.dependency_analyzer:
            self.dependency_analyzer.set_execution_plan_and_mappings(
                self.execution_plan, comm_to_comm, recv_to_wait
            )

    def print_dependency_graph(self, save_path=".") -> None:
        if any(
            [
                len(self.execution_plan[dev_id]) == 0
                for dev_id in range(len(self.execution_plan))
            ]
        ):
            self.generate_execution_plan()
        if self.dependency_analyzer:
            self.dependency_analyzer.print_dependency_graph(save_path=save_path)

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
                subpart = (
                    f"({task.task_desc.subpart_start}-{task.task_desc.subpart_end}/{task.task_desc.num_subparts})"
                    if task.task_desc.subpart_start != 0 or task.task_desc.subpart_end != task.task_desc.num_subparts
                    else ""
                )
                output.append(
                    f"    {task.task_desc.type} mb{task.task_desc.mb_id} chunk{task.task_desc.chunk_id} {subpart} start{task.start_time} end{task.end_time}"
                )

                output.append("  Post Events:")
                for event in task.post_events:
                    output.append(f"    {event}")
            output.append("=" * 120)
            output.append("")

        return "\n".join(output)


if "__main__" == __name__:
    pipeline = get_default_static_schedule(
        pipeline_name="Interleaved1F1B", num_devices=2, num_microbatches=8
    )
    planner = ExecutionPlanner(pipeline)
    planner.generate_execution_plan()
    # planner.print_dependency_graph()
    print(planner.print_execution_plan())
