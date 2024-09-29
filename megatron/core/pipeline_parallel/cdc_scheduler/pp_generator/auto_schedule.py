import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple
from .pipeline_config import SystemConfig
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum, value
import pulp
import gurobipy as gp
import scipy.sparse as sp

gurobi_options = {
    "WLSACCESSID": "813ae627-c773-452d-8bd5-2cbe1225118c",
    "WLSSECRET": "77e9fd95-2d5a-479f-977a-1c3cc2b28567",
    "LICENSEID": 2531862,
    "THREADS": psutil.cpu_count(logical=False),
}


class DependencyGraph:
    def __init__(self, system_cfg: SystemConfig) -> None:
        self.system_cfg = system_cfg
        self.num_dev = system_cfg.num_devices
        self.num_mb = system_cfg.num_microbatches
        self.num_chunk = system_cfg.num_chunks
        self.nnodes = self.num_dev * self.num_mb * self.num_chunk * 3

        self.inherent_direct_dep: Optional[np.array] = None
        self.inherent_dep: Optional[sp.csr_matrix] = None
        self.prob: Optional[LpProblem] = None
        self.prob_F: Optional[Dict[int, LpVariable]] = None

        assert all([type(Tf) is int for Tf in self.system_cfg.T_F])
        assert all([type(Tb) is int for Tb in self.system_cfg.T_B])
        assert all([type(Tw) is int for Tw in self.system_cfg.T_W])
        # assert isinstance(self.system_cfg.T_C, np.ndarray) and np.issubdtype(self.system_cfg.T_C.dtype, np.integer)
        assert all(
            [
                f + b + w == 0
                for f, b, w in zip(
                    self.system_cfg.M_F, self.system_cfg.M_B, self.system_cfg.M_W
                )
            ]
        )

    # ID: [dev][mb][task_type]
    def _get_id(self, dev: int, mb: int, task_type: int) -> int:
        raise NotImplementedError

    def _get_dev(self, id: int) -> int:
        raise NotImplementedError

    def _get_mb(self, id: int) -> int:
        raise NotImplementedError

    def _get_task_type(self, id: int) -> int:
        raise NotImplementedError

    def _get_task_time_cost(self, id: int) -> int:
        raise NotImplementedError

    def _get_comm_cost(self, src_dev_id, dst_dev_id) -> int:
        raise NotImplementedError

    def _get_mem_cost(self, id: int) -> int:
        raise NotImplementedError

    def _init_direct_inherent_dependency(self) -> None:
        raise NotImplementedError

    def _init_inherent_dependency(self) -> None:
        # propagated from the direct inherent dependency
        # e.g. the F block of mb 1 mush be finished before the F block of mb 2,3,.. on the same device
        assert self.inherent_dep is None
        assert self.inherent_direct_dep is not None

        adj = sp.lil_matrix((self.nnodes, self.nnodes), dtype=int)
        for i in range(self.nnodes):
            for j in self.inherent_direct_dep[i]:
                adj[j, i] = 1
        adj = adj.tocsr()
        while True:
            new_adj = adj.dot(adj) + adj
            # set nnz to 1
            new_adj.data[:] = 1
            if (adj != new_adj).nnz == 0:
                break
            adj = new_adj
        self.inherent_dep = adj

    def _schedulable_on_dev(self, id_i: int, id_j: int) -> bool:
        # only effective for the same device
        return (
            id_i != id_j
            and self._get_dev(id_i) == self._get_dev(id_j)
            and not self.inherent_dep[id_i, id_j]
            and not self.inherent_dep[id_j, id_i]
        )

    def build_ilp(self) -> None:
        raise NotImplementedError

    def solve_ilp(self, verbose=True, warm_start=False, time_limit=200) -> None:
        # assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True)
        # solver = pulp.PULP_CBC_CMD(
        #     mip=True,
        #     msg=verbose,
        #     warmStart=warm_start,
        #     gapRel=1e-8,
        #     threads=mp.cpu_count() // 2,
        #     timeLimit=time_limit,
        # )
        try:
            with gp.Env(params=gurobi_options) as env:
                solver = pulp.GUROBI(
                    mip=True,
                    msg=verbose,
                    warmStart=warm_start,
                    gapRel=1e-8,
                    timeLimit=time_limit,
                    env=env,
                )
                status = self.prob.solve(solver)
                print(f"Status: {LpStatus[status]}")
        except Exception as e:
            print(e)
        finally:
            if solver is not None:
                solver.close()

    def get_schedule(self) -> List[List[Tuple[int, int, str, int]]]:
        raise NotImplementedError

    def get_lp_status(self) -> int:
        # if prob is None, return -4
        if self.prob is None:
            return -4
        return self.prob.status

    def get_objective_value(self) -> int:
        if self.prob is None:
            return None
        return pulp.value(self.prob.objective)


class UnidirectionalZBDependencyGraph(DependencyGraph):
    def __init__(self, system_cfg: SystemConfig) -> None:
        super().__init__(system_cfg)

        self._init_direct_inherent_dependency()
        self._init_inherent_dependency()

    # ID: [dev][mb][task_type]
    def _get_id(self, dev: int, mb: int, task_type: int) -> int:
        return dev * self.num_mb * 3 + mb * 3 + task_type

    def _get_dev(self, id: int) -> int:
        return id // (self.num_mb * 3)

    def _get_mb(self, id: int) -> int:
        return (id // 3) % self.num_mb

    def _get_task_type(self, id: int) -> int:
        return id % 3

    def _get_task_time_cost(self, id: int) -> int:
        task_type = self._get_task_type(id)
        dev = self._get_dev(id)
        return [
            self.system_cfg.T_F[dev],
            self.system_cfg.T_B[dev],
            self.system_cfg.T_W[dev],
        ][task_type]

    def _get_comm_cost(self, src_dev_id, dst_dev_id) -> int:
        return self.system_cfg.T_C[src_dev_id][dst_dev_id]

    def _get_mem_cost(self, id: int) -> int:
        task_type = self._get_task_type(id)
        dev = self._get_dev(id)
        return [
            self.system_cfg.M_F[dev],
            self.system_cfg.M_B[dev],
            self.system_cfg.M_W[dev],
        ][task_type]

    def _init_direct_inherent_dependency(self) -> None:
        # inherent dependency is the type of dependency that is not affected by the scheduling
        assert self.inherent_direct_dep is None

        parents = []  # parents[id] is a set of direct dependencies of id
        for dev in range(self.num_dev):
            for mb in range(self.num_mb):
                for task_type in range(3):
                    p = set()
                    if task_type == 0:
                        # F block
                        if mb > 0:
                            # prev mb from same device
                            p.add(self._get_id(dev, mb - 1, 0))
                        if dev > 0:
                            # same mb from prev device
                            p.add(self._get_id(dev - 1, mb, 0))
                    elif task_type == 1:
                        # B block
                        if dev == self.num_dev - 1:
                            # last device from corresponding F block
                            p.add(self._get_id(dev, mb, 0))
                        else:
                            # same mb for next device
                            p.add(self._get_id(dev + 1, mb, 1))
                        if mb > 0:
                            # prev mb from same device
                            p.add(self._get_id(dev, mb - 1, 1))
                    elif task_type == 2:
                        # W block
                        # corresponding B block
                        p.add(self._get_id(dev, mb, 1))
                        if mb > 0:
                            # prev mb from same device
                            # not necessary, but shrink the search space
                            p.add(self._get_id(dev, mb - 1, 2))
                    else:
                        raise ValueError("Invalid task type")
                    parents.append(p)
        self.inherent_direct_dep = parents

    def build_ilp(self) -> None:
        prob = LpProblem("AutoSchedule", LpMinimize)

        # dependency order graph
        # P[i][j] = 1 if i is scheduled before j
        # i and j are on the same device
        # schedulable dep as lp variables
        P: Dict[Tuple, LpVariable] = {}
        for i in range(self.nnodes):
            for j in range(i):
                if self._schedulable_on_dev(i, j):
                    P[(i, j)] = LpVariable(f"P_{i}_{j}", 0, 1, cat="Binary")
                    P[(j, i)] = 1 - P[(i, j)]

        # completion time
        F: Dict[int, LpVariable] = LpVariable.dicts(
            "F", (range(self.nnodes),), None, None, cat="Continuous"
        )

        inf = (
            (
                max(self.system_cfg.T_F)
                + max(self.system_cfg.T_B)
                + max(self.system_cfg.T_W)
                + np.max(self.system_cfg.T_C) * 3
            )
            * self.num_dev
            * self.num_mb
        )

        # anchor the first task of the 0th device
        first_task = self._get_id(0, 0, 0)
        prob += F[first_task] >= self._get_task_time_cost(first_task)

        M_limits = []

        for i in range(self.nnodes):
            mem_cost = []
            for prev in range(self.nnodes):
                if i == prev:
                    continue
                if prev in self.inherent_direct_dep[i]:
                    # direct dependency, cross device or same device
                    prob += F[i] >= F[prev] + self._get_task_time_cost(i) + (
                        self._get_comm_cost(self._get_dev(prev), self._get_dev(i))
                    )

                if self._get_dev(i) == self._get_dev(prev):
                    if self.inherent_dep[i, prev]:
                        pass
                    elif self.inherent_dep[prev, i]:
                        mem_cost.append(self._get_mem_cost(prev))
                    else:
                        # schedulable dependency
                        prob += (
                            F[i]
                            >= F[prev]
                            + self._get_task_time_cost(i)
                            - inf * P[(i, prev)]
                        )
                        mem_cost.append(self._get_mem_cost(prev) * P[(prev, i)])

            mem_i = lpSum(mem_cost) + self._get_mem_cost(i)
            M_limits.append(mem_i)
            if self.system_cfg.M_Limit[self._get_dev(i)] > 0:
                prob += mem_i <= self.system_cfg.M_Limit[self._get_dev(i)]

        res = LpVariable("res")
        # minimize the maximum completion time
        for i in range(self.nnodes):
            cost_sum = []
            for after in range(self.nnodes):
                if i == after or self._get_dev(i) != self._get_dev(after):
                    continue
                if self.inherent_dep[after, i]:
                    continue
                elif self.inherent_dep[i, after]:
                    cost_sum.append(self._get_task_time_cost(after))
                else:
                    cost_sum.append(self._get_task_time_cost(after) * P[(i, after)])
            dev = self._get_dev(i)
            prob += res >= F[i] + lpSum(cost_sum) - F[
                self._get_id(dev, 0, 0)
            ] + self._get_task_time_cost(self._get_id(dev, 0, 0))

        # for dev in range(self.num_dev):
        #     # Notice: different from paper, we minimize the maximum completion time of the whole pipeline
        #     # instead of the max time range of arbitrary device
        #     # (0,0,0) was anchored
        #     prob += res >= F[self._get_id(dev, self.num_mb - 1, 2)] - F[
        #         self._get_id(0, 0, 0)
        #     ] + self._get_task_time_cost(self._get_id(0, 0, 0))

        for i in range(self.num_dev):
            prob += res >= F[self._get_id(i, self.num_mb - 1, 2)] - F[
                self._get_id(i, 0, 0)
            ] + self._get_task_time_cost(self._get_id(i, 0, 0))

        prob.setObjective(res)

        self.prob = prob
        self.prob_F = F

    def get_schedule(self) -> List[List[Tuple[int, int, str, int]]]:
        assert self.prob is not None
        assert self.prob_F is not None

        type_id_to_task = ["F", "B", "W"]
        schedule = [[] for _ in range(self.num_dev)]

        try:
            for dev in range(self.num_dev):
                for mb in range(self.num_mb):
                    for task_type in range(3):
                        task_id = self._get_id(dev, mb, task_type)
                        schedule[dev].append(
                            (
                                dev,
                                mb,
                                type_id_to_task[task_type],
                                int(value(self.prob_F[task_id])),
                            )
                        )

            # sort by completion time
            for dev in range(self.num_dev):
                schedule[dev].sort(key=lambda x: x[3])
        except Exception as e:
            return None

        return schedule


class WaveLikeZBDependencyGraph(DependencyGraph):
    def __init__(self, system_cfg: SystemConfig, enable_relax: bool = False) -> None:
        super().__init__(system_cfg)
        self.enable_relax = enable_relax
        assert self.num_chunk == 2, "Wave-like ZB only supports 2 chunks (1 V) for now"

        self._init_direct_inherent_dependency()
        self._init_inherent_dependency()

    # ID: [dev][mb][chunk][task_type]
    def _get_id(self, dev: int, mb: int, chunk: int, task_type: int) -> int:
        return (
            dev * self.num_mb * self.num_chunk * 3
            + mb * self.num_chunk * 3
            + chunk * 3
            + task_type
        )

    def _get_dev(self, id: int) -> int:
        return id // (self.num_mb * self.num_chunk * 3)

    def _get_mb(self, id: int) -> int:
        return (id // (3 * self.num_chunk)) % self.num_mb

    def _get_chunk(self, id: int) -> int:
        return (id // 3) % self.num_chunk

    def _get_task_type(self, id: int) -> int:
        return id % 3

    def _get_task_time_cost(self, id: int) -> int:
        task_type = self._get_task_type(id)
        dev = self._get_dev(id)
        return [
            self.system_cfg.T_F[dev],
            self.system_cfg.T_B[dev],
            self.system_cfg.T_W[dev],
        ][task_type]

    def _get_comm_cost(self, src_dev_id, dst_dev_id) -> int:
        return self.system_cfg.T_C[src_dev_id][dst_dev_id]

    def _get_mem_cost(self, id: int) -> int:
        task_type = self._get_task_type(id)
        dev = self._get_dev(id)
        return [
            self.system_cfg.M_F[dev],
            self.system_cfg.M_B[dev],
            self.system_cfg.M_W[dev],
        ][task_type]

    def _init_direct_inherent_dependency(self) -> None:
        # inherent dependency is the type of dependency that is not affected by the scheduling
        assert self.inherent_direct_dep is None

        parents = []  # parents[id] is a set of direct dependencies of id
        for dev in range(self.num_dev):
            for mb in range(self.num_mb):
                for chunk in range(self.num_chunk):
                    for task_type in range(3):
                        p = set()
                        if task_type == 0:
                            # F block
                            if mb > 0:
                                # prev mb from same device
                                p.add(self._get_id(dev, mb - 1, chunk, 0))
                            prev_dev = dev - 1 if chunk % 2 == 0 else dev + 1
                            if 0 <= prev_dev < self.num_dev:
                                # same mb from prev device
                                p.add(self._get_id(prev_dev, mb, chunk, 0))
                            if chunk > 0:
                                # same mb same dev from prev chunk
                                p.add(self._get_id(dev, mb, chunk - 1, 0))
                        elif task_type == 1:
                            # B block
                            if chunk == self.num_chunk - 1 and (
                                (chunk % 2 == 0 and dev == self.num_dev - 1)
                                or (chunk % 2 == 1 and dev == 0)
                            ):
                                # last device from corresponding F block
                                p.add(self._get_id(dev, mb, chunk, 0))
                            else:
                                # same mb from prev device
                                prev_dev = dev + 1 if chunk % 2 == 0 else dev - 1
                                if 0 <= prev_dev < self.num_dev:
                                    p.add(self._get_id(prev_dev, mb, chunk, 1))
                            if mb > 0:
                                # prev mb from same device
                                p.add(self._get_id(dev, mb - 1, chunk, 1))
                            if chunk < self.num_chunk - 1:
                                # same mb same dev from prev chunk
                                p.add(self._get_id(dev, mb, chunk + 1, 1))
                        elif task_type == 2:
                            # W block
                            # corresponding B block
                            p.add(self._get_id(dev, mb, chunk, 1))
                            if mb > 0:
                                # prev mb from same device
                                # not necessary, but shrink the search space
                                p.add(self._get_id(dev, mb - 1, chunk, 2))
                            if chunk < self.num_chunk - 1:
                                # same mb same dev from prev chunk
                                p.add(self._get_id(dev, mb, chunk + 1, 2))
                        else:
                            raise ValueError("Invalid task type")
                        parents.append(p)
        self.inherent_direct_dep = parents

    def build_ilp(self) -> None:
        prob = LpProblem("AutoSchedule", LpMinimize)

        # dependency order graph
        # P[i][j] = 1 if i is scheduled before j
        # i and j are on the same device
        # schedulable dep as lp variables
        P: Dict[Tuple, LpVariable] = {}
        for i in range(self.nnodes):
            for j in range(i):
                if self._schedulable_on_dev(i, j):
                    if self.enable_relax:
                        P[(i, j)] = LpVariable(f"P_{i}_{j}", 0, 1, cat="Continuous")
                    else:
                        P[(i, j)] = LpVariable(f"P_{i}_{j}", 0, 1, cat="Binary")
                    P[(j, i)] = 1 - P[(i, j)]

        # completion time
        F: Dict[int, LpVariable] = LpVariable.dicts(
            "F", (range(self.nnodes),), None, None, cat="Continuous"
        )

        inf = (
            (
                max(self.system_cfg.T_F)
                + max(self.system_cfg.T_B)
                + max(self.system_cfg.T_W)
                + np.max(self.system_cfg.T_C) * 3
            )
            * self.num_dev
            * self.num_mb
            * self.num_chunk
        )

        # anchor the first task of the 0th device
        first_task = self._get_id(0, 0, 0, 0)
        prob += F[first_task] >= self._get_task_time_cost(first_task)

        M_limits = []

        for i in range(self.nnodes):
            mem_cost = []
            for prev in range(self.nnodes):
                if i == prev:
                    continue
                if prev in self.inherent_direct_dep[i]:
                    # direct dependency, cross device or same device
                    prob += F[i] >= F[prev] + self._get_task_time_cost(i) + (
                        self._get_comm_cost(self._get_dev(prev), self._get_dev(i))
                    )

                if self._get_dev(i) == self._get_dev(prev):
                    if self.inherent_dep[i, prev]:
                        pass
                    elif self.inherent_dep[prev, i]:
                        mem_cost.append(self._get_mem_cost(prev))
                    else:
                        # schedulable dependency
                        prob += (
                            F[i]
                            >= F[prev]
                            + self._get_task_time_cost(i)
                            - inf * P[(i, prev)]
                        )
                        mem_cost.append(self._get_mem_cost(prev) * P[(prev, i)])

            mem_i = lpSum(mem_cost) + self._get_mem_cost(i)
            M_limits.append(mem_i)
            if self.system_cfg.M_Limit[self._get_dev(i)] > 0:
                prob += mem_i <= self.system_cfg.M_Limit[self._get_dev(i)]

        res = LpVariable("res")
        # minimize the maximum completion time
        for i in range(self.nnodes):
            cost_sum = []
            for after in range(self.nnodes):
                if i == after or self._get_dev(i) != self._get_dev(after):
                    continue
                if self.inherent_dep[after, i]:
                    continue
                elif self.inherent_dep[i, after]:
                    cost_sum.append(self._get_task_time_cost(after))
                else:
                    cost_sum.append(self._get_task_time_cost(after) * P[(i, after)])
            dev = self._get_dev(i)
            prob += res >= F[i] + lpSum(cost_sum) - F[
                self._get_id(dev, 0, 0, 0)
            ] + self._get_task_time_cost(self._get_id(dev, 0, 0, 0))

        # for dev in range(self.num_dev):
        #     # Notice: different from paper, we minimize the maximum completion time of the whole pipeline
        #     # instead of the max time range of arbitrary device
        #     # (0,0,0) was anchored
        #     prob += res >= F[self._get_id(dev, self.num_mb - 1, 2)] - F[
        #         self._get_id(0, 0, 0)
        #     ] + self._get_task_time_cost(self._get_id(0, 0, 0))

        for i in range(self.num_dev):
            prob += res >= F[self._get_id(i, self.num_mb - 1, 0, 2)] - F[
                self._get_id(i, 0, 0, 0)
            ] + self._get_task_time_cost(self._get_id(i, 0, 0, 0))

        prob.setObjective(res)

        self.prob = prob
        self.prob_F = F

    def get_schedule(self) -> List[List[Tuple[int, int, str, int]]]:
        assert self.prob is not None
        assert self.prob_F is not None

        type_id_to_task = ["F", "B", "W"]
        schedule = [[] for _ in range(self.num_dev)]
        try:
            for dev in range(self.num_dev):
                for mb in range(self.num_mb):
                    for chunk in range(self.num_chunk):
                        for task_type in range(3):
                            task_id = self._get_id(dev, mb, chunk, task_type)
                            schedule[dev].append(
                                (
                                    dev,
                                    mb,
                                    chunk,
                                    type_id_to_task[task_type],
                                    int(value(self.prob_F[task_id])),
                                )
                            )

            # sort by completion time
            for dev in range(self.num_dev):
                schedule[dev].sort(key=lambda x: x[4])

        except Exception as e:
            return None

        return schedule
