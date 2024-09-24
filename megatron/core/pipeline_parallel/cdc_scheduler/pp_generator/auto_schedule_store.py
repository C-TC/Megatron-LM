from dataclasses import dataclass
import os
import pickle
from typing import Dict, List

from auto_schedule import UnidirectionalZBDependencyGraph, WaveLikeZBDependencyGraph
from pipeline_config import SystemConfig


@dataclass
class AutoScheduleResult:
    schedule: List
    lp_status: int
    time_limit: int
    objective_value: int = None


class AutoScheduleStore:
    def __init__(
        self,
        ud_file: str = "ud_store.pkl",
        wave_file: str = "wave_store.pkl",
        save_interval: int = 4,
    ) -> None:
        self.ud_file = ud_file
        self.wave_file = wave_file

        self.ud_store: Dict[SystemConfig, AutoScheduleResult] = {}
        self.wave_store: Dict[SystemConfig, AutoScheduleResult] = {}

        self.new_schedule_added = 0
        self.save_interval = save_interval

        if os.path.exists(ud_file):
            with open(ud_file, "rb") as f:
                self.ud_store = pickle.load(f)

        if os.path.exists(wave_file):
            with open(wave_file, "rb") as f:
                self.wave_store = pickle.load(f)

    def get_ud_schedule_result(
        self,
        sys_config: SystemConfig,
        compute_if_not_exist: bool = True,
        time_limit: int = 200,
        verbose: bool = False,
    ) -> AutoScheduleResult:
        if (
            sys_config in self.ud_store
            # and self.ud_store[sys_config].schedule is not None
            and time_limit <= self.ud_store[sys_config].time_limit
        ):
            return self.ud_store[sys_config]
        elif not compute_if_not_exist:
            return None
        dg = UnidirectionalZBDependencyGraph(sys_config)
        dg.build_ilp()
        dg.solve_ilp(verbose=verbose, time_limit=time_limit, warm_start=False)
        schedule = dg.get_schedule()
        self.ud_store[sys_config] = AutoScheduleResult(
            schedule, dg.get_lp_status(), time_limit, dg.get_objective_value()
        )
        self.save_dict_if_needed()
        return self.ud_store[sys_config]

    def get_wave_schedule_result(
        self,
        sys_config: SystemConfig,
        compute_if_not_exist: bool = True,
        time_limit: int = 200,
        verbose: bool = False,
    ) -> AutoScheduleResult:
        if (
            sys_config in self.wave_store
            # and self.wave_store[sys_config].schedule is not None
            and time_limit <= self.wave_store[sys_config].time_limit
        ):
            return self.wave_store[sys_config]
        elif not compute_if_not_exist:
            return None
        dg = WaveLikeZBDependencyGraph(sys_config)
        dg.build_ilp()
        dg.solve_ilp(verbose=verbose, time_limit=time_limit, warm_start=False)
        schedule = dg.get_schedule()
        self.wave_store[sys_config] = AutoScheduleResult(
            schedule, dg.get_lp_status(), time_limit, dg.get_objective_value()
        )
        self.save_dict_if_needed()
        return self.wave_store[sys_config]

    def save_dict_if_needed(self):
        self.new_schedule_added += 1
        if self.new_schedule_added % self.save_interval == 0:
            with open(self.ud_file, "wb") as f:
                pickle.dump(self.ud_store, f)
            with open(self.wave_file, "wb") as f:
                pickle.dump(self.wave_store, f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.ud_file, "wb") as f:
            pickle.dump(self.ud_store, f)
        with open(self.wave_file, "wb") as f:
            pickle.dump(self.wave_store, f)
