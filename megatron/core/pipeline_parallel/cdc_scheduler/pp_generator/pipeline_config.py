from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np


@dataclass
class SystemConfig:
    T_F: Union[int, List[int]] = 100
    T_B: Union[int, List[int]] = 200
    T_C: Union[int, np.ndarray] = 5
    T_W: Union[int, List[int]] = 0

    # Memory
    M_F: int = -1
    M_B: int = -1
    M_W: int = -1
    M_Limit: int = -1

    num_devices: int = 4
    num_microbatches: int = 8

    num_chunks: int = 1

    # heuristic wave only
    aux_tear_down_opt: bool = False
    aux_w_if_b_mem_limited: bool = False
    bootstrap_soft_bound: bool = False

    # heuristic ud only
    aux_interleave_priority: bool = False

    def __hash__(self) -> int:
        # Convert list and ndarray attributes to tuples for hashing
        T_F_hashable = tuple(self.T_F) if isinstance(self.T_F, list) else self.T_F
        T_B_hashable = tuple(self.T_B) if isinstance(self.T_B, list) else self.T_B
        T_C_hashable = (
            tuple(self.T_C.flatten().tolist())
            if isinstance(self.T_C, np.ndarray)
            else self.T_C
        )
        T_W_hashable = tuple(self.T_W) if isinstance(self.T_W, list) else self.T_W
        M_F_hashable = tuple(self.M_F) if isinstance(self.M_F, list) else self.M_F
        M_B_hashable = tuple(self.M_B) if isinstance(self.M_B, list) else self.M_B
        M_W_hashable = tuple(self.M_W) if isinstance(self.M_W, list) else self.M_W
        M_Limit_hashable = (
            tuple(self.M_Limit) if isinstance(self.M_Limit, list) else self.M_Limit
        )

        # Compute the combined hash
        return hash(
            (
                T_F_hashable,
                T_B_hashable,
                T_C_hashable,
                T_W_hashable,
                M_F_hashable,
                M_B_hashable,
                M_W_hashable,
                M_Limit_hashable,
                self.num_devices,
                self.num_microbatches,
                self.num_chunks,
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SystemConfig):
            return False

        return (
            self.T_F == other.T_F
            and self.T_B == other.T_B
            and np.array_equal(self.T_C, other.T_C)
            and self.T_W == other.T_W
            and self.M_F == other.M_F
            and self.M_B == other.M_B
            and self.M_W == other.M_W
            and self.M_Limit == other.M_Limit
            and self.num_devices == other.num_devices
            and self.num_microbatches == other.num_microbatches
            and self.num_chunks == other.num_chunks
        )

    def __post_init__(self):
        if not isinstance(self.T_F, list):
            self.T_F = [self.T_F] * self.num_devices
        if not isinstance(self.T_B, list):
            self.T_B = [self.T_B] * self.num_devices
        if not isinstance(self.T_W, list):
            self.T_W = [self.T_W] * self.num_devices

        if not isinstance(self.T_C, np.ndarray):
            comm_matrix = (
                np.ones((self.num_devices, self.num_devices)) * self.T_C
                - np.eye(self.num_devices) * self.T_C
            )
            self.T_C = comm_matrix

        if not isinstance(self.M_F, list):
            self.M_F = [self.M_F] * self.num_devices
        if not isinstance(self.M_B, list):
            self.M_B = [self.M_B] * self.num_devices
        if not isinstance(self.M_W, list):
            self.M_W = [self.M_W] * self.num_devices
        if not isinstance(self.M_Limit, list):
            self.M_Limit = [self.M_Limit] * self.num_devices
