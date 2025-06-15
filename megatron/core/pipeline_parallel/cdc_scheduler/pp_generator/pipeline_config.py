from dataclasses import dataclass
from typing import Any, List, Union, Optional

import numpy as np


def _broadcast_to_2d(val, num_chunks, num_devices):
    if isinstance(val, int):
        return np.full((num_chunks, num_devices), val, dtype=int)
    if isinstance(val, (list, np.ndarray)):
        # Already 2D
        if isinstance(val, list) and all(isinstance(row, list) for row in val):
            assert len(val) == num_chunks and all(len(row) == num_devices for row in val)
            return np.array(val, dtype=int)
        # 1D list/array, broadcast to all chunks
        if isinstance(val, list):
            assert len(val) == num_devices
            return np.tile(np.array(val, dtype=int), (num_chunks, 1))
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                assert val.shape[0] == num_devices
                return np.tile(val, (num_chunks, 1))
            elif val.ndim == 2:
                assert val.shape == (num_chunks, num_devices)
                return val
            else:
                raise ValueError(f"Unsupported numpy array shape: {val.shape}")
    raise ValueError(f"Unsupported type for broadcast: {type(val)}")

@dataclass
class SystemConfig:
    # (chunk, dev)
    T_F: Union[int, np.ndarray] = 100
    T_B: Union[int, np.ndarray] = 200
    T_W: Union[int, np.ndarray] = 0
    
    T_alpha: Union[int, np.ndarray] = 0  # latency
    # link bw time: inverse of bandwidth
    T_beta: Union[int, np.ndarray] = 0

    # Memory
    M_F: Union[int, np.ndarray] = -1
    M_B: Union[int, np.ndarray] = -1
    M_W: Union[int, np.ndarray] = -1
    M_Limit: Union[int, np.ndarray] = -1

    # Data parallel communication time (num_chunks, pp_size, 2): allgather param and reduce_scatter grad
    T_DP: Union[int, np.ndarray] = 0
    zero_1_dp_modeling: bool = False

    num_devices: int = 4
    num_microbatches: int = 8

    num_chunks: int = 1
    
    # cross-DC
    # needed in bandwidth simulation (loop schedules only)
    two_dc: Optional[bool] = None
    # TODO: currently we assume equal split
    two_dc_num_dev_per_dc: int = -1
    
    aux_w_if_b_mem_limited: bool = False

    # heuristic wave only
    aux_tear_down_opt: bool = False

    # heuristic ud only
    aux_interleave_priority: bool = False

    def __hash__(self) -> int:
        def hash_array(val):
            if isinstance(val, np.ndarray):
                return tuple(val.flatten().tolist())
            return val
        return hash(
            (
                hash_array(self.T_F),
                hash_array(self.T_B),
                hash_array(self.T_alpha),
                hash_array(self.T_beta),
                hash_array(self.T_W),
                hash_array(self.M_F),
                hash_array(self.M_B),
                hash_array(self.M_W),
                hash_array(self.M_Limit),
                hash_array(self.T_DP),
                self.num_devices,
                self.num_microbatches,
                self.num_chunks,
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SystemConfig):
            return False
        def eq_or_array(a, b):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return np.array_equal(a, b)
            return a == b
        return (
            eq_or_array(self.T_F, other.T_F)
            and eq_or_array(self.T_B, other.T_B)
            and eq_or_array(self.T_alpha, other.T_alpha)
            and eq_or_array(self.T_beta, other.T_beta)
            and eq_or_array(self.T_W, other.T_W)
            and eq_or_array(self.M_F, other.M_F)
            and eq_or_array(self.M_B, other.M_B)
            and eq_or_array(self.M_W, other.M_W)
            and eq_or_array(self.M_Limit, other.M_Limit)
            and eq_or_array(self.T_DP, other.T_DP)
            and self.num_devices == other.num_devices
            and self.num_microbatches == other.num_microbatches
            and self.num_chunks == other.num_chunks
        )

    def __post_init__(self):
        self.T_F = _broadcast_to_2d(self.T_F, self.num_chunks, self.num_devices)
        self.T_B = _broadcast_to_2d(self.T_B, self.num_chunks, self.num_devices)
        self.T_W = _broadcast_to_2d(self.T_W, self.num_chunks, self.num_devices)
        self.M_F = _broadcast_to_2d(self.M_F, self.num_chunks, self.num_devices)
        self.M_B = _broadcast_to_2d(self.M_B, self.num_chunks, self.num_devices)
        self.M_W = _broadcast_to_2d(self.M_W, self.num_chunks, self.num_devices)
        
        if not isinstance(self.T_alpha, np.ndarray):
            if isinstance(self.T_alpha, list):
                self.T_alpha = np.array(self.T_alpha, dtype=int)
            else:
                assert isinstance(self.T_alpha, int)
                comm_matrix = (
                    np.ones((self.num_devices, self.num_devices), dtype=int) * self.T_alpha
                    - np.eye(self.num_devices, dtype=int) * self.T_alpha
                )
                self.T_alpha = comm_matrix
        elif self.T_alpha.ndim != 2 or self.T_alpha.shape != (self.num_devices, self.num_devices):
            raise ValueError(f"T_alpha must be a {self.num_devices}x{self.num_devices} matrix")
                
        if not isinstance(self.T_beta, np.ndarray):
            if isinstance(self.T_beta, list):
                self.T_beta = np.array(self.T_beta, dtype=int)
            else:
                assert isinstance(self.T_beta, int)
                beta_matrix = (
                    np.ones((self.num_devices, self.num_devices), dtype=int) * self.T_beta
                    - np.eye(self.num_devices, dtype=int) * self.T_beta
                )
                self.T_beta = beta_matrix
        elif self.T_beta.ndim != 2 or self.T_beta.shape != (self.num_devices, self.num_devices):
            raise ValueError(f"T_beta must be a {self.num_devices}x{self.num_devices} matrix")
                
        if not isinstance(self.M_Limit, np.ndarray):
            if isinstance(self.M_Limit, list):
                self.M_Limit = np.array(self.M_Limit, dtype=int)
            else:
                self.M_Limit = np.full(self.num_devices, self.M_Limit, dtype=int)
        elif self.M_Limit.ndim != 1 or self.M_Limit.shape[0] != self.num_devices:
            raise ValueError(f"M_Limit must be a 1D array of length {self.num_devices}")
        
        if not isinstance(self.T_DP, np.ndarray):
            if isinstance(self.T_DP, list):
                self.T_DP = np.array(self.T_DP, dtype=int)
            else:
                self.T_DP = np.zeros((self.num_chunks, self.num_devices, 2), dtype=int)
        elif self.T_DP.ndim != 3 or self.T_DP.shape != (self.num_chunks, self.num_devices, 2):
            raise ValueError(f"T_DP must be a {self.num_chunks}x{self.num_devices}x2 array")


@dataclass
class PipelineBlockDesc:
    device_id: int
    mb_id: int
    task_type: str
    chunk_id: int = 0
    start_time: int = 0
    end_time: int = 0
    subpart_start: int = 0
    subpart_end: int = 1
    num_subparts: int = 1
    post_send_time: int = -1
