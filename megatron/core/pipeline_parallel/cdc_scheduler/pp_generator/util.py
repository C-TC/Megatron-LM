import math
from typing import List, Tuple
import numpy as np


def generate_comm_mat(num_DC, num_dev_per_DC, T_intra, T_inter) -> np.ndarray:
    num_dev = num_DC * num_dev_per_DC
    ret = (np.ones((num_dev, num_dev)) - np.eye(num_dev)) * T_intra
    for i in range(num_DC):
        for j in range(num_DC):
            if i != j:
                ret[
                    i * num_dev_per_DC : (i + 1) * num_dev_per_DC,
                    j * num_dev_per_DC : (j + 1) * num_dev_per_DC,
                ] = T_inter
    return ret


def scale_to_int(val_list: List[float], min_interval=8) -> Tuple[List[int], float]:
    min_gap = math.inf
    for i, val_x in enumerate(val_list):
        for j in range(i + 1, len(val_list)):
            min_gap = min(min_gap, abs(abs(val_list[j]) - abs(val_x)))
    scale_factor = min_interval / min_gap
    return [int(round(x * scale_factor)) for x in val_list], scale_factor
