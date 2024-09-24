import math
import sys
import traceback
from typing import List, Tuple

from auto_schedule_store import AutoScheduleStore
from pipeline_config import SystemConfig
from simulator import SimCfgGen
from util import generate_comm_mat

'''
TODO: testing and cleanup
'''

def generate_sys_configs(
    T_F_total: int,
    T_B_total: int,
    T_W_total: int,
    num_devices_total: int,
    num_DC: int,
    num_microbatches: int,
    T_C_intra: int,
    T_C_inter: int,
    M_F_total: int,
    M_B_total: int,
    M_W_total: int,
) -> Tuple[List[SystemConfig]]:
    # UD
    assert T_F_total % (num_devices_total * 2) == 0
    assert T_B_total % (num_devices_total * 2) == 0
    assert T_W_total % (num_devices_total * 2) == 0
    assert M_F_total % (num_devices_total * 2) == 0
    assert M_B_total % (num_devices_total * 2) == 0
    assert M_W_total % (num_devices_total * 2) == 0
    assert M_F_total + M_B_total + M_W_total == 0
    assert num_devices_total % num_DC == 0

    cfg_ud_list = []
    T_F_ud = T_F_total // num_devices_total
    T_B_ud = T_B_total // num_devices_total
    T_W_ud = T_W_total // num_devices_total
    M_F_ud = M_F_total // num_devices_total
    M_B_ud = M_B_total // num_devices_total
    M_W_ud = M_W_total // num_devices_total
    expected_num_mb_ud = math.ceil(
        2 * num_devices_total
        - 1
        + 2 * num_devices_total * T_C_inter * (num_DC - 1) / T_F_total
    )
    for num_mb_warmup in [
        num_devices_total,
        num_devices_total * 2,
        expected_num_mb_ud,
        expected_num_mb_ud * 2,
        num_microbatches,
    ]:
        actual_mb_warmup = min(num_mb_warmup, num_microbatches)
        M_limit = actual_mb_warmup * M_F_ud
        cfg_ud_list.append(
            SystemConfig(
                num_devices=num_devices_total,
                num_microbatches=num_microbatches,
                T_F=T_F_ud,
                T_B=T_B_ud,
                T_W=T_W_ud,
                T_C=generate_comm_mat(
                    num_DC, num_devices_total // num_DC, T_C_intra, T_C_inter
                ),
                M_F=M_F_ud,
                M_B=M_B_ud,
                M_W=M_W_ud,
                M_Limit=M_limit,
            )
        )

    # wave
    cfg_wave_list = []
    T_F_wave = T_F_total // num_devices_total // 2
    T_B_wave = T_B_total // num_devices_total // 2
    T_W_wave = T_W_total // num_devices_total // 2
    M_F_wave = M_F_total // num_devices_total // 2
    M_B_wave = M_B_total // num_devices_total // 2
    M_W_wave = M_W_total // num_devices_total // 2
    expected_num_mb_wave = math.ceil(
        2 * num_devices_total
        - 1
        + 4 * num_devices_total * T_C_inter * (num_DC - 1) / T_F_total
    )
    for num_mb_warmup in [
        num_devices_total,
        num_devices_total * 2,
        expected_num_mb_wave,
        expected_num_mb_wave * 2,
        num_microbatches,
    ]:
        actual_mb_warmup = min(num_mb_warmup, num_microbatches)
        M_limit = actual_mb_warmup * M_F_wave
        cfg_wave_list.append(
            SystemConfig(
                num_devices=num_devices_total,
                num_microbatches=num_microbatches,
                num_chunks=2,
                T_F=T_F_wave,
                T_B=T_B_wave,
                T_W=T_W_wave,
                T_C=generate_comm_mat(
                    num_DC, num_devices_total // num_DC, T_C_intra, T_C_inter
                ),
                M_F=M_F_wave,
                M_B=M_B_wave,
                M_W=M_W_wave,
                M_Limit=M_limit,
            )
        )
    return cfg_ud_list, cfg_wave_list


def generate_auto_schedules():
    # 96 layers
    # Per layer Tf=20, Tb=20, Tw=20
    # Per layer Mf=20, Mb=-10, Mw=-10
    for num_DC in [2, 3, 4]:
        for device_per_DC in [2, 4]:
            num_device = num_DC * device_per_DC
            for num_mb in [x * num_device for x in [2, 4, 6]]:
                T_C_intra = 0
                for T_C_inter_ratio in [0, 1, 2, 3, 4]:
                    print("------------------------------------")
                    print(
                        f"----Generating: num_DC={num_DC}, device_per_DC={device_per_DC}, num_mb={num_mb}, T_C_inter_ratio={T_C_inter_ratio}"
                    )
                    T_C_inter = T_C_inter_ratio * 1920 / num_device
                    try:
                        cfg_ud_list, cfg_wave_list = generate_sys_configs(
                            T_F_total=1920,
                            T_B_total=1920,
                            T_W_total=1920,
                            num_devices_total=num_device,
                            num_DC=num_DC,
                            num_microbatches=num_mb,
                            T_C_intra=T_C_intra,
                            T_C_inter=T_C_inter,
                            M_F_total=1920,
                            M_B_total=-960,
                            M_W_total=-960,
                        )
                    except AssertionError:
                        print("----Invalid configuration, skip...")

                    with AutoScheduleStore(save_interval=1) as store:
                        for cfg in cfg_ud_list:
                            store.get_ud_schedule_result(cfg)
                        for cfg in cfg_wave_list:
                            store.get_wave_schedule_result(cfg)


def generate_real_model_sim_schedule():
    PPxDP = 1024
    seq_len = 8196
    tp = 8
    num_layers = 128
    num_mb_per_pp_stage = 4
    with open("405B_result.csv", "w") as f:
        f.write(
            "model_size seq_len mbs tp pp dp num_mb num_chunks num_DC num_layers optimal objective_value time_limit\n"
        )

    with AutoScheduleStore(
        "405_ud_store.pkl", "405_wave_store.pkl", save_interval=1
    ) as store:
        # strong scaling
        for pp in [2, 4, 8]:
            dp = PPxDP // pp
            for n_DC in [2, 4]:
                if pp % n_DC != 0:
                    continue
                if num_layers % (pp * 2) != 0:
                    continue
                try:
                    print(f"----- strong scaling: pp={pp}, n_DC={n_DC}")
                    sim_ud = SimCfgGen(
                        llama_model_size=405,
                        seq_len=seq_len,
                        mbs=1,
                        tp=tp,
                        pp=pp,
                        dp=dp,
                        num_mb_per_pp_stage=num_mb_per_pp_stage,
                        num_chunks=1,
                        gpu_mem_bytes=88 * 1024**3,  # around 96GB
                        gpu_avg_perf_flops=350 * 10**12,
                        num_DC=n_DC,
                        DC_comm_latency=0.01,
                        DC_comm_bandwidth=32 * 10**9,
                        num_layers=num_layers,
                    )

                    sim_ud_res = store.get_ud_schedule_result(
                        sim_ud.get_system_config()
                    )
                    with open("405B_result.csv", "a") as f:
                        f.write(
                            f"405 {seq_len} {sim_ud.mbs} {tp} {pp} {dp} {num_mb_per_pp_stage*pp} {1} {n_DC} {num_layers} {sim_ud_res.lp_status == 1} {sim_ud_res.objective_value / sim_ud.time_scale_factor} {sim_ud_res.time_limit}\n"
                        )
                except Exception as e:
                    # Print detailed error information with line number
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)

                    # Alternatively, print just the exception with traceback as a string
                    error_message = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    print(f"Error occurred: {error_message}")

                try:
                    sim_wave = SimCfgGen(
                        llama_model_size=405,
                        seq_len=seq_len,
                        mbs=1,
                        tp=tp,
                        pp=pp,
                        dp=dp,
                        num_mb_per_pp_stage=num_mb_per_pp_stage,
                        num_chunks=2,
                        gpu_mem_bytes=88 * 1024**3,  # around 96GB
                        gpu_avg_perf_flops=350 * 10**12,
                        num_DC=n_DC,
                        DC_comm_latency=0.01,
                        DC_comm_bandwidth=32 * 10**9,
                        num_layers=num_layers,
                    )
                    sim_wave_res = store.get_wave_schedule_result(
                        sim_wave.get_system_config(), verbose=True
                    )
                    with open("405B_result.csv", "a") as f:
                        f.write(
                            f"405 {seq_len} {sim_wave.mbs} {tp} {pp} {dp} {num_mb_per_pp_stage*pp} {2} {n_DC} {num_layers} {sim_wave_res.lp_status == 1} {sim_wave_res.objective_value / sim_wave.time_scale_factor} {sim_wave_res.time_limit}\n"
                        )
                except Exception as e:
                    # Print detailed error information with line number
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)

                    # Alternatively, print just the exception with traceback as a string
                    error_message = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    print(f"Error occurred: {error_message}")

        # weak scaling
        for num_layers in [32, 64, 128, 256, 512]:
            for pp in [2, 4, 8]:
                dp = 64
                for n_DC in [2, 4]:
                    if pp % n_DC != 0:
                        continue
                    if num_layers % (pp * 2) != 0:
                        continue
                    try:
                        print(
                            f"----- weak scaling: num_layers={num_layers}, pp={pp}, n_DC={n_DC}"
                        )
                        sim_ud = SimCfgGen(
                            llama_model_size=405,
                            seq_len=seq_len,
                            mbs=1,
                            tp=tp,
                            pp=pp,
                            dp=dp,
                            num_mb_per_pp_stage=num_mb_per_pp_stage,
                            num_chunks=1,
                            gpu_mem_bytes=88 * 1024**3,  # around 96GB
                            gpu_avg_perf_flops=350 * 10**12,
                            num_DC=n_DC,
                            DC_comm_latency=0.01,
                            DC_comm_bandwidth=32 * 10**9,
                            num_layers=num_layers,
                        )

                        sim_ud_res = store.get_ud_schedule_result(
                            sim_ud.get_system_config()
                        )
                        with open("405B_result.csv", "a") as f:
                            f.write(
                                f"405 {seq_len} {sim_ud.mbs} {tp} {pp} {dp} {num_mb_per_pp_stage*pp} {1} {n_DC} {num_layers} {sim_ud_res.lp_status == 1} {sim_ud_res.objective_value / sim_ud.time_scale_factor} {sim_ud_res.time_limit}\n"
                            )
                    except Exception as e:
                        # Print detailed error information with line number
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_exception(exc_type, exc_value, exc_traceback)

                        # Alternatively, print just the exception with traceback as a string
                        error_message = "".join(
                            traceback.format_exception(
                                exc_type, exc_value, exc_traceback
                            )
                        )
                        print(f"Error occurred: {error_message}")

                    try:
                        sim_wave = SimCfgGen(
                            llama_model_size=405,
                            seq_len=seq_len,
                            mbs=1,
                            tp=tp,
                            pp=pp,
                            dp=dp,
                            num_mb_per_pp_stage=num_mb_per_pp_stage,
                            num_chunks=2,
                            gpu_mem_bytes=88 * 1024**3,  # around 96GB
                            gpu_avg_perf_flops=350 * 10**12,
                            num_DC=n_DC,
                            DC_comm_latency=0.01,
                            DC_comm_bandwidth=32 * 10**9,
                            num_layers=num_layers,
                        )
                        sim_wave_res = store.get_wave_schedule_result(
                            sim_wave.get_system_config()
                        )
                        with open("405B_result.csv", "a") as f:
                            f.write(
                                f"405 {seq_len} {sim_wave.mbs} {tp} {pp} {dp} {num_mb_per_pp_stage*pp} {2} {n_DC} {num_layers} {sim_wave_res.lp_status == 1} {sim_wave_res.objective_value / sim_wave.time_scale_factor} {sim_wave_res.time_limit}\n"
                            )
                    except Exception as e:
                        # Print detailed error information with line number
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_exception(exc_type, exc_value, exc_traceback)

                        # Alternatively, print just the exception with traceback as a string
                        error_message = "".join(
                            traceback.format_exception(
                                exc_type, exc_value, exc_traceback
                            )
                        )
                        print(f"Error occurred: {error_message}")

        # bandwidth scaling
        num_layers = 128
        pp = 4
        dp = 64
        for DC_bw in [1, 2, 4, 8, 16, 32, 64, 128]:
            DC_comm_bandwidth = DC_bw * 10**9
            for n_DC in [2, 4]:
                if pp % n_DC != 0:
                    continue
                if num_layers % (pp * 2) != 0:
                    continue
                try:
                    print(f"----- Bandwidth scaling: DC_bw={DC_bw}, n_DC={n_DC}")
                    sim_ud = SimCfgGen(
                        llama_model_size=405,
                        seq_len=seq_len,
                        mbs=1,
                        tp=tp,
                        pp=pp,
                        dp=dp,
                        num_mb_per_pp_stage=4,
                        num_chunks=1,
                        gpu_mem_bytes=88 * 1024**3,  # around 96GB
                        gpu_avg_perf_flops=350 * 10**12,
                        num_DC=n_DC,
                        DC_comm_latency=0.01,
                        DC_comm_bandwidth=DC_comm_bandwidth,
                        num_layers=num_layers,
                    )

                    sim_ud_res = store.get_ud_schedule_result(
                        sim_ud.get_system_config()
                    )
                    with open("405B_result.csv", "a") as f:
                        f.write(
                            f"405 {seq_len} {sim_ud.mbs} {tp} {pp} {dp} {num_mb_per_pp_stage*pp} {1} {n_DC} {num_layers} {sim_ud_res.lp_status == 1} {sim_ud_res.objective_value / sim_ud.time_scale_factor} {sim_ud_res.time_limit}\n"
                        )
                except Exception as e:
                    # Print detailed error information with line number
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)

                    # Alternatively, print just the exception with traceback as a string
                    error_message = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    print(f"Error occurred: {error_message}")

                try:
                    sim_wave = SimCfgGen(
                        llama_model_size=405,
                        seq_len=seq_len,
                        mbs=1,
                        tp=tp,
                        pp=pp,
                        dp=dp,
                        num_mb_per_pp_stage=4,
                        num_chunks=2,
                        gpu_mem_bytes=88 * 1024**3,  # around 96GB
                        gpu_avg_perf_flops=350 * 10**12,
                        num_DC=n_DC,
                        DC_comm_latency=0.01,
                        DC_comm_bandwidth=32 * 10**9,
                        num_layers=num_layers,
                    )
                    sim_wave_res = store.get_wave_schedule_result(
                        sim_wave.get_system_config()
                    )
                    with open("405B_result.csv", "a") as f:
                        f.write(
                            f"405 {seq_len} {sim_wave.mbs} {tp} {pp} {dp} {num_mb_per_pp_stage*pp} {2} {n_DC} {num_layers} {sim_wave_res.lp_status == 1} {sim_wave_res.objective_value / sim_wave.time_scale_factor} {sim_wave_res.time_limit}\n"
                        )
                except Exception as e:
                    # Print detailed error information with line number
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)

                    # Alternatively, print just the exception with traceback as a string
                    error_message = "".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
                    print(f"Error occurred: {error_message}")
