from typing import Type

from pipeline_config import SystemConfig
from auto_schedule import UnidirectionalZBDependencyGraph, WaveLikeZBDependencyGraph
from auto_schedule_store import AutoScheduleStore
from pipeline import (
    AutoUDZBPipeline,
    AutoWaveZBPipeline,
    GpipePipeline,
    Hanayo1F1BPipeline,
    HeuristicWaveZBPipeline,
    HeuristicWaveZBPipelineV2,
    HeuristicZBVPipeline,
    HeuristicUDPipeline,
    HeuristicZBUDPipeline,
    Interleaved1F1BPipeline,
    OneFOneBPipeline,
    Pipeline,
    ZBH1Pipeline,
)
from simulator import BFSPPSimCfgGen, SimCfgGen
from util import generate_comm_mat
from megatron.core.pipeline_parallel.cdc_scheduler.execution_planner import ExecutionPlanner


def test_ud_zb():
    sys_config = SystemConfig(
        num_devices=4,
        num_microbatches=12,
        T_F=200,
        T_B=200,
        T_W=200,
        T_C=generate_comm_mat(2, 2, 0, 100),
        M_F=2,
        M_B=-1,
        M_W=-1,
        M_Limit=800,
    )  # 8

    dg = UnidirectionalZBDependencyGraph(sys_config)
    dg.build_ilp()
    dg.solve_ilp(time_limit=20, warm_start=False)
    schedule = dg.get_schedule()

    azb = AutoUDZBPipeline(sys_config)
    azb.schedule(schedule)
    azb.solve_dependencies()
    azb.print_debug_schedule(verbose=1)
    azb.print_schedule()


def test_wave_zb(relax: bool = False):
    sys_config = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        num_chunks=2,
        T_F=100,
        T_B=100,
        T_W=100,
        T_C=0,
        M_F=2,
        M_B=-1,
        M_W=-1,
        M_Limit=16,
    )  # 8

    dg = WaveLikeZBDependencyGraph(sys_config, enable_relax=relax)
    dg.build_ilp()
    dg.solve_ilp(time_limit=200, warm_start=False)
    schedule = dg.get_schedule()

    azb = AutoWaveZBPipeline(sys_config)
    azb.schedule(schedule)
    azb.solve_dependencies()
    azb.print_debug_schedule(verbose=1)
    azb.print_schedule(save=True)


def test_schedule_store():
    ud_sys_cfg = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        T_F=200,
        T_B=200,
        T_W=200,
        T_C=generate_comm_mat(2, 2, 0, 100),
        M_F=4,
        M_B=-2,
        M_W=-2,
        M_Limit=800,
    )
    wave_sys_cfg = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        num_chunks=2,
        T_F=100,
        T_B=100,
        T_W=100,
        T_C=generate_comm_mat(2, 2, 0, 100),
        M_F=2,
        M_B=-1,
        M_W=-1,
        M_Limit=800,
    )

    with AutoScheduleStore("ud_store.pkl", "wave_store.pkl") as store:
        ud_schedule = store.get_ud_schedule_result(ud_sys_cfg).schedule
        wave_schedule = store.get_wave_schedule_result(wave_sys_cfg).schedule


def test_bfs_simulator():
    sim = BFSPPSimCfgGen(
        llama_model_size=405,
        seq_len=8196,
        mbs=1,
        tp=8,
        pp=8,
        dp=64,
        num_chunks=4,
        gpu_mem_bytes=88 * 1024**3,  # around 96GB
        gpu_avg_perf_flops=350 * 10**12,
        num_DC=2,
        DC_comm_latency=0.01,
        DC_comm_bandwidth=8 * 10**9,
        DC_intra_comm_bandwidth=160 * 10**9,
        num_layers=128,
        layer_recompute=True,
    )
    print(f"Simulated MILP BFSPP runtime: {sim.get_milp_sol_runtime()}")
    print(f"Equation estimated BFSPP runtime: {sim.get_simulated_runtime()}")
    print(f"Total computation time: {sim.get_total_computation_time_per_device()}")


def test_simulator():
    sim = SimCfgGen(
        llama_model_size=405,
        seq_len=8196,
        mbs=1,
        tp=8,
        pp=8,
        dp=64,
        num_mb_per_pp_stage=2,
        num_chunks=1,
        gpu_mem_bytes=88 * 1024**3,  # around 96GB
        gpu_avg_perf_flops=350 * 10**12,
        num_DC=2,
        DC_comm_latency=0.01,
        DC_comm_bandwidth=32 * 10**9,
        num_layers=128,
    )

    sys_config = sim.get_system_config()
    print(sys_config)
    sim.print_stats()
    dg = UnidirectionalZBDependencyGraph(sys_config)
    dg.build_ilp()
    dg.solve_ilp(time_limit=20, warm_start=False)
    schedule = dg.get_schedule()

    azb = AutoUDZBPipeline(sys_config)
    azb.schedule(schedule)
    azb.solve_dependencies()
    azb.print_debug_schedule(verbose=1)
    azb.print_schedule()


def test_simulator_wave():
    sim = SimCfgGen(
        llama_model_size=405,
        seq_len=8196,
        mbs=1,
        tp=8,
        pp=4,
        dp=64,
        num_mb_per_pp_stage=4,
        num_chunks=2,
        gpu_mem_bytes=88 * 1024**3,  # around 96GB
        gpu_avg_perf_flops=350 * 10**12,
        num_DC=2,
        DC_comm_latency=0.01,
        DC_comm_bandwidth=32 * 10**9,
        num_layers=128,
    )

    sys_config = sim.get_system_config()
    print(sys_config)
    sim.print_stats()
    dg = WaveLikeZBDependencyGraph(sys_config)
    dg.build_ilp()
    dg.solve_ilp(time_limit=20, warm_start=False)
    schedule = dg.get_schedule()

    azb = AutoWaveZBPipeline(sys_config)
    azb.schedule(schedule)
    azb.solve_dependencies()
    azb.print_debug_schedule(verbose=1)
    azb.print_schedule()


def test_pipeline(
    PipelineClass: Type[Pipeline],
    sys_config: SystemConfig,
    upper_limit: int = -1,
    verbose: int = 0,
) -> None:
    pipeline = PipelineClass(sys_config)
    pipeline.schedule()
    pipeline.solve_dependencies()
    if upper_limit > 0:
        assert (
            pipeline.get_schedule_time() <= upper_limit
        ), f"Pipeline {PipelineClass.__name__} runtime: {pipeline.get_schedule_time()}, upper limit: {upper_limit}, sys_config: {sys_config}"
    if verbose > 0:
        pipeline.print_debug_schedule(verbose=1)
    else:
        print(f"{PipelineClass.__name__} runtime: {pipeline.get_schedule_time()}")
    
    pipeline.print_schedule(save=True)


def test_basic_schedule():
    num_dev = 4
    num_microbatches = 8
    T_F = 200
    T_B = 200
    T_W = 200
    num_chunks = 2
    T_F_chunk = T_F / num_chunks
    T_B_chunk = T_B / num_chunks
    T_W_chunk = T_W / num_chunks

    comm_matrix = generate_comm_mat(1, num_dev, 0, 0)
    # gpipe, 1f1b,
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=num_microbatches,
        T_F=T_F,
        T_B=T_B + T_W,
        T_C=comm_matrix,
    )
    # iv1f1b, hanayo
    interleaved_sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=num_microbatches,
        T_F=T_F_chunk,
        T_B=T_B_chunk + T_W_chunk,
        T_C=comm_matrix,
        num_chunks=num_chunks,
    )

    zbh1_sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=num_microbatches,
        T_F=T_F,
        T_B=T_B,
        T_W=T_W,
        T_C=comm_matrix,
    )
    heur_zbv_sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=num_microbatches,
        T_F=T_F_chunk,
        T_B=T_B_chunk,
        T_W=T_W_chunk,
        T_C=comm_matrix,
        num_chunks=num_chunks,
    )
    test_pipeline(OneFOneBPipeline, sys_config, 6600)
    test_pipeline(GpipePipeline, sys_config, 6600)
    test_pipeline(Interleaved1F1BPipeline, interleaved_sys_config, 5700)
    test_pipeline(Hanayo1F1BPipeline, interleaved_sys_config, 6500)
    test_pipeline(ZBH1Pipeline, zbh1_sys_config, 5400)
    test_pipeline(HeuristicWaveZBPipeline, heur_zbv_sys_config, 5100)
    test_pipeline(HeuristicZBVPipeline, heur_zbv_sys_config)
    test_pipeline(HeuristicWaveZBPipelineV2, heur_zbv_sys_config)

def test_heuristic_zb_v2():    
    sys_config = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        T_F=200,
        T_B=200,
        T_W=200,
        T_C=generate_comm_mat(1, 4, 5, 0),
        num_chunks=2,
        M_F=2,
        M_B=-1,
        M_W=-1,
        M_Limit=16,
    )
    pp = HeuristicWaveZBPipelineV2(sys_config)
    pp.schedule()
    pp.solve_dependencies()
    pp.print_schedule(save=True)

def test_heuristic_ud():
    ud_cfg = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        T_F=200,
        T_B=400,
        T_W=0,
        T_C=generate_comm_mat(2, 2, 0, 800),
        M_F=2,
        M_B=-2,
        M_W=0,
        M_Limit=8)
    
    
    udzb_cfg = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        T_F=200,
        T_B=200,
        T_W=200,
        T_C=generate_comm_mat(2, 2, 0, 800),
        M_F=2,
        M_B=-1,
        M_W=-1,
        M_Limit=8,
    )
    udpp = HeuristicUDPipeline(ud_cfg)
    udpp.schedule()
    udpp.solve_dependencies()
    udpp.print_schedule(save=True)
    udzbpp = HeuristicZBUDPipeline(udzb_cfg)
    udzbpp.schedule()
    udzbpp.solve_dependencies()
    udzbpp.print_schedule(save=True)


def test_execution_planner_0():
    sys_config = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        T_F=200,
        T_B=400,
        T_W=0,
        T_C=generate_comm_mat(1, 4, 50, 0),
    )
    pp = OneFOneBPipeline(sys_config)
    pp.schedule()
    pp.solve_dependencies()
    planner = ExecutionPlanner(pp)
    planner.generate_execution_plan()
    planner.print_execution_plan()

def test_execution_planner_1():
    sys_config = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        T_F=200,
        T_B=200,
        T_W=200,
        T_C=generate_comm_mat(1, 4, 50, 0),
    )
    pp = ZBH1Pipeline(sys_config)
    pp.schedule()
    pp.solve_dependencies()
    planner = ExecutionPlanner(pp)
    planner.generate_execution_plan()
    planner.print_execution_plan()

def test_execution_planner_2():
    sys_config = SystemConfig(
        num_devices=4,
        num_microbatches=8,
        T_F=200,
        T_B=200,
        T_W=200,
        T_C=generate_comm_mat(1, 4, 50, 0),
        num_chunks=2,
        M_F=2,
        M_B=-1,
        M_W=-1,
        M_Limit=16,
    )
    pp = HeuristicZBVPipeline(sys_config)
    pp.schedule()
    pp.solve_dependencies()
    pp.print_schedule(save=True)
    planner = ExecutionPlanner(pp)
    planner.generate_execution_plan()
    planner.print_execution_plan()
    
if __name__ == "__main__":
    # test_basic_schedule()
    # test_ud_zb()
    # test_wave_zb()
    # test_wave_zb(relax=False)
    # test_schedule_store()
    # test_bfs_simulator()
    # test_simulator()
    # test_simulator_wave()
    # test_heuristic_zb_v2()
    # test_heuristic_ud()
    
    # test_execution_planner_0()
    # test_execution_planner_1()
    test_execution_planner_2()
    
