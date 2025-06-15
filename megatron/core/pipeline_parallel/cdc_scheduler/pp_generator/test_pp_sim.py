from typing import Type

from .pipeline_config import SystemConfig
from .auto_schedule import UnidirectionalZBDependencyGraph, WaveLikeZBDependencyGraph
from .pipeline import (
    AutoZBUDPipeline,
    AutoWaveZBPipeline,
    CPZBLoopPipeline,
    CPZBUDPipeline,
    CPZBWavePipeline,
    GpipePipeline,
    HeuristicZBVPipeline,
    Interleaved1F1BPipeline,
    OneFOneBPipeline,
    Pipeline,
    ZBH1Pipeline,
)
from .subpipeline import DynZBUDSubPipeline
from .util import generate_comm_mat
from megatron.core.pipeline_parallel.cdc_scheduler.execution_planner import ExecutionPlanner



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
        two_dc=False,
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
    test_pipeline(ZBH1Pipeline, zbh1_sys_config, 5400)
    test_pipeline(HeuristicZBVPipeline, heur_zbv_sys_config)

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
    
def test_cp_ud_auto_solver():
    num_dev = 32
    num_parts = 1
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=2*num_dev,
        T_F=20 * num_parts,
        T_B=25 * num_parts,
        T_W=17 * num_parts,
        T_C=generate_comm_mat(2, num_dev // 2, 0, 30 * num_parts),
        T_beta=generate_comm_mat(2, num_dev // 2, 0, 40 * num_parts),
        M_F=20 * num_parts,
        M_B=-1 * num_parts,
        M_W=-19 * num_parts,
        M_Limit=20*num_parts*num_dev,
        num_chunks=1,
    )
    pp = CPZBUDPipeline(sys_config, warm_start=False, use_cplex=True)
    pp.schedule(logging=True, relative_gap=0.04, time_limit_sec=300)
    pp.solve_dependencies()
    # pp.print_debug_schedule(verbose=1)
    pp.print_schedule(save=True, include_info=num_dev <= 8)
    print(f'Runtime: {pp.get_schedule_time(device_wise=True) / num_parts}, Bubble: {pp.get_bubble_ratio(device_wise=True)}')


def test_cp_wave_auto_solver():
    num_dev = 8
    num_parts = 1
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=2*num_dev,
        T_F=20 * num_parts,
        T_B=25 * num_parts,
        T_W=17 * num_parts,
        T_C=generate_comm_mat(2, num_dev // 2, 0, 5 * num_parts),
        T_beta=generate_comm_mat(2, num_dev // 2, 0, 40 * num_parts),
        M_F=20 * num_parts,
        M_B=-1 * num_parts,
        M_W=-19 * num_parts,
        M_Limit=20*num_parts*2*num_dev,
        num_chunks=2,
    )
    pp = CPZBWavePipeline(sys_config, warm_start=False, use_cplex=True)
    pp.schedule(logging=True, relative_gap=0.01, time_limit_sec=200)
    pp.solve_dependencies()
    # pp.print_debug_schedule(verbose=1)
    pp.print_schedule(save=True, include_info=num_dev <= 8)
    print(f'Runtime: {pp.get_schedule_time(device_wise=True) / num_parts}, Bubble: {pp.get_bubble_ratio(device_wise=True)}')


def test_cp_loop_auto_solver():
    num_dev = 4
    num_parts = 1
    num_dc = 2
    num_dev_per_dc = num_dev // num_dc
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=2*num_dev,
        T_F=20 * num_parts,
        T_B=25 * num_parts,
        T_W=17 * num_parts,
        T_C=generate_comm_mat(num_dc, num_dev_per_dc, 0, 0 * num_parts),
        T_beta=generate_comm_mat(num_dc, num_dev_per_dc, 0, 40 * num_parts),
        M_F=20 * num_parts,
        M_B=-1 * num_parts,
        M_W=-19 * num_parts,
        M_Limit=20*num_parts*2*num_dev,
        num_chunks=2,
        two_dc=True
    )
    pp = CPZBLoopPipeline(sys_config, warm_start=False, use_cplex=True)
    pp.schedule(logging=True, relative_gap=0.01, time_limit_sec=200)
    pp.solve_dependencies()
    # pp.print_debug_schedule(verbose=1)
    pp.print_schedule(save=True, include_info=num_dev <= 8)
    print(f'Runtime: {pp.get_schedule_time(device_wise=True) / num_parts}, Bubble: {pp.get_bubble_ratio(device_wise=True)}')


def test_ud_auto():
    num_dev = 16
    num_parts = 1
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=2*num_dev,
        T_F=20 * num_parts,
        T_B=25 * num_parts,
        T_W=17 * num_parts,
        T_C=generate_comm_mat(2, num_dev // 2, 0, 30 * num_parts),
        T_beta=generate_comm_mat(2, num_dev // 2, 0, 0 * num_parts),
        M_F=20 * num_parts,
        M_B=-1 * num_parts,
        M_W=-19 * num_parts,
        M_Limit=20*num_parts*num_dev,
        num_chunks=1,
    )

    azb = AutoZBUDPipeline(sys_config)
    azb.schedule(verbose=True, time_limit=1200, warm_start=True)
    azb.solve_dependencies()
    azb.print_schedule(save=True, include_info=num_dev <= 8)


def test_wave_auto():
    num_dev = 16
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=2*num_dev,
        T_F=20,
        T_B=25,
        T_W=21,
        T_C=generate_comm_mat(2, num_dev // 2, 0, 57),
        M_F=1,
        M_B=0,
        M_W=-1,
        M_Limit=2 * num_dev,
        num_chunks=2,
    )

    azb = AutoWaveZBPipeline(sys_config)
    azb.schedule(verbose=True, time_limit=1200, warm_start=True)
    azb.solve_dependencies()
    azb.print_schedule(save=True)

def test_subpipe_ud():   
    num_dev = 16
    num_parts = 4
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=2*num_dev,
        T_F=20 * num_parts,
        T_B=25 * num_parts,
        T_W=17 * num_parts,
        T_C=generate_comm_mat(2, num_dev // 2, 0, 30 * num_parts),
        T_beta=generate_comm_mat(2, num_dev // 2, 0, 0 * num_parts),
        M_F=20 * num_parts,
        M_B=-1 * num_parts,
        M_W=-19 * num_parts,
        M_Limit=20*num_parts*num_dev,
        num_chunks=1,
    )
    pp = DynZBUDSubPipeline(sys_config, num_subparts=num_parts)
    pp.schedule()
    # pp.print_debug_schedule(verbose=0)
    pp.solve_dependencies()
    print(f'Runtime: {pp.get_schedule_time(device_wise=True) / num_parts}, Bubble: {pp.get_bubble_ratio(device_wise=True)}')
    pp.print_schedule(save=True, include_info=num_dev <= 8)
    planner = ExecutionPlanner(pp)
    planner.generate_execution_plan()
    print(planner.print_execution_plan())
    

def test_subpipe_ud_recomp():   
    num_dev = 16
    num_parts = 4
    sys_config = SystemConfig(
        num_devices=num_dev,
        num_microbatches=2*num_dev,
        T_F=20 * num_parts,
        T_B=45 * num_parts,
        T_W=17 * num_parts,
        T_C=generate_comm_mat(2, num_dev // 2, 0, 30 * num_parts),
        T_beta=generate_comm_mat(2, num_dev // 2, 0, 0 * num_parts),
        M_F=1 * num_parts,
        M_B=18 * num_parts,
        M_W=-19 * num_parts,
        M_Limit=20*num_parts*num_dev,
        num_chunks=1,
    )
    pp = DynZBUDSubPipeline(sys_config, num_subparts=num_parts)
    pp.schedule()
    # pp.print_debug_schedule(verbose=0)
    pp.solve_dependencies()
    print(f'Runtime: {pp.get_schedule_time(device_wise=True) / num_parts}, Bubble: {pp.get_bubble_ratio(device_wise=True)}')
    pp.print_schedule(save=True, include_info=num_dev <= 8)
    planner = ExecutionPlanner(pp)
    planner.generate_execution_plan()


if __name__ == "__main__":
    test_basic_schedule()
    # test_wave_auto(relax=True)
    # test_schedule_store()
    # test_bfs_simulator()
    # test_simulator()
    # test_simulator_wave()
    # test_heuristic_zb_v2()
    # test_heuristic_ud()
    
    # test_execution_planner_0()
    # test_execution_planner_1()
    # test_execution_planner_2()
    
    # test_cp_ud_auto_solver()
    # test_cp_wave_auto_solver()
    # test_cp_loop_auto_solver()
    # test_ud_auto()
    # test_wave_auto()
    test_subpipe_ud()
    test_subpipe_ud_recomp()
    # test_subpipe_wave()
    
