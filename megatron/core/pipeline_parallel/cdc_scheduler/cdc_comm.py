import threading
import time
import torch
from torch import Tensor
from torch.distributed import ProcessGroupNCCL
import torch.distributed as dist
from torch._C._distributed_c10d import OpType
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.cdc_scheduler.pp_scheduler import get_cdc_pp_scheduler


'''
These two global variables might be prone to error?
The work.seq_ should correspond to seqp2p_ in ProcessGroupNCCL.cpp?
Best way is to expose work.getSequencenumber() and recompile torch?
'''
_CURRENT_SEQ_NUM = -1
_DELAY_WORKDICT = {}
_DICT_LOCK = threading.Lock()


_CDC_ProcessGroupNCCL = None # Only at most one cdc group per device.

'''
isend/irecv with latency injection

same api as torch.distributed.isend/irecv

rely on _register_on_completion_hook

isend is not modified
'''

def precise_stall(duration: float):
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < duration:
        pass

def cdc_comm_completion_hook(work_info: torch._C._distributed_c10d.WorkInfo):
    ''' usage: pg._register_on_completion_hook(hook)'''
    duration: float = work_info.active_duration.total_seconds()
    # ! Duration does not reflect send/recv time since recv is posted earlier.
    op_type: OpType = work_info.op_type
    seq_num: int = work_info.seq
    if op_type != OpType.RECV:
        return
    
    delay_in_seconds = get_cdc_pp_scheduler().get_cdc_recv_delay()
    precise_stall(delay_in_seconds)
    
    with _DICT_LOCK:
        _DELAY_WORKDICT[seq_num] = True

class CDCWork:
    def __init__(self, work: torch._C._distributed_c10d.Work, seq_num: int):
        self.work = work
        self.seq_num = seq_num
    
    def is_completed(self):
        raise NotImplementedError
    
    def is_success(self):
        raise NotImplementedError
    
    def exception(self):
        raise NotImplementedError
    
    def wait(self, timeout: float=100):
        # check if the seq_num is in _DELAY_WORKDICT, but within timeout
        # if time out, raise exception
        # if not, wait for the work to complete
        start_time = time.perf_counter()
        
        while True:
            with _DICT_LOCK:
                if self.seq_num in _DELAY_WORKDICT:
                    del _DELAY_WORKDICT[self.seq_num]
                    break
            if time.perf_counter() - start_time > timeout:
                raise TimeoutError("CDC Work timeout when waiting for hook completion.")
            # TODO: reduce the sleep time?
            precise_stall(0.0005)
        
        return
        
    
    def source_rank(self):
        raise NotImplementedError
    
    def _source_rank(self):
        raise NotImplementedError
    
    def result(self):
        raise NotImplementedError
    
    def synchronize(self):
        raise NotImplementedError
    
    def boxed(self):
        raise NotImplementedError
    
    @staticmethod
    def unboxed(self):
        raise NotImplementedError

def irecv(tensor: Tensor, dst: int, group: ProcessGroupNCCL | None = None):
    '''
    IMPORTANT: cdc recv process group should only use this irecv! 
    and only cdc recv can use this irecv!
    To make sure the global seq counter matches the actual seq number.
    Notice: current limitation, only one cdc group (either from prev or next rank) per rank.
    Since we don't have enough info in the hook to distinguish between PGs.
    '''
    global _CDC_ProcessGroupNCCL
    if _CDC_ProcessGroupNCCL is None:
        recv_prev_pg = parallel_state.get_pipeline_extra_recv_prev_group()
        recv_next_pg = parallel_state.get_pipeline_extra_recv_next_group()
        assert group == recv_prev_pg or group == recv_next_pg
        _CDC_ProcessGroupNCCL = group
    else:
        assert group == _CDC_ProcessGroupNCCL, "Only one cdc group per rank is allowed."
    
    global _CURRENT_SEQ_NUM
    _CURRENT_SEQ_NUM += 1
    work = dist.irecv(tensor, dst, group)
    return CDCWork(work, _CURRENT_SEQ_NUM)
