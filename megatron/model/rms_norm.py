import numbers
import torch


class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-8, sequence_parallel=False):
        super(RMSNorm, self).__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.sequence_parallel = sequence_parallel
        
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)

        self.weight = torch.nn.Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        
    @torch.compile
    def forward(self, input):
        rms = torch.sqrt(torch.mean(input**2, -1, keepdim=True) + self.eps)
        x = input / rms
        return self.weight * x
            