import numbers
import torch

from megatron.core.transformer.transformer_config import TransformerConfig


@torch.compile
class CompiledRMSNorm(torch.nn.Module):
    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.config = config
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(*hidden_size))
        self.reset_parameters()
        self.sequence_parallel = config.sequence_parallel

        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight
