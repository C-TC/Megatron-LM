try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    raise ImportError("Please install the fast_hadamard_transform package")
import torch
from torch import nn

class HadamardNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, input):
        hid_dim = input.size(-1)
        input = hadamard_transform(input, torch.rsqrt(torch.tensor(hid_dim).float()))
        return self.act(input)

    