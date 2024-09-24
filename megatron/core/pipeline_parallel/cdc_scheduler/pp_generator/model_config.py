from dataclasses import dataclass


@dataclass
class LlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    name: str


LLAMA_7B_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_attention_heads=32,
    num_hidden_layers=32,
    num_key_value_heads=32,
    name="7B",
)

LLAMA_13B_CONFIG = LlamaConfig(
    hidden_size=5120,
    intermediate_size=13824,
    num_attention_heads=40,
    num_hidden_layers=40,
    num_key_value_heads=40,
    name="13B",
)

LLAMA_34B_CONFIG = LlamaConfig(
    hidden_size=8192,
    intermediate_size=22016,
    num_attention_heads=64,
    num_hidden_layers=48,
    num_key_value_heads=8,
    name="34B",
)

LLAMA_70B_CONFIG = LlamaConfig(
    hidden_size=8192,
    intermediate_size=28672,
    num_attention_heads=64,
    num_hidden_layers=80,
    num_key_value_heads=8,
    name="70B",
)

LLAMA_405B_CONFIG = LlamaConfig(
    hidden_size=16384,
    intermediate_size=53248,
    num_attention_heads=128,
    num_hidden_layers=126,
    num_key_value_heads=16,
    name="405B",
)

Custom_400B_CONFIG = LlamaConfig(
    hidden_size=16384,
    intermediate_size=53248,
    num_attention_heads=128,
    num_hidden_layers=128,
    num_key_value_heads=16,
    name="1x400B",
)


Custom_3200B_CONFIG = LlamaConfig(
    hidden_size=32768,
    intermediate_size=106496,
    num_attention_heads=256,
    num_hidden_layers=256,
    num_key_value_heads=32,
    name="8x400B",
)


Custom_25600B_CONFIG = LlamaConfig(
    hidden_size=65536,
    intermediate_size=212992,
    num_attention_heads=512,
    num_hidden_layers=512,
    num_key_value_heads=64,
    name="64x400B",
)

LLAMA_CONFIGS = [
    LLAMA_7B_CONFIG,
    LLAMA_13B_CONFIG,
    LLAMA_34B_CONFIG,
    LLAMA_70B_CONFIG,
    LLAMA_405B_CONFIG,
]
LLAMA_SIZE_TO_CONFIG = {
    7: LLAMA_7B_CONFIG,
    13: LLAMA_13B_CONFIG,
    34: LLAMA_34B_CONFIG,
    70: LLAMA_70B_CONFIG,
    405: LLAMA_405B_CONFIG,
}

CUSTOM_CONFIGS = [
    Custom_400B_CONFIG,
    Custom_3200B_CONFIG,
    Custom_25600B_CONFIG,
]
CUSTOM_SIZE_TO_CONFIG = {
    400: Custom_400B_CONFIG,
    3200: Custom_3200B_CONFIG,
    25600: Custom_25600B_CONFIG,
}
