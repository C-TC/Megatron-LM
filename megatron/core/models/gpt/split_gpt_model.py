import torch
from typing import List, Optional, Tuple, Literal

from torch import Tensor, nn
from megatron.core import InferenceParams, tensor_parallel 
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_viewless_tensor
from contextlib import nullcontext
from .gpt_model import GPTModel

class SplitGPTModel(GPTModel):
    """GPT Model with manual split control for forward and backward passes."""
    
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        # Initialize parent GPT model
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )
        from megatron.training import get_args
        args = get_args()
        num_splits = args.num_subparts
        assert args.dynamic_schedule == "subud", "SplitGPTModel requires dynamic_schedule=subud"
        assert args.head_tail_as_one_layer, "SplitGPTModel requires head_tail_as_one_layer=True"
        self.num_splits = num_splits
        self.num_microbatches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
        
        self.num_splits = num_splits
        self._setup_split_execution()
        
    def _setup_split_execution(self):
        """Setup split execution state and layer grouping"""
        # Calculate total layers including pre/post process
        self.all_layers = []
        
        if self.pre_process:
            self.all_layers.append(('embedding', self.embedding))
            # if self.position_embedding_type == 'rope':
            #     self.all_layers.append(('rotary', self.rotary_pos_emb))
        
        # Add decoder layers
        for idx, layer in enumerate(self.decoder.layers):
            self.all_layers.append((f'decoder_{idx}', layer))
            
        if self.post_process:
            self.all_layers.append(('output', self.output_layer))
            
        # Validate and create splits
        total_layers = len(self.all_layers)
        if total_layers < self.num_splits:
            raise ValueError(f"Cannot split {total_layers} layers into {self.num_splits} parts")
        
        # Create splits with equal distribution
        self.splits = self._create_splits(total_layers, self.num_splits)
        
        # Initialize states for split execution
        self.reset_execution_state()
        
    def _create_splits(self, total_layers: int, num_splits: int) -> List[List[Tuple[str, nn.Module]]]:
        """Create equal splits of layers"""
        splits = []
        base_size = total_layers // num_splits
        extra = total_layers % num_splits
        
        start = 0
        for i in range(num_splits):
            size = base_size + (1 if i < extra else 0)
            end = start + size
            splits.append(self.all_layers[start:end])
            start = end
            
        return splits
        
    def reset_execution_state(self):
        """Reset the model's execution state for a new forward-backward cycle"""
        self.activations = [[None] * self.num_splits for _ in range(self.num_microbatches)]
        self.input_tensors = [[None] * self.num_splits for _ in range(self.num_microbatches)]
        self.attention_masks = [None] * self.num_microbatches
        self.current_inputs = [None] * self.num_microbatches
        self.labels = [None] * self.num_microbatches
        self.last_forward_idx = [-1] * self.num_microbatches
        self.last_backward_idx = [self.num_splits] * self.num_microbatches
        self.stored_rotary_pos_embs = [None] * self.num_microbatches

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, rotary_pos_emb
            ):
                for index in range(start, end):
                    layer = self.decoder._get_layer(index)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.decoder.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.decoder.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def forward_split(
        self, 
        subpart_idx: int,
        microbatch_idx: int,
        input_dict: Optional[dict] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None
    ):
        """Forward pass through specified subpart"""
        # Validate indices and execution order
        if not 0 <= subpart_idx < self.num_splits:
            raise ValueError(f"Subpart index must be between 0 and {self.num_splits-1}")
            
        if not 0 <= microbatch_idx < self.num_microbatches:
            raise ValueError(f"Microbatch index must be between 0 and {self.num_microbatches-1}")
            
        if subpart_idx != self.last_forward_idx[microbatch_idx] + 1:
            raise ValueError(
                f"Must execute forward passes in order for each microbatch. "
                f"Expected {self.last_forward_idx[microbatch_idx] + 1} for microbatch {microbatch_idx}"
            )
        
        
        # Handle first subpart input
        if subpart_idx == 0:
            if input_dict is None:
                raise ValueError("input_dict required for first subpart")
            self.current_inputs[microbatch_idx] = input_dict['input_ids']
            self.attention_masks[microbatch_idx] = input_dict['attention_mask']
            self.labels[microbatch_idx] = input_dict.get('labels')
        else:
            # print(f'current {subpart_idx=}: {self.splits[subpart_idx]}')
            # print(f'current activations: {self.activations[microbatch_idx]}')
            self.current_inputs[microbatch_idx] = self.activations[microbatch_idx][subpart_idx - 1].detach().requires_grad_()
        
        self.input_tensors[microbatch_idx][subpart_idx] = self.current_inputs[microbatch_idx]
        
        x = self.current_inputs[microbatch_idx]
        # Process each layer in the split without checkpointing
        for layer_name, layer in self.splits[subpart_idx]:
            if layer_name == 'embedding':
                x = layer(
                    input_ids=input_dict['input_ids'],
                    position_ids=input_dict['position_ids']
                )
                if self.position_embedding_type == 'rope':
                    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                        inference_params,
                        self.decoder,
                        x,
                        self.config
                    )
                    self.stored_rotary_pos_embs[microbatch_idx] = self.rotary_pos_emb(rotary_seq_len)
            elif layer_name.startswith('decoder_'):
                if layer_name == 'decoder_0':
                    if not self.decoder.pre_process:
                        x = self.decoder.input_tensor
                x = make_viewless_tensor(inp=x, requires_grad=True, keep_graph=True)
                if self.config.recompute_granularity == 'full' and self.training:
                    x = self._checkpointed_forward(
                        hidden_states=x,
                        attention_mask=self.attention_masks[microbatch_idx],
                        context=None,
                        context_mask=None,
                        rotary_pos_emb=self.stored_rotary_pos_embs[microbatch_idx],
                        packed_seq_params=packed_seq_params
                    )
                else:
                    x, _ = layer(
                        hidden_states=x,
                        attention_mask=self.attention_masks[microbatch_idx],
                        rotary_pos_emb=self.stored_rotary_pos_embs[microbatch_idx],
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params
                    )
                if layer_name == f'decoder_{len(self.decoder.layers) - 1}':
                    if self.decoder.final_layernorm is not None:
                        x = self.decoder.final_layernorm(x)
                        x = make_viewless_tensor(inp=x, requires_grad=True, keep_graph=True)
                    
            elif layer_name == 'output':
                output_weight = None
                if self.share_embeddings_and_output_weights:
                    output_weight = self.shared_embedding_or_output_weight()
                x, _ = layer(x, weight=output_weight)
                if self.labels[microbatch_idx] is None:
                    x = x.transpose(0, 1).contiguous()
                else:
                    x = self.compute_language_model_loss(labels=self.labels[microbatch_idx], logits=x)
        
        self.activations[microbatch_idx][subpart_idx] = x
        self.last_forward_idx[microbatch_idx] = subpart_idx
        
        return x
        
    def backward_split(self, subpart_idx: int, microbatch_idx: int, grad_output: Optional[torch.Tensor] = None, output: Optional[torch.Tensor] = None):
        """Backward pass through specified subpart for given microbatch"""
        if not 0 <= subpart_idx < self.num_splits:
            raise ValueError(f"Subpart index must be between 0 and {self.num_splits-1}")
            
        if not 0 <= microbatch_idx < self.num_microbatches:
            raise ValueError(f"Microbatch index must be between 0 and {self.num_microbatches-1}")
            
        if subpart_idx != self.last_backward_idx[microbatch_idx] - 1:
            raise ValueError(
                f"Must execute backward passes in reverse order for each microbatch. "
                f"Expected {self.last_backward_idx[microbatch_idx] - 1} for microbatch {microbatch_idx}"
            )

        if subpart_idx == self.num_splits - 1:
            gradient = grad_output
            output_tensor = output
        else:
            gradient = self.input_tensors[microbatch_idx][subpart_idx + 1].grad
            output_tensor = self.activations[microbatch_idx][subpart_idx]
        
        # from megatron.core import parallel_state
        # if parallel_state.is_pipeline_last_stage():
        #     print(f'gradient: {gradient}')
        #     print(f'output_tensor: {output_tensor}')
        torch.autograd.backward(output_tensor, gradient)
        
        # Clear memory for tensors that are no longer needed
        # Clear activation from the next layer since we've used its gradient
        if subpart_idx < self.num_splits - 1:
            self.activations[microbatch_idx][subpart_idx + 1] = None
            
        # Clear input tensor from the next layer
        if subpart_idx < self.num_splits - 1:
            self.input_tensors[microbatch_idx][subpart_idx + 1] = None
            
        # If this is the last backward pass for this microbatch, clear remaining tensors
        if subpart_idx == 0:
            self.activations[microbatch_idx].clear()
            self.attention_masks[microbatch_idx] = None
            self.labels[microbatch_idx] = None
            self.input_tensors[microbatch_idx].clear()
            self.stored_rotary_pos_embs[microbatch_idx] = None
            self.current_inputs[microbatch_idx] = None
            
        self.last_backward_idx[microbatch_idx] = subpart_idx
        
        if microbatch_idx == self.num_microbatches - 1 and subpart_idx == 0:
            self.reset_execution_state()
        
        return None
        
    def _get_fp8_context(self):
        return nullcontext()
        
    def forward(self, *args, **kwargs):
        """Override parent's forward to explain split usage"""
        raise NotImplementedError(
            "This is a split model - use forward_split() and backward_split() methods instead of forward(). "
            "Example usage with microbatches:\n"
            "    # Forward passes for microbatch 0\n"
            "    h1 = model.forward_split(0, 0, input_dict)\n"
            "    h2 = model.forward_split(1, 0)\n"
            "    h3 = model.forward_split(2, 0)\n"
            "    out = model.forward_split(3, 0)\n"
            "    # Backward passes for microbatch 0\n"
            "    model.backward_split(3, 0, grad_output)\n"
            "    model.backward_split(2, 0)\n"
            "    model.backward_split(1, 0)\n"
            "    model.backward_split(0, 0)\n"
            "    # Repeat for other microbatches..."
        )
