from functools import partial
from torch import nn
from .attention import PerformerAttention
from .modules import (
    groupby_prefix_and_trim,
    equals,
    Residual,
    FeedForward,
    LayerIntermediates
)

class PerformerAttentionLayers(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            causal: bool = False,
            cross_attend: bool = False,
            only_cross: bool = False,
            nb_features: int = None,  # New parameter for Performer
            generalized_attention: bool = False,  # New parameter for Performer
            kernel_fn: callable = None,  # New parameter for Performer
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        
        # Normalization function
        norm_fn = partial(nn.LayerNorm, dim)
        
        # Split kwargs for different components
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)
        
        # Add Performer-specific kwargs
        performer_kwargs = {
            'nb_features': nb_features,
            'generalized_attention': generalized_attention,
            'kernel_fn': kernel_fn
        }
        attn_kwargs.update(performer_kwargs)

        # Determine layer sequence
        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')  # attention, cross attention, feedforward
        elif cross_attend and only_cross:
            default_block = ('c', 'f')       # only cross attention and feedforward
        else:
            default_block = ('a', 'f')       # regular attention and feedforward

        self.layer_types = default_block * depth
        self.num_attn_layers = len(list(filter(equals('a'), self.layer_types)))

        # Create layers
        for layer_type in self.layer_types:
            if layer_type == 'a':
                # Self-attention layer
                layer = PerformerAttention(
                    dim,
                    heads=heads,
                    causal=causal,
                    **attn_kwargs
                )
            elif layer_type == 'c':
                # Cross-attention layer
                layer = PerformerAttention(
                    dim,
                    heads=heads,
                    **attn_kwargs
                )
            elif layer_type == 'f':
                # Feedforward layer
                layer = FeedForward(dim, **ff_kwargs)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            # Wrap with residual connection
            residual_fn = Residual()
            
            # Add normalization, layer, and residual to module list
            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                residual_fn
            ]))

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            return_hiddens=False
    ):
        hiddens = []
        intermediates = []

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)

            residual = x
            x = norm(x)

            if layer_type == 'a':
                # Self-attention
                out, inter = block(x, mask=mask)
            elif layer_type == 'c':
                # Cross-attention
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask)
            elif layer_type == 'f':
                # Feedforward
                out = block(x)
                inter = None

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )
            return x, intermediates

        return x