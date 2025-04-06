# based on https://github.com/lucidrains/x-transformers

from .layers import PerformerAttentionLayers


class TransformerEncoder(PerformerAttentionLayers):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        nb_features: int = None,
        generalized_attention: bool = False,
        kernel_fn: callable = None,
        **kwargs
    ):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            causal=False,
            nb_features=nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            **kwargs
        )


class TransformerDecoder(PerformerAttentionLayers):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        nb_features: int = None,
        generalized_attention: bool = False,
        kernel_fn: callable = None,
        **kwargs
    ):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            causal=True,
            nb_features=nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            **kwargs
        )


class TransformerCrossAttender(PerformerAttentionLayers):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        nb_features: int = None,
        generalized_attention: bool = False,
        kernel_fn: callable = None,
        **kwargs
    ):
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            cross_attend=True,
            only_cross=True,
            nb_features=nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            **kwargs
        )