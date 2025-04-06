from torch import nn
from .vision_transformer.tnt import tnt_s_patch16_224, TNT

class VisionTransformerEncoder(nn.Module):
    def __init__(self, attn_dropout=0.0, model='tnt_s_patch16_224', pretrained=False):
        super(VisionTransformerEncoder, self).__init__()
        if model == 'tnt_s_patch16_224':
            self.model = tnt_s_patch16_224(pretrained=pretrained)
            # Modify dropout if needed
            # if attn_dropout > 0:
            #     for block in self.model.blocks:
            #         block.attn_in.attn_drop.p = attn_dropout
            #         block.attn_out.attn_drop.p = attn_dropout
        else:
            raise ValueError('Unsupported model')

    def forward(self, x):
        # Get the full sequence of patch embeddings including CLS token
        if isinstance(self.model, TNT):
            features = self.model.forward_features(x)
            # Return the full feature sequence: (batch_size, 197, 384)
            return features
        else:
            return self.model(x)