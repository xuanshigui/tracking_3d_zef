import math
import torch
from torch import nn

class PositionEmbeddingSine(nn.Module):
    """
        This is a different version of the standard position embedding.
    """

    def __init__(self, num_pos_feats=64, num_frames=2, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.frames = num_frames

    def forward(self, pre_tgt):
        # pre_tgt b, n, 256
        b, np, c = pre_tgt.shape
        # get mask
        mask = torch.ones((b, np, c), dtype=torch.bool)
        for pre, m in zip(pre_tgt, mask):
            m[: pre.shape[1], :pre.shape[2]] = False
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 2, 1)
        return pos

def build_position_encoding(args):
    sine_emedding_func = PositionEmbeddingSine
    position_embedding = sine_emedding_func(64, normalize=True)