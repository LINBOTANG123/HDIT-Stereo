# foj_cond_transformer_v2.py

import torch
import torch.nn as nn
from .image_transformer_v2 import ImageTransformerDenoiserModelV2

class FoJCondTransformerV2(ImageTransformerDenoiserModelV2):
    """
    A FoJ denoiser that conditions on an RGB image.
    - x:       the noisy FoJ tensor         [B, C_foj, H, W]
    - sigma:   noise level                  [B]
    - aug_cond: the RGB image for conditioning [B, 3, H, W]
    We encode the image into a vector of length mapping_cond_dim via a small CNN + GAP,
    and feed that into the transformer's mapping network as `mapping_cond`.
    """
    def __init__(self,
                 levels,           # as in the original V2 spec
                 mapping,          # ditto
                 in_channels,      # = C_foj
                 out_channels,     # = C_foj
                 patch_size,       # H==W
                 num_classes=0,    # usually 0
                 mapping_cond_dim=128,  # dim of our image cond vector
                 cond_channels=3,  # RGB
                 **kw):
        super().__init__(levels, mapping, in_channels, out_channels,
                         patch_size, num_classes, mapping_cond_dim, **kw)

        # small encoder that maps [B,3,H,W] → [B, mapping_cond_dim]
        self.image_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, mapping_cond_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mapping_cond_dim, mapping_cond_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → [B, mapping_cond_dim,1,1]
        )

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        # aug_cond is our conditioning image
        if aug_cond is None:
            raise ValueError("FoJCondTransformerV2 requires aug_cond=image tensor")
        # encode image → [B, mapping_cond_dim]
        img_feat = self.image_encoder(aug_cond)                 # [B, D,1,1]
        img_feat = img_feat.view(img_feat.size(0), -1)          # [B, D]
        # discard the built-in aug_cond (we set it to zeros internally)
        return super().forward(
            x,
            sigma,
            aug_cond=None,                # no aug here
            class_cond=class_cond,
            mapping_cond=img_feat,
        )
