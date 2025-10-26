# Sample ADL Extraction Model
# Depth-only, ADMarker-like lightweight 8-layer 3D-CNN -> structured activity sequences.
# Reference Results from ADMarker (https://dl.acm.org/doi/pdf/10.1145/3636534.3649370), for the accuracy of biomarker (ADLs) detection, the mean accuracy is over 94.51% for AD subjects, while 87.83% and 91.67% for cognitively normal and MCI.
# Input : (B, 1, T, H, W)
# Output: dict(feat_seq: (B, S, D), logits_t: (B, S, C)|None, logits_clip: (B, C)|None)

from typing import Optional, Dict
import torch
import torch.nn as nn


# ---------------------- utils ----------------------
def init_weights_kaiming(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ConvBNReLU3D(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------- backbone ----------------------
class Backbone3D_8L(nn.Module):
    """
    8-layer 3D-CNN backbone (ADMarker-like):
      Stage1: 2× Conv3d(BN,ReLU) + Pool(1x2x2)   # keep temporal resolution
      Stage2: 2× Conv3d(BN,ReLU) + Pool(2x2x2)   # T / 2
      Stage3: 2× Conv3d(BN,ReLU) + Pool(2x2x2)   # T / 4
      Stage4: 2× Conv3d(BN,ReLU) + Pool(1x2x2)   # spatial only
    Output: (B, C4, T', H', W')
    """
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        C1, C2, C3, C4 = base, base * 2, base * 4, base * 8

        # Stage 1
        self.conv1 = ConvBNReLU3D(in_ch, C1)
        self.conv2 = ConvBNReLU3D(C1, C1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Stage 2
        self.conv3 = ConvBNReLU3D(C1, C2)
        self.conv4 = ConvBNReLU3D(C2, C2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Stage 3
        self.conv5 = ConvBNReLU3D(C2, C3)
        self.conv6 = ConvBNReLU3D(C3, C3)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Stage 4
        self.conv7 = ConvBNReLU3D(C3, C4)
        self.conv8 = ConvBNReLU3D(C4, C4)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.out_channels = C4
        init_weights_kaiming(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, H, W)
        x = self.conv1(x); x = self.conv2(x); x = self.pool1(x)
        x = self.conv3(x); x = self.conv4(x); x = self.pool2(x)
        x = self.conv5(x); x = self.conv6(x); x = self.pool3(x)
        x = self.conv7(x); x = self.conv8(x); x = self.pool4(x)
        return x  # (B, C4, T', H', W')


# ---------------------- heads ----------------------
class SequenceHead(nn.Module):
    """
    Global average over spatial -> (B, C, T') -> (B, T', C) -> Linear -> (B, T', D)
    """
    def __init__(self, in_ch: int, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_ch, embed_dim)
        init_weights_kaiming(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T', H', W')
        x = x.mean(dim=(3, 4))                 # (B, C, T')
        x = x.permute(0, 2, 1).contiguous()    # (B, T', C)
        x = self.proj(x)                       # (B, T', D)
        return x


class TemporalClassifier(nn.Module):
    """Per-timestep classification: (B, T', D) -> (B, T', C)"""
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        init_weights_kaiming(self)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.fc(seq)


class ClipClassifier(nn.Module):
    """Clip-level classification by temporal average."""
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        init_weights_kaiming(self)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        feat = seq.mean(dim=1)  # (B, D)
        return self.fc(feat)


# ---------------------- model ----------------------
class ADLDepthNet(nn.Module):
    """
    Depth-only, ADMarker-like model:
      - 8-layer 3D-CNN backbone
      - Sequence head -> structured activity sequences (feat_seq)
      - Optional temporal / clip classifier heads
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        embed_dim: int = 128,
        num_classes: Optional[int] = None,
        with_clip_head: bool = True,
    ):
        super().__init__()
        self.backbone = Backbone3D_8L(in_ch=in_channels, base=base_channels)
        self.seq_head = SequenceHead(self.backbone.out_channels, embed_dim)
        self.num_classes = num_classes

        self.t_head = TemporalClassifier(embed_dim, num_classes) if num_classes is not None else None
        self.c_head = ClipClassifier(embed_dim, num_classes) if (num_classes is not None and with_clip_head) else None

    @torch.no_grad()
    def load_from(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """Convenience loader, ADMarker-style."""
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        # x: (B, 1, T, H, W)
        x = self.backbone(x)             # (B, C4, T', H', W')
        feat_seq = self.seq_head(x)      # (B, T', D)

        logits_t = self.t_head(feat_seq) if self.t_head is not None else None
        logits_clip = self.c_head(feat_seq) if self.c_head is not None else None

        return {"feat_seq": feat_seq, "logits_t": logits_t, "logits_clip": logits_clip}


# ---------------------- quick test ----------------------
if __name__ == "__main__":
    B, T, H, W = 2, 16, 112, 112
    dummy = torch.randn(B, 1, T, H, W)

    # embeddings only
    net = ADLDepthNet(num_classes=None)
    out = net(dummy)
    print("feat_seq:", tuple(out["feat_seq"].shape))  # (B, S, D)

    # with classification heads
    net_cls = ADLDepthNet(num_classes=8, with_clip_head=True)
    out2 = net_cls(dummy)
    print("logits_t:", None if out2["logits_t"] is None else tuple(out2["logits_t"].shape))      # (B, S, 8)
    print("logits_clip:", None if out2["logits_clip"] is None else tuple(out2["logits_clip"].shape))  # (B, 8)
