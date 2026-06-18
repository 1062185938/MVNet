from typing import Dict, Optional

import torch
from torch import nn

from .descriptors import compute_structure_descriptors


MODEL_NAMES = ("iq_cnn", "ap_cnn", "fft_cnn", "concat", "vanilla_gate", "ssg_gate")


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScoreMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(8, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleViewCNN(nn.Module):
    def __init__(
        self, view: str, feature_dim: int = 64, num_classes: int = 11, dropout: float = 0.3
    ) -> None:
        super().__init__()
        if view not in ("iq", "ap", "fft"):
            raise ValueError(f"Unknown single view '{view}'.")
        self.view = view
        in_channels = 1 if view == "fft" else 2
        self.branch = ConvBranch(in_channels, feature_dim=feature_dim)
        self.classifier = ClassifierHead(feature_dim, feature_dim, num_classes, dropout)

    def forward(
        self, iq: torch.Tensor, ap: torch.Tensor, fft: torch.Tensor, return_aux: bool = False
    ):
        x = {"iq": iq, "ap": ap, "fft": fft}[self.view]
        z = self.branch(x)
        logits = self.classifier(z)
        if return_aux:
            return {"logits": logits, "features": {self.view: z}}
        return logits


class ConcatFusionCNN(nn.Module):
    def __init__(self, feature_dim: int = 64, num_classes: int = 11, dropout: float = 0.3) -> None:
        super().__init__()
        self.iq_branch = ConvBranch(2, feature_dim=feature_dim)
        self.ap_branch = ConvBranch(2, feature_dim=feature_dim)
        self.fft_branch = ConvBranch(1, feature_dim=feature_dim)
        self.classifier = ClassifierHead(feature_dim * 3, feature_dim, num_classes, dropout)

    def forward(
        self, iq: torch.Tensor, ap: torch.Tensor, fft: torch.Tensor, return_aux: bool = False
    ):
        z_iq = self.iq_branch(iq)
        z_ap = self.ap_branch(ap)
        z_fft = self.fft_branch(fft)
        z_concat = torch.cat([z_iq, z_ap, z_fft], dim=1)
        logits = self.classifier(z_concat)
        if return_aux:
            return {
                "logits": logits,
                "features": {"iq": z_iq, "ap": z_ap, "fft": z_fft, "concat": z_concat},
            }
        return logits


class VanillaGateCNN(nn.Module):
    def __init__(self, feature_dim: int = 64, num_classes: int = 11, dropout: float = 0.3) -> None:
        super().__init__()
        self.iq_branch = ConvBranch(2, feature_dim=feature_dim)
        self.ap_branch = ConvBranch(2, feature_dim=feature_dim)
        self.fft_branch = ConvBranch(1, feature_dim=feature_dim)
        self.feat_score_iq = ScoreMLP(feature_dim)
        self.feat_score_ap = ScoreMLP(feature_dim)
        self.feat_score_fft = ScoreMLP(feature_dim)
        self.classifier = ClassifierHead(feature_dim, feature_dim, num_classes, dropout)

    def _encode(self, iq: torch.Tensor, ap: torch.Tensor, fft: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "iq": self.iq_branch(iq),
            "ap": self.ap_branch(ap),
            "fft": self.fft_branch(fft),
        }

    def _scores(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                self.feat_score_iq(features["iq"]),
                self.feat_score_ap(features["ap"]),
                self.feat_score_fft(features["fft"]),
            ],
            dim=1,
        )

    @staticmethod
    def _fuse(features: Dict[str, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        return (
            weights[:, 0:1] * features["iq"]
            + weights[:, 1:2] * features["ap"]
            + weights[:, 2:3] * features["fft"]
        )

    def forward(
        self, iq: torch.Tensor, ap: torch.Tensor, fft: torch.Tensor, return_aux: bool = False
    ):
        features = self._encode(iq, ap, fft)
        weights = torch.softmax(self._scores(features), dim=1)
        z_fused = self._fuse(features, weights)
        logits = self.classifier(z_fused)
        if return_aux:
            return {
                "logits": logits,
                "gate_weights": weights,
                "features": features,
                "fused": z_fused,
            }
        return logits


class SignalStructureGuidedGateCNN(VanillaGateCNN):
    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 11,
        dropout: float = 0.3,
        fft_shift: bool = True,
    ) -> None:
        super().__init__(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
        self.fft_shift = fft_shift
        self.struct_score_iq = ScoreMLP(2, hidden_dim=8)
        self.struct_score_ap = ScoreMLP(3, hidden_dim=12)
        self.struct_score_fft = ScoreMLP(3, hidden_dim=12)

    def _scores_with_structure(
        self, features: Dict[str, torch.Tensor], descriptors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        score_iq = self.feat_score_iq(features["iq"]) + self.struct_score_iq(descriptors["iq"])
        score_ap = self.feat_score_ap(features["ap"]) + self.struct_score_ap(descriptors["ap"])
        score_fft = self.feat_score_fft(features["fft"]) + self.struct_score_fft(descriptors["fft"])
        return torch.cat([score_iq, score_ap, score_fft], dim=1)

    def forward(
        self, iq: torch.Tensor, ap: torch.Tensor, fft: torch.Tensor, return_aux: bool = False
    ):
        features = self._encode(iq, ap, fft)
        descriptors = compute_structure_descriptors(iq, fft_shift=self.fft_shift)
        weights = torch.softmax(self._scores_with_structure(features, descriptors), dim=1)
        z_fused = self._fuse(features, weights)
        logits = self.classifier(z_fused)
        if return_aux:
            return {
                "logits": logits,
                "gate_weights": weights,
                "features": features,
                "descriptors": descriptors,
                "fused": z_fused,
            }
        return logits


def build_model(
    model_name: str,
    feature_dim: int = 64,
    num_classes: int = 11,
    dropout: float = 0.3,
    fft_shift: bool = True,
) -> nn.Module:
    if model_name == "iq_cnn":
        return SingleViewCNN("iq", feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    if model_name == "ap_cnn":
        return SingleViewCNN("ap", feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    if model_name == "fft_cnn":
        return SingleViewCNN("fft", feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    if model_name == "concat":
        return ConcatFusionCNN(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    if model_name == "vanilla_gate":
        return VanillaGateCNN(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout)
    if model_name == "ssg_gate":
        return SignalStructureGuidedGateCNN(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            fft_shift=fft_shift,
        )
    raise ValueError(f"Unknown model '{model_name}'. Use one of {MODEL_NAMES}.")
