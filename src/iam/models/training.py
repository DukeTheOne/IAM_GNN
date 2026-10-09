"""Focal loss, composite RGCN link prediction model, and inductive training loop.

Implements focal binary cross-entropy loss (\\gamma=2.0, \alpha=0.25),
the end-to-end RGCNLinkPredictionModel coupling encoder and bilinear decoder,
AdamW optimization, cosine annealing learning rate scheduling, and early stopping
on validation PR-AUC.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim
from torch_geometric.data import HeteroData

from iam.baselines.metrics import compute_pr_auc, compute_roc_auc
from iam.models.decoder import BilinearRelationPredictor, DecoderConfig
from iam.models.masking import MaskingCondition, StructuredMaskingSuite
from iam.models.rgcn import HeteroRGCNEncoder, RGCNConfig
from iam.models.sampler import StratifiedNegativeSampler


class FocalLoss(nn.Module):
    """Focal Binary Cross-Entropy Loss with logits.

    Formula (Lin et al., 2017):
        p_t = \\sigma(z) if y=1 else 1 - \\sigma(z)
        \alpha_t = \alpha if y=1 else 1 - \alpha
        FL(p_t) = -\alpha_t (1 - p_t)^\\gamma \\log(p_t)

    Mitigates extreme negative link sparsity by down-weighting easily classified
    benign candidate non-edges and focusing learning on hard false negatives.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        reduction: str = "mean",
    ) -> None:
        """Initialize FocalLoss with focusing and balancing hyperparameters."""
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha is not None and not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be one of ('mean', 'sum', 'none'), got '{reduction}'")

        self.gamma: float = gamma
        self.alpha: float | None = alpha
        self.reduction: str = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal binary cross-entropy loss.

        Args:
            logits: Uncalibrated logit predictions [N].
            targets: Binary ground-truth labels in {0, 1} [N].

        Returns:
            loss: Computed focal loss tensor.
        """
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        targets_f = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")

        # p_t = exp(-bce_loss)
        p_t = torch.exp(-bce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets_f * self.alpha + (1.0 - targets_f) * (1.0 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class RGCNLinkPredictionModel(nn.Module):
    """End-to-end multi-relational link prediction model combining RGCN and Bilinear Decoder.

    Executes multi-relational message passing on observed authorization graphs
    and evaluates latent relation probabilities between entity representations.
    """

    def __init__(
        self,
        encoder: HeteroRGCNEncoder | None = None,
        decoder: BilinearRelationPredictor | None = None,
        encoder_config: RGCNConfig | None = None,
        decoder_config: DecoderConfig | None = None,
    ) -> None:
        """Initialize composite link prediction architecture."""
        super().__init__()
        self.encoder: HeteroRGCNEncoder = (
            encoder if encoder is not None else HeteroRGCNEncoder(encoder_config)
        )
        self.decoder: BilinearRelationPredictor = (
            decoder if decoder is not None else BilinearRelationPredictor(decoder_config)
        )

        if self.encoder.config.hidden_channels != self.decoder.config.hidden_channels:
            raise ValueError(
                f"Encoder hidden channels ({self.encoder.config.hidden_channels}) must match "
                f"decoder hidden channels ({self.decoder.config.hidden_channels})."
            )

    def forward(
        self,
        data: HeteroData | None = None,
        x_dict: dict[str, torch.Tensor] | None = None,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute continuous node representations across all entity types."""
        return cast(
            dict[str, torch.Tensor],
            self.encoder(data=data, x_dict=x_dict, edge_index_dict=edge_index_dict),
        )

    def encode(
        self,
        data: HeteroData | None = None,
        x_dict: dict[str, torch.Tensor] | None = None,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Alias for forward: compute continuous node representations across all entity types."""
        return self.forward(data=data, x_dict=x_dict, edge_index_dict=edge_index_dict)

    def score_edge_index(
        self,
        data: HeteroData,
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute link probabilities for candidate edge indices in data."""
        h_dict = self.forward(data)
        return self.decoder.score_edge_index(
            h_dict=h_dict,
            edge_type=edge_type,
            edge_index=edge_index,
            temperature=temperature,
        )

    def score_matrix(
        self,
        data: HeteroData,
        edge_type: tuple[str, str, str],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute all-pairs probability matrix Z in R^{N_src x N_dst}."""
        h_dict = self.forward(data)
        src_type, _, dst_type = edge_type
        rel_id = self.decoder.get_relation_id(edge_type)
        return self.decoder.score_matrix(
            h_src=h_dict[src_type],
            h_dst=h_dict[dst_type],
            rel_id=rel_id,
            temperature=temperature,
        )

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class TrainingConfig:
    """Hyperparameter configuration contracts for InductiveLinkPredictionTrainer."""

    lr: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 50
    patience: int = 10
    focal_gamma: float = 2.0
    focal_alpha: float | None = 0.25
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    negative_ratio: float = 3.0

    def __post_init__(self) -> None:
        """Validate hyperparameter settings."""
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.patience <= 0:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.negative_ratio <= 0.0:
            raise ValueError(f"negative_ratio must be positive, got {self.negative_ratio}")


@dataclass
class TrainingHistory:
    """Records training and validation convergence dynamics across epochs."""

    epoch_train_losses: list[float]
    epoch_val_pr_aucs: list[float]
    epoch_val_roc_aucs: list[float]
    epoch_lrs: list[float]
    best_epoch: int
    best_val_pr_auc: float
    early_stopped: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize training trajectory to dictionary."""
        return {
            "best_epoch": self.best_epoch,
            "best_val_pr_auc": round(self.best_val_pr_auc, 4),
            "early_stopped": self.early_stopped,
            "epochs_completed": len(self.epoch_train_losses),
            "final_train_loss": round(self.epoch_train_losses[-1], 4)
            if self.epoch_train_losses
            else 0.0,
            "final_val_pr_auc": round(self.epoch_val_pr_aucs[-1], 4)
            if self.epoch_val_pr_aucs
            else 0.0,
        }

    def generate_markdown_report(self) -> str:
        """Generate GitHub Flavored Markdown summary report of training convergence."""
        lines: list[str] = [
            "### Inductive Training Convergence Summary",
            "",
            f"- **Best Epoch**: `{self.best_epoch}` with **Validation PR-AUC**: `{self.best_val_pr_auc:.4f}`",
            f"- **Total Epochs Run**: `{len(self.epoch_train_losses)}` / Early Stopped: `{self.early_stopped}`",
            "",
            "| Epoch | Train Focal Loss | Val PR-AUC | Val ROC-AUC | Learning Rate |",
            "| :---: | :---: | :---: | :---: | :---: |",
        ]
        step = max(1, len(self.epoch_train_losses) // 10)
        indices = list(range(0, len(self.epoch_train_losses), step))
        if len(self.epoch_train_losses) - 1 not in indices:
            indices.append(len(self.epoch_train_losses) - 1)

        for ep in indices:
            lines.append(
                f"| `{ep + 1}` | `{self.epoch_train_losses[ep]:.4f}` | "
                f"`{self.epoch_val_pr_aucs[ep]:.4f}` | `{self.epoch_val_roc_aucs[ep]:.4f}` | "
                f"`{self.epoch_lrs[ep]:.2e}` |"
            )

        return "\n".join(lines)


class InductiveLinkPredictionTrainer:
    """Trainer orchestrating inductive link prediction over heterogeneous cloud environments."""

    def __init__(
        self,
        model: RGCNLinkPredictionModel,
        config: TrainingConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize trainer with model, optimizer, scheduler, and loss function."""
        self.config: TrainingConfig = config if config is not None else TrainingConfig()
        self.device: torch.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: RGCNLinkPredictionModel = model.to(self.device)

        self.criterion: FocalLoss = FocalLoss(
            gamma=self.config.focal_gamma,
            alpha=self.config.focal_alpha,
            reduction="mean",
        )

        self.optimizer: optim.AdamW = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler: optim.lr_scheduler.CosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs,
            eta_min=self.config.min_lr,
        )

        self.sampler: StratifiedNegativeSampler = StratifiedNegativeSampler()

    def _extract_supervision_batch(
        self,
        data: HeteroData,
        condition: MaskingCondition,
        seed: int,
    ) -> tuple[HeteroData, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]]:
        """Apply structured masking and extract positive and negative link candidate tensors."""
        mask_res = StructuredMaskingSuite.apply_condition(data=data, condition=condition, seed=seed)
        obs_data = mask_res.observed_data

        supervision: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

        # 1. Collect positive hidden edges and sample corresponding negatives
        if mask_res.hidden_edges:
            for edge_type, pos_edges in mask_res.hidden_edges.items():
                if pos_edges.numel() == 0 or pos_edges.size(1) == 0:
                    continue

                # Sample negative edges for this relation
                num_neg = max(1, int(round(pos_edges.size(1) * self.config.negative_ratio)))
                src_type, _, dst_type = edge_type
                n_src = obs_data[src_type].x.size(0)
                n_dst = obs_data[dst_type].x.size(0)

                # Uniform negative candidate generation
                neg_src = torch.randint(0, n_src, (num_neg,))
                neg_dst = torch.randint(0, n_dst, (num_neg,))
                neg_edges = torch.stack([neg_src, neg_dst], dim=0)

                supervision[edge_type] = (pos_edges, neg_edges)

        # Fallback: If no hidden edges produced, sample from observed forward edges
        if not supervision:
            for et in obs_data.edge_types:
                if et[1].startswith("Rev"):
                    continue
                if hasattr(obs_data[et], "edge_index") and obs_data[et].edge_index.size(1) > 0:
                    pos_all = cast(torch.Tensor, obs_data[et].edge_index)
                    num_sample = min(10, pos_all.size(1))
                    perm = torch.randperm(pos_all.size(1))[:num_sample]
                    pos_edges = pos_all[:, perm]

                    src_type, _, dst_type = et
                    n_src = obs_data[src_type].x.size(0)
                    n_dst = obs_data[dst_type].x.size(0)
                    neg_src = torch.randint(0, n_src, (num_sample * 2,))
                    neg_dst = torch.randint(0, n_dst, (num_sample * 2,))
                    neg_edges = torch.stack([neg_src, neg_dst], dim=0)

                    supervision[et] = (pos_edges, neg_edges)
                    break

        return obs_data, supervision

    def train_epoch(
        self,
        train_environments: list[HeteroData],
        condition: MaskingCondition = MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
        seed: int = 42,
    ) -> float:
        """Execute one complete training epoch across training environments."""
        self.model.train()
        total_loss = 0.0
        env_count = 0

        for env_idx, env in enumerate(train_environments):
            obs_data, supervision = self._extract_supervision_batch(
                data=env,
                condition=condition,
                seed=seed + env_idx,
            )

            if not supervision:
                continue

            self.optimizer.zero_grad()
            h_dict = self.model(obs_data)

            env_logits_list: list[torch.Tensor] = []
            env_targets_list: list[torch.Tensor] = []

            for edge_type, (pos_edges, neg_edges) in supervision.items():
                if edge_type not in self.model.decoder.relation_to_id:
                    continue
                rel_id = self.model.decoder.get_relation_id(edge_type)
                src_type, _, dst_type = edge_type

                # Positive scores
                h_u_pos = h_dict[src_type][pos_edges[0].to(self.device)]
                h_v_pos = h_dict[dst_type][pos_edges[1].to(self.device)]
                pos_logits = self.model.decoder(h_u_pos, h_v_pos, rel_id)

                # Negative scores
                h_u_neg = h_dict[src_type][neg_edges[0].to(self.device)]
                h_v_neg = h_dict[dst_type][neg_edges[1].to(self.device)]
                neg_logits = self.model.decoder(h_u_neg, h_v_neg, rel_id)

                env_logits_list.extend([pos_logits, neg_logits])
                env_targets_list.extend(
                    [
                        torch.ones_like(pos_logits),
                        torch.zeros_like(neg_logits),
                    ]
                )

            if not env_logits_list:
                continue

            all_logits = torch.cat(env_logits_list, dim=0)
            all_targets = torch.cat(env_targets_list, dim=0)

            loss = self.criterion(all_logits, all_targets)
            loss.backward()

            if self.config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()

            total_loss += float(loss.item())
            env_count += 1

        self.scheduler.step()
        return total_loss / max(env_count, 1)

    def evaluate(
        self,
        val_environments: list[HeteroData],
        condition: MaskingCondition = MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
        seed: int = 1001,
    ) -> tuple[float, float]:
        """Compute mean PR-AUC and ROC-AUC across held-out validation environments."""
        self.model.eval()
        pr_aucs: list[float] = []
        roc_aucs: list[float] = []

        with torch.no_grad():
            for env_idx, env in enumerate(val_environments):
                obs_data, supervision = self._extract_supervision_batch(
                    data=env,
                    condition=condition,
                    seed=seed + env_idx,
                )

                if not supervision:
                    continue

                h_dict = self.model(obs_data)
                env_scores: list[float] = []
                env_labels: list[int] = []

                for edge_type, (pos_edges, neg_edges) in supervision.items():
                    if edge_type not in self.model.decoder.relation_to_id:
                        continue
                    rel_id = self.model.decoder.get_relation_id(edge_type)
                    src_type, _, dst_type = edge_type

                    # Predict positive probabilities
                    h_u_pos = h_dict[src_type][pos_edges[0].to(self.device)]
                    h_v_pos = h_dict[dst_type][pos_edges[1].to(self.device)]
                    pos_probs = self.model.decoder.predict_proba(h_u_pos, h_v_pos, rel_id)

                    # Predict negative probabilities
                    h_u_neg = h_dict[src_type][neg_edges[0].to(self.device)]
                    h_v_neg = h_dict[dst_type][neg_edges[1].to(self.device)]
                    neg_probs = self.model.decoder.predict_proba(h_u_neg, h_v_neg, rel_id)

                    env_scores.extend(pos_probs.cpu().tolist())
                    env_labels.extend([1] * pos_probs.size(0))

                    env_scores.extend(neg_probs.cpu().tolist())
                    env_labels.extend([0] * neg_probs.size(0))

                if not env_scores or sum(env_labels) == 0:
                    continue

                scores_arr = np.array(env_scores)
                labels_arr = np.array(env_labels)

                pr_auc = compute_pr_auc(labels_arr, scores_arr)
                roc_auc = compute_roc_auc(labels_arr, scores_arr)

                pr_aucs.append(pr_auc)
                roc_aucs.append(roc_auc)

        mean_pr = float(np.mean(pr_aucs)) if pr_aucs else 0.0
        mean_roc = float(np.mean(roc_aucs)) if roc_aucs else 0.0
        return mean_pr, mean_roc

    def fit(
        self,
        train_environments: list[HeteroData],
        val_environments: list[HeteroData],
        condition: MaskingCondition = MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
        seed: int = 42,
    ) -> TrainingHistory:
        """Execute full training loop with cosine scheduling and early stopping."""
        best_val_pr = -1.0
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        early_stopped = False

        train_losses: list[float] = []
        val_pr_aucs: list[float] = []
        val_roc_aucs: list[float] = []
        lrs: list[float] = []

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(
                train_environments=train_environments,
                condition=condition,
                seed=seed + epoch,
            )
            val_pr, val_roc = self.evaluate(
                val_environments=val_environments,
                condition=condition,
                seed=seed + 1000 + epoch,
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            train_losses.append(train_loss)
            val_pr_aucs.append(val_pr)
            val_roc_aucs.append(val_roc)
            lrs.append(current_lr)

            # Check for improvement on primary metric (Validation PR-AUC)
            if val_pr > best_val_pr + 1e-4:
                best_val_pr = val_pr
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    early_stopped = True
                    break

        # Restore optimal parameters
        self.model.load_state_dict(best_state)

        return TrainingHistory(
            epoch_train_losses=train_losses,
            epoch_val_pr_aucs=val_pr_aucs,
            epoch_val_roc_aucs=val_roc_aucs,
            epoch_lrs=lrs,
            best_epoch=best_epoch,
            best_val_pr_auc=best_val_pr if best_val_pr >= 0.0 else 0.0,
            early_stopped=early_stopped,
        )
