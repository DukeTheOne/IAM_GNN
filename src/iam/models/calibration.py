r"""Post-hoc uncertainty calibration and Expected Calibration Error (ECE) suite.

Implements temperature scaling on validation environments:
    \hat{p}_{uv} = \sigma(z_{uv} / T)
with NLL loss optimization, Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and binned reliability diagram diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class CalibrationBin:
    """Statistics for an individual calibration confidence bin."""

    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    mean_accuracy: float
    calibration_gap: float


@dataclass(frozen=True)
class CalibrationResult:
    """Summary of temperature scaling calibration performance."""

    temperature: float
    uncalibrated_ece: float
    calibrated_ece: float
    uncalibrated_nll: float
    calibrated_nll: float
    uncalibrated_mce: float
    calibrated_mce: float
    sample_count: int
    bins_uncalibrated: list[CalibrationBin]
    bins_calibrated: list[CalibrationBin]

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration summary to dictionary."""
        return {
            "temperature": round(self.temperature, 4),
            "uncalibrated_ece": round(self.uncalibrated_ece, 4),
            "calibrated_ece": round(self.calibrated_ece, 4),
            "ece_reduction_pct": round(
                (1.0 - self.calibrated_ece / max(self.uncalibrated_ece, 1e-8)) * 100.0, 2
            ),
            "uncalibrated_nll": round(self.uncalibrated_nll, 4),
            "calibrated_nll": round(self.calibrated_nll, 4),
            "uncalibrated_mce": round(self.uncalibrated_mce, 4),
            "calibrated_mce": round(self.calibrated_mce, 4),
            "sample_count": self.sample_count,
            "bins_calibrated": [
                {
                    "range": f"[{b.bin_lower:.2f}, {b.bin_upper:.2f})",
                    "samples": b.sample_count,
                    "confidence": round(b.mean_confidence, 4),
                    "accuracy": round(b.mean_accuracy, 4),
                    "gap": round(b.calibration_gap, 4),
                }
                for b in self.bins_calibrated
            ],
        }

    def generate_markdown_diagram(self) -> str:
        """Generate GitHub Flavored Markdown reliability diagram report."""
        lines: list[str] = [
            "### Reliability Diagram & Uncertainty Calibration Summary",
            "",
            f"- **Optimal Temperature ($T$)**: `{self.temperature:.4f}`",
            f"- **Uncalibrated ECE**: `{self.uncalibrated_ece * 100.0:.2f}%` ➔ **Calibrated ECE**: `{self.calibrated_ece * 100.0:.2f}%`",
            f"- **Uncalibrated NLL**: `{self.uncalibrated_nll:.4f}` ➔ **Calibrated NLL**: `{self.calibrated_nll:.4f}`",
            f"- **Uncalibrated MCE**: `{self.uncalibrated_mce * 100.0:.2f}%` ➔ **Calibrated MCE**: `{self.calibrated_mce * 100.0:.2f}%`",
            "",
            "| Confidence Bin | Samples | Mean Confidence | Empirical Accuracy | Calibration Gap | Alignment Bar |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for b in self.bins_calibrated:
            if b.sample_count == 0:
                continue
            gap_pct = b.calibration_gap * 100.0
            bar_len = int(round(gap_pct / 5.0))
            bar_repr = "█" * min(bar_len, 20) if bar_len > 0 else "✓"
            lines.append(
                f"| `[{b.bin_lower:.2f}, {b.bin_upper:.2f})` | {b.sample_count} | "
                f"`{b.mean_confidence:.3f}` | `{b.mean_accuracy:.3f}` | "
                f"`{b.calibration_gap:.3f}` | `{bar_repr}` |"
            )

        return "\n".join(lines)


def compute_nll(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> float:
    """Compute binary Negative Log-Likelihood (Log Loss) with numerical clipping."""
    p = (
        probabilities.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor)
        else np.asarray(probabilities)
    ).astype(np.float64)
    y = (
        targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    ).astype(np.float64)

    if len(p) == 0:
        return 0.0

    p_clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    nll = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
    return float(nll)


def compute_ece(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_bins: int = 10,
) -> tuple[float, float, list[CalibrationBin]]:
    """Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and bin stats.

    Args:
        probabilities: 1D array or tensor of predicted probabilities in [0.0, 1.0].
        targets: 1D array or tensor of binary ground truth labels in {0, 1}.
        num_bins: Number of equal-width calibration bins (default 10).

    Returns:
        (ece, mce, bins): Expected Calibration Error, Max Calibration Error, and per-bin stats.
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")

    p = (
        probabilities.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor)
        else np.asarray(probabilities)
    )
    y = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)

    total_samples = len(p)
    if total_samples == 0:
        return 0.0, 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[CalibrationBin] = []
    total_ece = 0.0
    max_ce = 0.0

    for i in range(num_bins):
        bin_lower = float(bin_boundaries[i])
        bin_upper = float(bin_boundaries[i + 1])

        if i == num_bins - 1:
            in_bin = (p >= bin_lower) & (p <= bin_upper)
        else:
            in_bin = (p >= bin_lower) & (p < bin_upper)

        sample_count = int(np.sum(in_bin))

        if sample_count > 0:
            mean_conf = float(np.mean(p[in_bin]))
            mean_acc = float(np.mean(y[in_bin]))
            gap = float(abs(mean_conf - mean_acc))
            weight = sample_count / total_samples

            total_ece += weight * gap
            if gap > max_ce:
                max_ce = gap
        else:
            mean_conf = (bin_lower + bin_upper) / 2.0
            mean_acc = 0.0
            gap = 0.0

        bins.append(
            CalibrationBin(
                bin_lower=bin_lower,
                bin_upper=bin_upper,
                sample_count=sample_count,
                mean_confidence=mean_conf,
                mean_accuracy=mean_acc,
                calibration_gap=gap,
            )
        )

    return float(total_ece), float(max_ce), bins


class TemperatureScalingCalibrator(nn.Module):
    r"""Post-hoc temperature scaling calibrator for link prediction logits.

    Transforms uncalibrated model logits z via temperature scaling:
        \hat{p} = \sigma(z / T)
    where scalar T > 0 is learned on validation data by minimizing NLL loss.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        num_bins: int = 10,
        max_iter: int = 50,
        lr: float = 0.05,
    ) -> None:
        """Initialize calibrator with hyperparameter settings."""
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError(f"initial_temperature must be positive, got {initial_temperature}")
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")

        self.num_bins: int = num_bins
        self.max_iter: int = max_iter
        self.lr: float = lr
        self.is_fitted: bool = False

        self.temperature_param = nn.Parameter(torch.tensor([float(initial_temperature)]))

    @property
    def temperature(self) -> float:
        """Return scalar temperature value T."""
        return float(self.temperature_param.item())

    def fit(
        self,
        logits: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> CalibrationResult:
        """Fit optimal scalar temperature T on validation logits and labels using L-BFGS.

        Args:
            logits: 1D array or tensor of uncalibrated model logits.
            targets: 1D array or tensor of true binary labels in {0, 1}.

        Returns:
            result: CalibrationResult detailing uncalibrated vs. calibrated ECE and NLL.
        """
        if isinstance(logits, np.ndarray):
            logits_t = torch.from_numpy(logits).float()
        else:
            logits_t = logits.detach().float()

        if isinstance(targets, np.ndarray):
            targets_t = torch.from_numpy(targets).float()
        else:
            targets_t = targets.detach().float()

        if logits_t.numel() == 0:
            raise ValueError("Cannot fit calibrator on empty logits.")

        device = logits_t.device
        self.to(device)

        # Baseline uncalibrated metrics (T = 1.0)
        uncal_probs = torch.sigmoid(logits_t)
        uncal_ece, uncal_mce, uncal_bins = compute_ece(uncal_probs, targets_t, self.num_bins)
        uncal_nll = compute_nll(uncal_probs, targets_t)

        # Setup optimization
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature_param], lr=self.lr, max_iter=self.max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            # Enforce strictly positive temperature via clamp
            t_clamped = torch.clamp(self.temperature_param, min=0.01, max=100.0)
            scaled_logits = logits_t / t_clamped
            loss = criterion(scaled_logits, targets_t)
            loss.backward()
            return cast(torch.Tensor, loss)

        optimizer.step(closure)  # type: ignore[no-untyped-call]

        # Final parameter clamp
        with torch.no_grad():
            self.temperature_param.clamp_(min=0.01, max=100.0)

        self.is_fitted = True

        # Calibrated metrics
        cal_probs = self.predict_proba(logits_t)
        cal_ece, cal_mce, cal_bins = compute_ece(cal_probs, targets_t, self.num_bins)
        cal_nll = compute_nll(cal_probs, targets_t)

        return CalibrationResult(
            temperature=self.temperature,
            uncalibrated_ece=uncal_ece,
            calibrated_ece=cal_ece,
            uncalibrated_nll=uncal_nll,
            calibrated_nll=cal_nll,
            uncalibrated_mce=uncal_mce,
            calibrated_mce=cal_mce,
            sample_count=int(logits_t.size(0)),
            bins_uncalibrated=uncal_bins,
            bins_calibrated=cal_bins,
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature: z / T."""
        t_clamped = torch.clamp(self.temperature_param, min=0.01, max=100.0)
        return logits / t_clamped

    def predict_proba(self, logits: torch.Tensor | np.ndarray) -> torch.Tensor:
        r"""Compute calibrated probabilities: \hat{p} = \sigma(z / T)."""
        if isinstance(logits, np.ndarray):
            logits_t = torch.from_numpy(logits).float()
        else:
            logits_t = logits.float()

        scaled_logits = self.forward(logits_t)
        return torch.sigmoid(scaled_logits)

    def calibrate_probabilities(
        self,
        probabilities: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Calibrate raw probability predictions by converting to logits, scaling, and sigmoiding."""
        if isinstance(probabilities, np.ndarray):
            p = torch.from_numpy(probabilities).float()
        else:
            p = probabilities.float()

        # Invert sigmoid: z = log(p / (1 - p))
        p_clamped = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        logits = torch.log(p_clamped / (1.0 - p_clamped))
        return self.predict_proba(logits)
