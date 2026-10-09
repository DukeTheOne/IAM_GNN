# Week 05: 2-Layer RGCN Architecture, Bilinear Decoder, Uncertainty Calibration & Thin-Slice Experiment

**Goal**: Implement the 2-layer Relational Graph Convolutional Network (RGCN) encoder with basis-sharing regularization, the bilinear latent relation prediction decoder head, post-hoc uncertainty calibration (ECE), the focal loss training pipeline, and the thin-slice empirical experiment answering the central research question on held-out cloud environments.
**Estimated Effort**: 18 Hours
**Target Milestone**: **First Empirical Answer to the Central Research Question (Thin-Slice Gate)**

---

## Architectural Decisions & Neural Modeling Context

Before implementing the neural modules, we synthesize the design requirements outlined in [proposal.md](file:///c:/Users/jack6/IAM/IAM_GNN/proposal.md) and [blueprint.md](file:///c:/Users/jack6/IAM/IAM_GNN/blueprint.md):

1. **Heterogeneous Input Dimensionality Alignment**:
   - The AWS IAM multigraph contains 5 heterogeneous node types (`User`, `Role`, `Policy`, `Group`, `Resource`) with distinct raw feature dimension contracts ($D_{\text{user}} \approx 51$, $D_{\text{role}} \approx 52$, $D_{\text{policy}} \approx 39$, $D_{\text{group}} \approx 42$, $D_{\text{resource}} \approx 20$).
   - We implement type-specific linear projection layers ($W_{nt} \in \mathbb{R}^{d_{\text{hidden}} \times D_{nt}}$) equipped with per-type `LayerNorm`, `LeakyReLU(0.2)`, and dropout ($p = 0.2$), embedding all heterogeneous entities into a common continuous semantic space ($d_{\text{hidden}} = 128$) prior to message passing.

2. **Multi-Relational Message Passing with Basis-Sharing Regularization**:
   - Standard GCNs fail on multi-relational graphs because they ignore edge semantics. However, unconstrained relational transformations ($W_r \in \mathbb{R}^{d \times d}$ for each relation $r \in \mathcal{R}$) lead to severe parameter explosion: with $|\mathcal{R}| = 26$, a single unconstrained layer consumes over $425{,}000$ parameters, causing rapid overfitting on sparse graphs.
   - We implement Schlichtkrull et al.'s multi-relational graph convolution with **basis-sharing decomposition** ($B = 8$ basis matrices):
     $$W_r^{(l)} = \sum_{b=1}^{B} a_{r,b}^{(l)} V_b^{(l)}$$
     $$h_i^{(l+1)} = \sigma \left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} \right)$$
   - This reduces layer parameter count from $425{,}984$ to $131{,}280$ (~$70\%$ reduction) while enabling cross-relation knowledge transfer (e.g., sharing structural representations between forward delegation `AssumesRole` and reverse message-passing `RevAssumesRole`).

3. **Inductive Invariance Across Organizations**:
   - Models must never memorize absolute entity IDs or environment-specific graph topologies.
   - By operating exclusively on permutation-invariant topological and semantic node features (department encodings, capability bitmasks, centrality metrics) and typed relations, the `HeteroRGCNEncoder` maintains strict inductive generalization to unseen cloud environments ($\mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}$).

4. **Separation of Concerns: Relation Scoring vs. Path Traversability**:
   - The neural decoder explicitly scores **Problem 1: Hidden Relation Prediction** ($\hat{y}_{uv} = P((u, r, v) \in E_{\text{latent}})$).
   - Downstream attacker–target reachability (**Problem 2**) is evaluated by instantiating predicted candidate completions into the graph and verifying path witnesses, avoiding the naive assumption of independent edge probabilities.

---

## Step 5.1: Multi-Relational Message Passing Layer (5.5h) - [COMPLETED]

**Objective**: Implement the `HeteroRGCNEncoder` module in PyTorch Geometric featuring heterogeneous input projections, 2-layer `RGCNConv` with $B = 8$ basis-sharing regularization, layer normalization, dropout ($p = 0.2$), and LeakyReLU activations.

**Implementation Details**:
1. **Canonical Relation Registry**:
   - Codified fixed mapping of all 26 canonical AWS IAM forward and reverse relational triples (`CANONICAL_RELATION_TYPES`).
2. **Configuration Dataclass (`RGCNConfig`)**:
   - Parameterizes `hidden_channels = 128`, `num_layers = 2`, `num_bases = 8`, `dropout = 0.2`, `act_negative_slope = 0.2`, `use_residual = True`, `aggr = "mean"`.
3. **Encoder Architecture (`HeteroRGCNEncoder`)**:
   - `input_proj`: `nn.ModuleDict` mapping each entity type's raw features ($D_{\text{user}} \approx 51, D_{\text{role}} \approx 52, \dots$) to $d_{\text{hidden}} = 128$.
   - `input_norm`: `nn.ModuleDict` applying per-type `LayerNorm`.
   - `convs`: `nn.ModuleList` of 2 `RGCNConv` layers with $B = 8$ shared bases and `root_weight = True`.
   - `norms`: Layer normalizations between convolutions.
   - Residual connections and dropout for gradient stability.
   - Unified flat tensor aggregation supporting empty edge tensors and variable-sized environments.
   - High-level `forward(data: HeteroData)` and flexible dictionary-based `forward(x_dict, edge_index_dict)` interfaces.
   - `count_basis_parameters()` utility proving >60% parameter savings via basis decomposition.

**Target File Artifacts**:
- [`src/iam/models/rgcn.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/rgcn.py): `HeteroRGCNEncoder` and `RGCNConfig`.
- [`tests/test_rgcn.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_rgcn.py): Unit, parameter-scaling, and inductive forward-pass test suite.

**Verification & Acceptance Results**:
- **Basis-Sharing Regularization**: Confirmed $B = 8$ basis matrices across 26 relations achieves ~70% parameter compression on convolution weights.
- **Gradient Flow**: Verified full end-to-end backpropagation gradients across all projection layers, shared bases, relation combination matrices, root weights, and layer norms.
- **Inductive Generalization**: Verified the exact same model instance processes variable-scale enterprise environments (150 nodes vs. 300 nodes) without modification.
- **Edge Cases**: Verified zero-edge disconnected graphs and empty graphs return properly formatted tensors without runtime failure.
- **Test Suite Results**: **15 / 15 passed** in `test_rgcn.py` with 100% clean passes on strict typing (`mypy --strict`) and linting (`ruff`).

---

## Step 5.2: Bilinear Latent Relation Decoder (3.5h) - [COMPLETED]

**Objective**: Implement the `BilinearRelationPredictor` scoring head for hidden-relation triples $(u, r, v)$: $\hat{y}_{uv} = \sigma(h_u^T W_r h_v + b_r)$, solving Problem 1 (hidden-relation prediction).

**Implementation Details**:
1. **Scoring Head Architecture (`BilinearRelationPredictor`)**:
   - Parameterizes relation matrices $W \in \mathbb{R}^{R \times d \times d}$ across all 26 canonical relations.
   - Supports 3 parameterization modes:
     - `"bilinear"` (standard full $W_r \in \mathbb{R}^{d \times d}$).
     - `"basis"` (basis-regularized $W_r = \sum_{b=1}^{B_{\text{dec}}} \beta_{r,b} U_b$).
     - `"distmult"` (factorized diagonal $W_r = \text{diag}(w_r)$).
   - Configurable per-relation scalar biases $b_r \in \mathbb{R}$ and regularization dropout.
2. **Batch Evaluation Routines**:
   - Homogeneous relation batch scoring ($r \in \mathbb{Z}$): $O(M \cdot d^2)$ vectorized computation $(h_u W_r) \odot h_v$.
   - Heterogeneous multi-relation batch scoring ($r_m \in \mathbb{Z}^M$): vectorized batch tensor contraction $\text{einsum}(md, mde, me \to m)$.
   - All-pairs matrix scoring: $Z^{(r)} = H_{\text{src}} W_r H_{\text{dst}}^T + b_r \in \mathbb{R}^{N_{\text{src}} \times N_{\text{dst}}}$ for candidate ranking.
   - High-level `score_edge_index(h_dict, edge_type, edge_index)` for evaluating masked candidate links.
3. **End-to-End Gradient Flow**:
   - Verified joint backpropagation from BCE loss through `BilinearRelationPredictor` and `HeteroRGCNEncoder` all the way to raw node feature linear projections.

**Target File Artifacts**:
- [`src/iam/models/decoder.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/decoder.py): `BilinearRelationPredictor` and `DecoderConfig`.
- [`tests/test_decoder.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_decoder.py): Unit and end-to-end integration test suite.

**Verification & Acceptance Results**:
- **Pointwise & Matrix Equivalence**: Proved that all-pairs matrix scoring $Z_{ij}$ strictly equals pointwise pair scoring $(h_i, h_j)$ within numerical tolerance ($< 10^{-5}$).
- **Temperature Scaling Sensitivity**: Confirmed temperature parameter $T$ softens probabilities towards $0.5$ when $T > 1$ and sharpens them towards binary extremes when $T < 1$.
- **End-to-End Backpropagation**: Verified non-zero gradients on all decoder weights ($W$) and encoder weights (`convs`, `input_proj["User"]`).
- **Test Suite Results**: **16 / 16 passed** in `test_decoder.py` with 100% clean passes on strict typing (`mypy --strict`) and linting (`ruff`).

---

## Step 5.3: Uncertainty Calibration (P1) (3.5h) - [COMPLETED]

**Objective**: Implement post-hoc temperature scaling $\hat{p}_{uv} = \sigma(z_{uv} / T)$ on validation environments, optimizing $T > 0$ via Negative Log-Likelihood (NLL) and computing Expected Calibration Error (ECE) and reliability diagrams.

**Implementation Details**:
1. **Temperature Scaling Module (`TemperatureScalingCalibrator`)**:
   - Single scalar temperature parameter $T > 0$ clamped to bounded range $[0.01, 100.0]$ to prevent degenerate probability saturation or collapse.
   - Fits $T$ by minimizing binary cross-entropy (NLL) on held-out validation predictions using L-BFGS.
   - Provides `calibrate_logits(z)`, `predict_proba(z)`, and `calibrate_probabilities(p)`.
2. **Calibration Diagnostics & Metrics**:
   - `compute_ece(probabilities, targets, num_bins)`: Computes Expected Calibration Error $\text{ECE} = \sum_m \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$, Maximum Calibration Error (MCE), and per-bin statistics.
   - `compute_nll(probabilities, targets)`: Numerically stable binary log-loss computation.
3. **Reliability Diagram Reporting**:
   - `generate_markdown_diagram()`: Renders ASCII-visualized GitHub Flavored Markdown reliability diagrams for automated thesis reports and triage dashboards.
   - `to_dict()`: Structured serialization for JSON logging and visualization pipelines.

**Target File Artifacts**:
- [`src/iam/models/calibration.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/calibration.py): `TemperatureScalingCalibrator`, `CalibrationResult`, `CalibrationBin`, `compute_ece`, and `compute_nll`.
- [`tests/test_calibration.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_calibration.py): Unit and validation test suite.

**Verification & Acceptance Results**:
- **Overconfidence Mitigation**: Verified that artificially overconfident logits ($T = 1.0$, high ECE) are properly softened with $T > 1.0$, yielding substantial ECE reductions.
- **Underconfidence Adjustment**: Verified that underconfident logits are sharpened with $T < 1.0$, minimizing NLL.
- **Inductive Validation Environment**: Successfully calibrated predictions on held-out enterprise cloud environments generated by `EnterpriseTopologyGenerator` and evaluated by `HeteroRGCNEncoder` + `BilinearRelationPredictor`.
- **Test Suite Results**: **13 / 13 passed** in `test_calibration.py` with 100% clean passes on strict typing (`mypy --strict`) and linting (`ruff`).

---

## Step 5.4: Loss Formulation & Training Loop (3.5h) - [PLANNED]
- Focal binary cross-entropy ($\gamma = 2.0$) with AdamW and cosine annealing schedule, early stopping on validation PR-AUC.

---

## Step 5.5: Thin-Slice Experiment (2.0h) - [PLANNED]
- End-to-end thin-slice evaluation comparing RGCN against Baselines (BFS, RF, Motif Completion) on held-out environments under Condition E.
