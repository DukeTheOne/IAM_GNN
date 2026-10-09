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

## Step 5.2: Bilinear Latent Relation Decoder (3.5h) - [PLANNED]
- Implement `BilinearRelationPredictor` scoring candidate triples: $\hat{y}_{uv} = \sigma(h_u^T W_r h_v + b_r)$.

---

## Step 5.3: Uncertainty Calibration (P1) (3.5h) - [PLANNED]
- Post-hoc temperature scaling $\hat{p} = \sigma(z / T)$ on validation environments, computing ECE and reliability diagrams.

---

## Step 5.4: Loss Formulation & Training Loop (3.5h) - [PLANNED]
- Focal binary cross-entropy ($\gamma = 2.0$) with AdamW and cosine annealing schedule, early stopping on validation PR-AUC.

---

## Step 5.5: Thin-Slice Experiment (2.0h) - [PLANNED]
- End-to-end thin-slice evaluation comparing RGCN against Baselines (BFS, RF, Motif Completion) on held-out environments under Condition E.
