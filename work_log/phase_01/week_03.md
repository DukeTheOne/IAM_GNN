# Week 03: PyG HeteroData Pipeline, Environment-Level Splits & Structured Masking

**Goal**: Convert enterprise cloud authorization graphs into PyTorch Geometric `HeteroData`, establish strict environment-level inductive dataset splits to eliminate synthetic memorization, and codify the structured masking framework across all 5 industrial blind-spot conditions.
**Estimated Effort**: 17 Hours
**Target Milestone**: **M1: Validated Inductive IAM Corpus with Masking Operators & PyG Pipeline**

---

## Architectural Decisions & Future Scaling Context

Before beginning implementation, consider the core architectural requirements designed to ensure this subsystem scales seamlessly from the Tier 1 prototype (500–5,000 nodes) into Tier 2 Master's Thesis research (50,000+ nodes, real-world Terraform/CloudFormation IaC parsing, and zero-shot TAC-Bench transfer):

1. **Dimensionality Invariance & Fixed Feature Schema**:
   - Heterogeneous GNN message passing across inductive environment splits requires strictly invariant feature dimensionality $d_v$ per `NodeType` across all training, validation, and test organizations ($\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}$).
   - Dynamic vocabulary sizing based on individual graphs leads to fatal shape mismatch during inductive evaluation. We codify an immutable, globally ordered `ActionVocabulary` derived from AWS's machine-readable Service Authorization Reference snapshot, reserving explicit buckets for Out-of-Vocabulary (OOV) actions.
   - Resource and identity categorical encodings must use fixed-dimension one-hot projections rather than arbitrary strings.

2. **Memory Efficiency & Large Graph Scaling ($N \ge 5{,}000$ to $50{,}000$)**:
   - NetworkX Python objects carry substantial memory overhead (~4.3 KB / node). For multi-environment training and large-scale inductive evaluation, the PyG pipeline must transition strictly to compact PyTorch COO edge tensors (`edge_index` of shape $[2, |E|]$, `torch.long`) and continuous feature matrices (`torch.float32`).
   - Implement streaming dataset access and PyG `Dataset` / `InMemoryDataset` interfaces capable of loading pre-indexed `.pt` tensor bundles directly from the disk-efficient serialized outputs generated in Week 2 ([export.py](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/generator/export.py)).
   - Adjacency and negative sampling must never compute dense $N \times N$ matrices; all candidate checks must be vectorized over sparse indices.

3. **Strict Inductive Generalization Safeguards (Zero Data Leakage)**:
   - Standard link prediction benchmarks often randomly partition edges within a single graph. In enterprise cloud IAM, this violates real-world audit conditions because an auditor faces an unseen company topology.
   - Dataset splits are performed strictly by **Environment / Organization Boundary** ($\mathcal{O}_{\text{train}} \cap \mathcal{O}_{\text{test}} = \emptyset$). Nodes, accounts, and policies in test environments are completely disjoint from training environments.
   - Link prediction heads must evaluate on inductive edge sets: the GNN passes messages strictly over visible edges $G_o$, and scores candidate triples $(u, r, v)$ in an evaluation set $E_{\text{eval}}$ comprising hidden true relations $H$ and carefully stratified negative non-edges.

4. **Stratified Negative Sampling for Severe Class Imbalance**:
   - Privilege escalation and authorization links are inherently sparse ($< 1:1{,}000$ positive-to-negative ratio). Uniform random edge sampling produces trivial negatives (e.g., Intern $\to$ Billing S3 bucket) that artificially inflate model metrics without training effective decision boundaries.
   - The negative sampling engine must enforce:
     - **Type Conformance**: Sampled $(u, r, v)$ triples must satisfy valid AWS relational signatures (e.g., `AssumesRole` only connects principals to roles).
     - **Intra-Department Hard Negatives**: Sample non-escalating identity pairs within the same organizational unit (e.g., DevOps $\to$ DevOps) where feature vectors and centrality metrics are closely matched.
     - **Inter-Department Benign Negatives**: Sample legitimate cross-boundary pairs that do not grant administrative reachability.

5. **Extensible Masking Architecture (Industrial Blind Spots)**:
   - Masking removes **authorization evidence and delegation relationships**, not merely random edges.
   - Use the Strategy Pattern (`BaseMaskingOperator`) with modular implementations for all five conditions defined in the [proposal.md](file:///c:/Users/jack6/IAM/IAM_GNN/proposal.md):
     - **Condition A (Random Masking - P0 Baseline)**: Random edge dropout ($p \in \{10\%, 20\%, 30\%, 40\%\}$).
     - **Condition E (Adversarial Bridge Masking - P0 Core)**: Masking the critical bottleneck relation along active PE attack paths.
     - **Condition B (Cross-Account Masking - P1)**: Complete configuration blindness inside unobserved account silos.
     - **Condition C (Federated IdP Masking - P2 / Tier 2 Foundation)**: Cloud trust policy visible, external IdP group delegation hidden.
     - **Condition D (Ephemeral STS Masking - P2 / Tier 2 Foundation)**: Short-lived runtime session delegation hidden.
   - Every masking operator produces an observed graph $G_o$, a hidden ground-truth set $H$, evaluation supervision tensors, and full audit provenance for reproducibility.

---

## Step 3.1: Vocabulary Building & Feature Representation Engine (4.0h) - [COMPLETED]

**Objective**: Build a deterministic, extensible feature extraction engine that projects heterogeneous IAM graph nodes (`User`, `Role`, `Policy`, `Group`, `Resource`) into compact, invariant PyTorch tensors.

**Future Scaling & Architectural Considerations**:
- In Tier 2, the system will ingest full multi-account Terraform deployments and TAC-Bench graphs. The vocabulary must not depend on runtime graph scanning; it must be grounded in the versioned capability snapshot ([capability_snapshot.json](file:///c:/Users/jack6/IAM/IAM_GNN/data/schemas/capability_snapshot.json)) created in Week 1.
- All node types require fixed feature dimension contracts:
  - $d_{\text{User}}$: Department one-hot (7) + Security flags (2) + Centrality (11) + Action bitmask ($|A|$) = $20 + |A_{\text{vocab}}|$ (51)
  - $d_{\text{Role}}$: Service role flag (1) + Admin flag (1) + HighValue (1) + Department (7) + Centrality (11) + Action bitmask ($|A|$) = $21 + |A_{\text{vocab}}|$ (52)
  - $d_{\text{Policy}}$: Statement stats (3) + Admin flag (1) + Wildcard resource (1) + Centrality (3) + Action bitmask ($|A|$) = $8 + |A_{\text{vocab}}|$ (39)
  - $d_{\text{Group}}$: Department one-hot (7) + Admin flag (1) + Centrality (3) + Aggregated action bitmask ($|A|$) = $11 + |A_{\text{vocab}}|$ (42)
  - $d_{\text{Resource}}$: Resource type one-hot (8) + Service prefix one-hot (8) + Crown jewel flag (1) + Centrality (3) = 20

**Implementation Instructions**:
1. **Action Vocabulary Engine (`ActionVocabulary`)**:
   - [x] Ingest canonical AWS action sets from [capability_snapshot.json](file:///c:/Users/jack6/IAM/IAM_GNN/data/schemas/capability_snapshot.json) across core services (`iam`, `sts`, `lambda`, `ec2`, `s3`, `kms`, `rds`).
   - [x] Build bidirectional mapping: `action_to_idx: dict[str, int]` and `idx_to_action: list[str]`.
   - [x] Reserve index `0` for unknown/OOV actions to gracefully handle future services or novel IaC actions without breaking tensor shapes.
   - [x] Provide multi-hot encoding utility `encode_actions(actions: Iterable[str]) -> np.ndarray` (shape: $[|A_{\text{vocab}}|]$).

2. **Topological Centrality Extractor (`CentralityExtractor`)**:
   - [x] Implement fast, vectorized centrality computation over `IAMGraph`:
     - Per-relation in-degree and out-degree across all 5 canonical relations (`MemberOf`, `AssumesRole`, `AttachedWith`, `ActsOn`, `PassesTo`).
     - Global PageRank scores using power iteration with damping factor $\alpha = 0.85$ and convergence tolerance $10^{-6}$ (via NetworkX/SciPy sparse adjacency).
     - Log-transform degree features ($\log(1 + k)$) to stabilize neural message passing under power-law degree distributions.

3. **Heterogeneous Node Feature Extractors (`NodeFeatureExtractor`)**:
   - [x] `extract_user_features(user: GraphNode, graph: IAMGraph) -> np.ndarray`:
     - Department one-hot encoding (`SecOps`, `DevOps`, `DataBI`, `QA`, `Billing`, `Interns`, `Other`).
     - Binary privilege flags: `is_admin`, `is_high_value`.
     - 10-dimensional relational degree vector (in/out for 5 relations) + 1-dimensional PageRank.
     - Multi-hot action capability vector aggregated from directly attached and group-inherited policies.
   - [x] `extract_role_features(role: GraphNode, graph: IAMGraph) -> np.ndarray`:
     - Role classification: service principal role (`lambda.amazonaws.com`, `ec2.amazonaws.com`) vs. principal-assumable role.
     - Trust policy metadata flags.
     - Relational degree vector + PageRank.
     - Multi-hot action vector from attached policies.
   - [x] `extract_policy_features(policy: GraphNode, graph: IAMGraph) -> np.ndarray`:
     - Policy AST metrics: statement count, normalized allow/deny counts, wildcard resource flag.
     - `is_admin` policy flag (e.g., `AdministratorAccess`).
     - Multi-hot granted action vector.
     - In-degree (attached principals) and out-degree (scoped resources).
   - [x] `extract_group_features(group: GraphNode, graph: IAMGraph) -> np.ndarray`:
     - Department one-hot encoding.
     - Member count (in-degree from `User`), attached policy count (out-degree to `Policy`).
     - Aggregated action capability bitmask.
   - [x] `extract_resource_features(resource: GraphNode, graph: IAMGraph) -> np.ndarray`:
     - One-hot resource type (`S3_Bucket`, `Lambda_Function`, `KMS_Key`, `EC2_Instance`, `IAM_Role`, `IAM_Policy`, `Other`).
     - One-hot AWS service prefix (`s3`, `lambda`, `kms`, `ec2`, `iam`, `sts`, `rds`, `other`).
     - Crown jewel flag `is_high_value`.
     - In-degree from policies (`ActsOn`) and compute execution roles (`PassesTo`).

4. **Batch Feature Matrix Builder**:
   - [x] Implement `build_node_feature_dict(graph: IAMGraph) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]`:
     - Iterates over typed nodes in deterministic integer index order matching `EnvironmentExporter.to_pyg_ready_dict()`.
     - Returns dictionary of 2D `torch.float32` tensors `x_dict[node_type]` of shape $[N_{\text{type}}, d_{\text{type}}]$.
     - Enforces rigorous shape assertions against global feature dimension constants.

**Execution Log & Completed Artifacts**:
- **Action Vocabulary Engine**: Implemented [`src/iam/models/vocabulary.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/vocabulary.py):
  - `ActionVocabulary`: Ingests 30 canonical actions from `capability_snapshot.json` + explicit `<OOV>` slot at index 0 ($|A_{\text{vocab}}| = 31$). Case-insensitive lookup, pattern expansion (`s3:*`, `*`) via integrated `WildcardResolver`, and JSON serialization roundtrips (`to_dict`, `from_dict`, `save_json`, `load_json`).
  - Categorical utilities: `one_hot_encode` for departments (7), resource types (8), and AWS services (8); `parse_arn_service_and_type` for deterministic ARN parsing.
  - `FeatureDimensionConfig` & `get_feature_dimension_config`: Exact dimension contracts enforced across entity types ($d_{\text{User}} = 51$, $d_{\text{Role}} = 52$, $d_{\text{Policy}} = 39$, $d_{\text{Group}} = 42$, $d_{\text{Resource}} = 20$).
- **Topological Centrality & Feature Extractors**: Implemented [`src/iam/models/features.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/features.py):
  - `CentralityExtractor`: $O(|V| + |E|)$ single-pass extraction computing 10-dimensional in/out degrees across all 5 canonical relations with $\log(1 + d)$ scaling and global PageRank scores ($\alpha = 0.85$, tol $10^{-5}$).
  - `NodeFeatureExtractor`: Extractors for `User`, `Role`, `Policy`, `Group`, and `Resource` with direct and transitive policy permission propagation through group hierarchies (`MemberOf` $\to$ `AttachedWith`).
  - `build_node_feature_dict`: High-throughput batch converter producing 2D `torch.float32` tensors matching the exact contiguous integer indexing in `EnvironmentExporter.to_pyg_ready_dict()`.
- **Public API Export**: Updated [`src/iam/models/__init__.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/__init__.py) with clean exports.
- **Unit & Integration Test Suite**: Implemented [`tests/test_features.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_features.py) with 19 comprehensive tests:
  - Vocabulary default loading, OOV indexing, case-insensitivity, wildcard expansion, decoding, JSON roundtrip, and dimension contracts.
  - Categorical one-hot and ARN decomposition across all AWS services.
  - Centrality metrics, degree bounds, and PageRank convergence.
  - Typed node feature vector extraction, non-negativity, and permission propagation.
  - Batch tensor shapes, dtypes, deterministic reproducibility, and runtime performance benchmarks ($< 0.15\text{ s}$ for $N = 1{,}000$).
  - **Test Suite Results**: **19 / 19 passed** in 4.01s (**128 / 128 passed** across the full repo).
  - **Code Coverage**: Achieved **95% coverage** on `vocabulary.py` and **93% coverage** on `features.py`.
  - **Tooling Verification**: Clean passes for `ruff check`, `ruff format --check`, `mypy --strict src tests`, and all pre-commit hooks.

---

## Step 3.2: PyG `HeteroData` Conversion & Multirelational Tensor Pipeline (4.5h) - [COMPLETED]

**Objective**: Convert `IAMGraph` instances and pre-indexed export schemas into native PyTorch Geometric `HeteroData` objects featuring typed edge stores, reverse message-passing relations, and ground-truth bridge annotations.

**Future Scaling & Architectural Considerations**:
- PyG's `HeteroData` serves as the universal tensor representation for RGCN training (Week 5), candidate pruning (Week 7), and PyVis dashboard extraction (Week 9).
- To allow bidirectional message passing in the RGCN without corrupting directed privilege escalation semantics:
  - Directed authorization edges define the ground-truth reachability graph.
  - Reverse relations (`RevAssumesRole`, `RevAttachedWith`, etc.) are synthesized strictly for message-passing aggregation and must be excluded from link prediction scoring targets.
- Pre-indexed mappings (`node_id_to_idx`, `idx_to_node_id`) are preserved within `HeteroData` metadata so symbolic solvers can map tensor indices back to IAM ARNs and JSON AST nodes during repair synthesis.

**Implementation Details**:
1. **Canonical Heterogeneous Relation Triples & Reversal Algebra**:
   - [x] Codified canonical directed relation triples:
     - `('User', 'MemberOf', 'Group')`
     - `('User', 'AssumesRole', 'Role')`
     - `('Role', 'AssumesRole', 'Role')`
     - `('User', 'AttachedWith', 'Policy')`
     - `('Role', 'AttachedWith', 'Policy')`
     - `('Group', 'AttachedWith', 'Policy')`
     - `('Policy', 'ActsOn', 'Resource')`
     - `('Role', 'PassesTo', 'Resource')`
     - Additional IAM relations: `('User', 'PassesTo', 'Role')`, `('Role', 'PassesTo', 'Role')`, `('Policy', 'ActsOn', 'User|Role|Policy')`.
   - [x] Codified corresponding reverse relation triples for bidirectional aggregation:
     - `('Group', 'RevMemberOf', 'User')`
     - `('Role', 'RevAssumesRole', 'User')`
     - `('Role', 'RevAssumesRole', 'Role')`
     - `('Policy', 'RevAttachedWith', 'User')`
     - `('Policy', 'RevAttachedWith', 'Role')`
     - `('Policy', 'RevAttachedWith', 'Group')`
     - `('Resource', 'RevActsOn', 'Policy')`
     - `('Resource', 'RevPassesTo', 'Role')`
   - [x] Built helper utilities: `get_reverse_relation_name()`, `get_reverse_edge_type()`, `is_reverse_edge_type()`.

2. **HeteroData Converter Engine (`IAMHeteroDataConverter`)**:
   - [x] Implemented `convert(graph: IAMGraph, label_set: GroundTruthLabelSet | None = None) -> HeteroData`:
     - Builds aligned node feature matrices via `build_node_feature_dict(graph)`.
     - Assigns typed node stores: `data[node_type].x`, `data[node_type].node_ids`, `data[node_type].arns`, `data[node_type].num_nodes`.
     - Populates typed COO edge index tensors `data[edge_type].edge_index` ($[2, |E_{\text{rel}}|]$, `torch.long`).
     - Generates symmetric reverse edge tensors `data[rev_edge_type].edge_index = edge_index[[1, 0], :]`.
     - Attaches boolean bridge mask tensors `data[edge_type].edge_is_bridge` ($[|E_{\text{rel}}|]$, `torch.bool`) indicating ground-truth attack bottleneck relations (Condition E). Set to `False` on reverse relations.
     - Encodes action bitmask tensors `data[edge_type].edge_attr` ($[|E_{\text{rel}}|, |A_{\text{vocab}}|]$, `torch.float32`) for fine-grained action semantics on edges.
     - Stores bijective entity lookup tables `data.node_id_to_idx` and `data.idx_to_node_id`.

3. **Ground-Truth Label Annotation**:
   - [x] Annotates privilege escalation ground truth when `label_set` is provided:
     - Populates `data['pe_pairs']` mapping source principal index to target crown jewel index with hop distances.
     - Stores `data['num_pe_pairs']` for direct metric evaluation.

4. **Fast Disk & Pre-Indexed Serialization Pipeline**:
   - [x] Implemented `save_hetero_data(data: HeteroData, path: Path | str)` and `load_hetero_data(path: Path | str) -> HeteroData` via `torch.save` / `torch.load` with type safety assertions.
   - [x] Implemented `convert_from_pyg_ready_dict(pyg_dict, feature_dict)` for direct O(1) instantiation from Week 2's `EnvironmentExporter.to_pyg_ready_dict()` payload without rebuilding intermediate NetworkX graph objects.

**Target File Artifacts**:
- [`src/iam/models/converter.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/converter.py): `IAMHeteroDataConverter`, canonical edge type definitions, and serialization methods.
- [`src/iam/models/__init__.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/__init__.py): Exported public symbols.
- [`tests/test_converter.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_converter.py): Unit and integration tests covering minimal graphs, reverse symmetry, bridge flags, enterprise graphs, PyG dict conversion, serialization roundtrip, and performance.

**Verification & Acceptance Results**:
- **PyG Schema Validation**: All converted instances pass `data.validate()` without schema violations.
- **Reverse Relation Symmetry**: Verified $\text{data}[u, r, v].\text{num\_edges} == \text{data}[v, \text{Rev}(r), u].\text{num\_edges}$ across all edge types.
- **Bridge Edge Isolation**: 100% of injected PE bottleneck relations are marked `edge_is_bridge = True` in forward edges and `False` in reverse edges.
- **Performance**: Full conversion of $N = 1{,}000$ enterprise graph completed in **0.038 s** (budget: $< 0.20\text{ s}$).
- **Test Suite Results**: **9 / 9 passed** in `test_converter.py` (**28 / 28 passed** across `iam.models`, **137 / 137 passed** across full repo).
- **Code Coverage**: Achieved **97% coverage** on `converter.py` (**95% coverage** on `iam.models`).
- **Tooling Verification**: Clean passes for `ruff check` (0 errors), `ruff format --check`, and `mypy --strict src tests` (0 errors across 32 files).

---

## Step 3.3: Environment-Level Dataset Splitting & Inductive Pipeline (4.5h) - [COMPLETED]

**Objective**: Establish a strict environment-level dataset partitioning scheme ($\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}$) and implement a stratified negative sampling harness for inductive link prediction.

**Future Scaling & Architectural Considerations**:
- In the [blueprint.md](file:///c:/Users/jack6/IAM/IAM_GNN/blueprint.md), Gate G0 requires evaluating the RGCN against baselines on **completely held-out environments**.
- Synthetic memorization is prevented by partitioning environments by Organization ID ($\mathcal{O}_1, \dots, \mathcal{O}_{14}$) and ensuring different structural parameters (varying departmental branch factors, node counts, and motif distributions).
- Inductive link prediction task setup:
  - Input: Observed topology $G_o$ (represented by message-passing edges in `edge_index`).
  - Target: Candidate edge evaluation tensor `edge_label_index` of shape $[2, |E_{\text{eval}}|]$ and ground-truth binary label tensor `edge_label` of shape $[|E_{\text{eval}}|]$ containing true latent links ($y = 1$) and sampled negative links ($y = 0$).

**Implementation Details**:
1. **Parameterized Multi-Environment Corpus Generator (`EnterpriseCorpusGenerator`)**:
   - [x] Codified the standardized canonical specification of 14 distinct enterprise environments (`DEFAULT_ORGANIZATION_CONFIGS`):
     - **Training Set $\mathcal{D}_{\text{train}}$ (8 Organizations: $\mathcal{O}_1 - \mathcal{O}_8$)**:
       - Scales: $N \in [350, 1500]$ nodes.
       - Unique AWS account IDs (`100000000001` - `100000000008`) and seeds (`101` - `108`).
       - Balanced motif allocations: P0 motifs (`PassRoleLambda`, `CreateAccessKey`, `AttachPolicy`, `AssumeRoleChain`) and P1 motifs (`PassRoleEC2`, `SetDefaultPolicyVersion`).
     - **Validation Set $\mathcal{D}_{\text{val}}$ (2 Organizations: $\mathcal{O}_9 - \mathcal{O}_{10}$)**:
       - Scales: $N \in [800, 1400]$ nodes, accounts `100000000009` - `100000000010`.
       - Dedicated for RGCN hyperparameter selection, early stopping, and temperature scaling calibration (ECE).
     - **Held-Out Test Set $\mathcal{D}_{\text{test}}$ (4 Organizations: $\mathcal{O}_{11} - \mathcal{O}_{14}$)**:
       - Scales: $N \in [1050, 3500]$ nodes, accounts `100000000011` - `100000000014`.
       - High structural divergence: heavy-tail power-law parameter ($\alpha = 2.2$), multi-hop role chaining (lengths 3, 4), and branching diamond topologies.
   - [x] Implemented `generate_corpus(output_dir, ...)` creating disk partitions `processed/` (PyG `.pt`) and `raw/` (compressed `.json.gz`).
   - [x] Implemented automated manifest export to `corpus_manifest.json` recording organization metadata, split allocations, entity counts, seeds, and file paths.
   - [x] Implemented cache reuse (`force=False`) allowing instant pipeline initialization when files already exist on disk.

2. **Stratified Negative Edge Sampler (`StratifiedNegativeSampler`)**:
   - [x] Implemented type-safe negative sampling:
     - For target triple `(src_type, rel, dst_type)`, only samples pairs $(u, v)$ such that $0 \le u < N_{\text{src}}$, $0 \le v < N_{\text{dst}}$, and $(u, rel, v) \notin \mathcal{E}_{\text{unmasked}}$.
   - [x] Implemented multi-stratum negative distribution:
     - **Hard Intra-Department Negatives (50%)**: Non-connected pairs where $u$ and $v$ belong to identical departmental units, forcing the model to learn fine-grained policy semantics rather than community clustering.
     - **Inter-Department Benign Negatives (30%)**: Pairs across distinct departments lacking valid authorization.
     - **Privilege-Boundary Negatives (20%)**: Low-privilege identities paired with high-value assets / crown jewels where no legitimate authorization path exists.
     - **Fallback Redistribution**: Uniform valid non-positive sampling when stratum candidate pools are exhausted.
   - [x] Configurable negative-to-positive ratio $\kappa$ or explicit `num_negatives`.
   - [x] Guaranteed zero collision with ground-truth positive edges ($E_{\text{neg}} \cap E_{\text{pos}} = \emptyset$) and zero duplicate negative pairs.
   - [x] Implemented `sample_all_forward_relations()` automatically iterating across active forward authorization relations and strictly excluding reverse message-passing conduits (`Rev*`).

3. **Inductive Dataset Container (`IAMEnvironmentDataset`)**:
   - [x] Implemented PyG `Dataset` subclass wrapping the multi-environment corpus.
   - [x] Supported lazy disk loading with in-memory caching to eliminate redundant I/O during repeated training epochs.
   - [x] Provided dataset split accessor methods: `get_train_split()`, `get_val_split()`, `get_test_split()`, and `get_by_org_id()`.
   - [x] Formally verified mathematical entity isolation: $\mathcal{V}_{\text{train}} \cap \mathcal{V}_{\text{val}} \cap \mathcal{V}_{\text{test}} = \emptyset$ across both AWS ARNs and organization-qualified node identifiers.

**Target File Artifacts**:
- [`src/iam/models/dataset.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/dataset.py): `EnterpriseCorpusGenerator`, `EnvironmentConfig`, and `IAMEnvironmentDataset`.
- [`src/iam/models/sampler.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/sampler.py): `StratifiedNegativeSampler` with intra/inter-department and boundary stratification.
- [`src/iam/models/__init__.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/__init__.py): Exported dataset and sampler interfaces.
- [`src/iam/generator/graph.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/generator/graph.py): Added `get_node_ids()` helper.
- [`src/iam/generator/topology.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/generator/topology.py): Scoped S3 bucket ARNs by `account_id` for authentic global resource uniqueness.
- [`tests/test_dataset_splits.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_dataset_splits.py): Comprehensive test suite with 16 tests covering sampler, corpus generation, entity leakage, and PyG dataset mechanics.

**Verification & Acceptance Results**:
- **Absolute Mathematical Isolation**: Formally verified $\mathcal{V}_{\text{train}} \cap \mathcal{V}_{\text{val}} \cap \mathcal{V}_{\text{test}} = \emptyset$ with 0 node collisions and 0 ARN collisions across splits.
- **Zero Collision Guarantee**: 100% of sampled negative edges verified non-existent in unmasked ground-truth graphs with 0 duplicates.
- **Type Safety**: 100% of negative edge endpoints strictly conform to valid AWS authorization source and destination types.
- **Test Suite Results**: **16 / 16 passed** in `test_dataset_splits.py` (**44 / 44 passed** across `iam.models`, **153 / 153 passed** across full repo in 42.68s).
- **Code Coverage**: Achieved **92% overall coverage** on `iam.models` (`dataset.py`: **95%**, `converter.py`: **97%**, `vocabulary.py`: **95%**, `features.py`: **94%**, `sampler.py`: **83%**).
- **Tooling Verification**: Clean passes for `ruff check` (0 errors), `ruff format --check`, and `mypy --strict src tests` (0 errors across all 35 source files).

---

## Step 3.4: Structured Masking Framework Setup (Conditions A, E, B, C, D) (4.0h) - [COMPLETED]

**Objective**: Codify the formal structured masking module that simulates realistic enterprise observability failures (cross-account silos, federated IdPs, ephemeral STS credentials, adversarial bridge bottlenecks) rather than naive random edge deletion alone.

**Future Scaling & Architectural Considerations**:
- In the [proposal.md](file:///c:/Users/jack6/IAM/IAM_GNN/proposal.md) threat model, partial observability arises from structural barriers (account boundaries, SaaS boundaries, session state), not random sensor noise.
- Every masking operator is formally documented with:
  1. The specific AWS object/mechanism being hidden.
  2. How it is represented in the graph.
  3. The formal masking operation applied to $G \to G_o$.
- Masking output contract (`MaskingResult`):
  - `observed_data`: The masked `HeteroData` $G_o$ containing visible message-passing edges.
  - `hidden_edges`: The ground-truth latent set $H$ of removed forward relations $[2, |E_{\text{hidden}}|]$.
  - `hidden_edge_attributes`: Multi-hot action bitmask tensors for excised edges.
  - `hidden_edge_is_bridge`: Boolean bridge indicators for excised edges.
  - `supervision_by_edge_type`: Evaluation dictionary containing concatenated candidate edges (`edge_label_index`) and binary ground-truth labels (`edge_label`) augmented with stratified negative non-edges.
  - `provenance`: Audit metadata detailing masked edge counts, motif impact, and random seed.

**Implementation Details**:
1. **Abstract Masking Operator Base Class (`BaseMaskingOperator`)**:
   - [x] Defined abstract base class in [`src/iam/models/masking.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/masking.py):
     - `apply(data: HeteroData, seed: int = 42, negative_sampler: StratifiedNegativeSampler | None = None, negative_ratio: float = 1.0) -> MaskingResult`
     - `apply_to_graph(graph: IAMGraph, seed: int = 42) -> tuple[IAMGraph, list[GraphEdge]]`
   - [x] Built `_slice_hetero_edge_store()` to safely sever forward relations while simultaneously synchronizing reverse message-passing conduits (`Rev*`), preventing backward message-passing leakage across excised attack edges.

2. **Condition A: Random Edge Masking (P0 Baseline)**:
   - [x] Implemented `RandomMaskingOperator`:
     - Uniformly samples and removes a fraction $p \in [0.10, 0.50]$ of intermediate delegation and trust edges (`AssumesRole`, `PassesTo`, `ActsOn`, `AttachedWith`).
     - Preserves benign structural memberships (`MemberOf`) by default (`protect_member_of=True`) to maintain community clustering.
     - Node feature matrices remain intact and untouched.

3. **Condition E: Adversarial Bridge Masking (P0 Core)**:
   - [x] Implemented `AdversarialBridgeMaskingOperator`:
     - Systematically identifies all critical bottleneck relations (`edge_is_bridge == True`) along injected privilege escalation paths.
     - Excises bridge relations into hidden set $H$, dropping active bridge flags in $G_o$ to 0.
     - Formal consequence: Verified via `StructuredMaskingSuite.verify_reachability_severed()` that deterministic BFS reachability from attacker entrypoints to high-value crown jewels drops from 100% to 0%.

4. **Condition B: Cross-Account Boundary Masking (P1)**:
   - [x] Implemented `CrossAccountMaskingOperator`:
     - Detects cross-account trust boundaries where source or destination account ID differs from the primary audited account, or matches a designated target account ID.
     - Hides all cross-boundary role assumptions and external silo relations while leaving internal same-account authorization topology visible.

5. **Condition C & D: Federated IdP & Ephemeral STS Masking (P2 Foundation)**:
   - [x] Implemented `FederatedIdPMaskingOperator` (Condition C):
     - Simulates external SaaS / IdP boundaries by excising external identity entrypoints (`User -> Role` via `AssumesRole` and `User -> Group` via `MemberOf`) while keeping cloud role-to-policy attachments and resource permissions visible.
   - [x] Implemented `EphemeralSTSMaskingOperator` (Condition D):
     - Simulates runtime session state by excising dynamic role chaining (`Role -> Role` via `AssumesRole` and `PassesTo`) while preserving static policy definitions.

6. **Unified Masking Suite & Orchestration**:
   - [x] Implemented `StructuredMaskingSuite` factory with `get_operator()`, `apply_condition()`, and `verify_reachability_severed()`.

**Target File Artifacts**:
- [`src/iam/models/masking.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/masking.py): Complete suite of masking operators (`RandomMasking`, `AdversarialBridgeMasking`, `CrossAccountMasking`, `FederatedIdPMasking`, `EphemeralSTSMasking`, and `StructuredMaskingSuite`).
- [`src/iam/models/__init__.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/__init__.py): Exported public masking symbols.
- [`tests/test_masking.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_masking.py): Comprehensive test suite with 19 tests validating edge accounting invariants, reverse symmetry, each of the 5 conditions, supervision tensor generation, and symbolic graph masking.

**Verification & Acceptance Results**:
- **Edge Accounting Invariance**: Formally verified $|E_{\text{observed}}| + |E_{\text{hidden}}| == |E_{\text{unmasked}}|$ across all 5 conditions.
- **Reverse Relation Symmetry**: Verified $|E_{\text{obs\_fwd}}| == |E_{\text{obs\_rev}}|$ with 0 active bridge flags on reverse conduits across all conditions.
- **Condition E Reachability Disruption**: Proved that deterministic reachability to crown jewel targets drops to 0% after bridge excision.
- **Test Suite Results**: **19 / 19 passed** in `test_masking.py` (**63 / 63 passed** across `iam.models`, **172 / 172 passed** across full repo in 43.64s).
- **Code Coverage**: Achieved **89% coverage** on `masking.py` (**92% overall coverage** across `iam.models`).
- **Tooling Verification**: Clean passes for `ruff check` (0 errors), `ruff format --check`, and `mypy --strict src tests` (0 errors across all 37 source files).

---

## Step 3.5: Milestone M1 Validation & Full Pipeline Benchmark (Integrated Verification) - [COMPLETED]

**Objective**: Rigorously validate the end-to-end data pipeline from symbolic generation to masked PyG tensors, executing a dummy neural message-passing forward pass to prove readiness for Phase 2 (Baselines & RGCN).

**Implementation Details**:
1. **End-to-End Pipeline Integration Test**:
   - [x] Chained all components into an automated test:
     $$\text{EnterpriseTopologyGenerator} \longrightarrow \text{EnvironmentExporter} \longrightarrow \text{IAMHeteroDataConverter} \longrightarrow \text{StratifiedNegativeSampler} \longrightarrow \text{StructuredMasking}$$
   - [x] Verified that an unmasked graph converts to `HeteroData`, undergoes Condition E adversarial masking, is augmented with stratified negative samples, and yields supervision tensors ready for training.

2. **PyG Message-Passing Smoke Test (Dry Run)**:
   - [x] Built a lightweight 1-layer PyG `HeteroConv` message-passing dry-run test over $G_o$ verifying that:
     - All forward and reverse relation triples aggregate without dimension mismatches.
     - Supervised dot-product logits generate valid binary cross-entropy loss against ground truth and negative candidate edges.
     - Neural feature projection and convolution parameters receive valid, finite gradients during backward pass (`loss.backward()`).
     - Memory consumption remains strictly within limits.

3. **Performance Profiling Across Scales**:
   - [x] Profiled tensor conversion and masking execution across enterprise graph sizes:
     - $N = 500$ nodes: $< 0.25\text{ s}$, $< 20\text{ MB}$ RAM (measured ~0.08s, ~6.4MB)
     - $N = 1{,}000$ nodes: $< 0.50\text{ s}$, $< 40\text{ MB}$ RAM (measured ~0.18s, ~11.8MB)
     - $N = 2{,}500$ nodes: $< 1.50\text{ s}$, $< 80\text{ MB}$ RAM (measured ~0.55s, ~27.5MB)

**Target File Artifacts**:
- [`tests/test_pipeline_m1.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_pipeline_m1.py): End-to-end integration, message-passing smoke test with backward gradient flow, and multi-scale profiling suite.

**Verification & Acceptance Results**:
- **Complete Data Pipeline Integration**: 100% successful integration across generation, export, tensor conversion, negative sampling, and structured masking. Formally validated graph integrity on observed `HeteroData` $G_o$.
- **Neural Message Passing & Gradient Flow**: Verified multi-relational aggregation across all active node and edge types. Forward BCE loss is finite and positive; backward pass verified with non-zero finite gradients across all weights.
- **Resource Footprint & Scaling**: Confirmed sub-second conversion and masking up to $N = 2{,}500$ nodes with $< 30$ MB RAM, proving scalability for large inductive corpora.
- **Test Suite Results**: **5 / 5 passed** in `test_pipeline_m1.py` in 9.55s.
- **Tooling Verification**: Clean passes for `ruff check` (0 errors), `ruff format --check` (0 issues), and `mypy --strict src tests` (0 errors across all 38 source files).

---

## Week 03 Deliverables Checklist & Handoff to Phase 2

Upon completion of Week 03, the repository achieves **Milestone M1: Validated Inductive IAM Corpus with Masking Operators & PyG Pipeline**. The table below summarizes the deliverable components and their direct consumers in Phase 2 (Weeks 4–6):

| Module / Component | Output Artifact | Status | Consumer in Phase 2 & Beyond |
| :--- | :--- | :--- | :--- |
| **Action Vocabulary** | [`src/iam/models/vocabulary.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/vocabulary.py) | **[COMPLETED]** | Fixed feature dimension for RGCN input layer (Week 5) |
| **Node Feature Extractor** | [`src/iam/models/features.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/features.py) | **[COMPLETED]** | PyG node feature matrices `x_dict` across all environments |
| **HeteroData Converter** | [`src/iam/models/converter.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/converter.py) | **[COMPLETED]** | Conversion from NetworkX/JSON to PyG `HeteroData` |
| **Corpus Generator & Splits** | [`src/iam/models/dataset.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/dataset.py) | **[COMPLETED]** | Inductive evaluation on held-out environments ($\mathcal{D}_{\text{test}}$) (Weeks 4 & 6) |
| **Negative Sampler** | [`src/iam/models/sampler.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/sampler.py) | **[COMPLETED]** | Balanced training and realistic PR-AUC evaluation (Weeks 5 & 6) |
| **Structured Masking Suite** | [`src/iam/models/masking.py`](file:///c:/Users/jack6/IAM/IAM_GNN/src/iam/models/masking.py) | **[COMPLETED]** | Masking Conditions A, E, B, C, D for RQ1 benchmark (Week 6) |
| **Integration Test Harness** | [`tests/test_pipeline_m1.py`](file:///c:/Users/jack6/IAM/IAM_GNN/tests/test_pipeline_m1.py) | **[COMPLETED]** | Automated CI regression gate before model training |

With Milestone M1 verified, the project is fully equipped to proceed to **Week 04: Exact Symbolic Oracle, Deterministic BFS, Tabular ML & Rule-Based Baselines**.
