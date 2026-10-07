# Week 02: Canonical PE Motifs & Parameterized Enterprise Cloud Generator

**Goal**: Implement 5–8 high-frequency privilege escalation motifs with exact ground truth, and build a highly optimized, parameterized enterprise cloud generator that scales to massive organizational graphs.
**Estimated Effort**: 17 Hours

---

## Optimization & Future Scaling Context

Before beginning implementation, consider the following design principles to ensure the codebase can scale gracefully in future phases (e.g., up to 5000+ node graphs, Tier 2 real IaC parsing, and TAC-Bench integration):
- **Graph Processing Efficiency**: NetworkX can be slow for massive graphs. Use it primarily for structural definition, but pre-calculate / vectorize adjacency operations where possible, or structure data efficiently for fast transition to PyG `HeteroData` in Week 3.
- **Memory Footprint**: The generator will scale up to 5000+ nodes and hundreds of thousands of edges. Use generators/iterators and avoid large intermediate lists in memory during the topology generation process.
- **Modularity & Extensibility**: The PE Motif definitions must be modular (e.g., using a Strategy pattern or base classes) so adding 10-15 more motifs in Tier 2 (Month 4-6) requires zero changes to the core generator engine.
- **Deterministic Seeding**: The generation process MUST be fully deterministic given a seed to ensure reproducibility in evaluation and to maintain valid environment splits.

---

## Step 2.1: Canonical PE Motifs with Exact Ground Truth (5.0h) - [COMPLETED]

**Objective**: Define the core subgraph patterns for Privilege Escalation (PE) that the generator will inject into benign environments.

**Implementation Instructions**:
1. **P0 Motifs (Required - 4 Motifs)**:
   - [x] Motif 1: `iam:PassRole` + `lambda:CreateFunction` + `lambda:InvokeFunction`
   - [x] Motif 2: `iam:CreateAccessKey` targeting an existing high-privilege user
   - [x] Motif 3: `iam:AttachRolePolicy` / `iam:AttachUserPolicy` attaching `AdministratorAccess`
   - [x] Motif 4: `sts:AssumeRole` multi-hop chaining across intermediate service roles
2. **P1 Motifs (Optional - Only if on schedule)**:
   - [x] Motif 5: `iam:PassRole` + `ec2:RunInstances` (instance profile elevation)
   - [x] Motif 6: `iam:SetDefaultPolicyVersion`
3. **Motif Architecture & Optimization**:
   - [x] Create an extensible Motif Base Class (e.g. `src/iam/generator/motifs.py`) ensuring each motif defines its multi-step dependencies explicitly, rather than trivial single-hop edges.
   - [x] For each motif, clearly define and tag its **bridge relation** (the critical relationship that the adversarial masking, Condition E, will hide later).

**Execution Log & Completed Artifacts**:
- **Heterogeneous Graph Data Structure**: Implemented [`src/iam/generator/graph.py`](file:///Users/duke/IAM/src/iam/generator/graph.py) providing typed node entities (`User`, `Role`, `Policy`, `Group`, `Resource`), multi-relational directed edges (`MemberOf`, `AssumesRole`, `AttachedWith`, `ActsOn`, `PassesTo`), and an optimized `IAMGraph` class backed by NetworkX `MultiDiGraph` featuring $O(1)$ indexing by ID, ARN, and NodeType, JSON serialization/deserialization, ego-network extraction, and deep cloning.
- **Extensible Motif Architecture**: Implemented [`src/iam/generator/motifs.py`](file:///Users/duke/IAM/src/iam/generator/motifs.py) utilizing the Strategy & Registry patterns:
  - `PEMotif`: Abstract base class enforcing capability-checked policy generation, multi-step subgraphs, and automated ground-truth reachability verification.
  - `MotifRegistry`: Dynamic registry enabling seamless registration of future Tier 2 PE motifs without changing generator logic.
- **Canonical PE Motifs Codified**:
  - `PassRoleLambdaMotif` (P0): `iam:PassRole` + Lambda execution; bridge relation `(source, admin_role, PassesTo)`.
  - `CreateAccessKeyMotif` (P0): `iam:CreateAccessKey` targeting high-privilege user; bridge relation `(source, admin_user, ActsOn)`.
  - `AttachPolicyMotif` (P0): `iam:AttachUserPolicy` attaching admin policy to self; bridge relation `(source, admin_policy, ActsOn)`.
  - `AssumeRoleChainMotif` (P0): Multi-hop `sts:AssumeRole` across intermediate roles; bridge relation `(r1, r2, AssumesRole)`.
  - `PassRoleEC2Motif` (P1): `iam:PassRole` + `ec2:RunInstances` instance profile elevation; bridge relation `(source, admin_role, PassesTo)`.
  - `SetDefaultPolicyVersionMotif` (P1): Policy rollback to dormant admin version; bridge relation `(source, cust_policy, ActsOn)`.
- **Exact Ground Truth & Counterfactual Verification**: Every motif implements `verify_ground_truth(graph, instance)` which formally verifies that:
  1. A valid escalation path $s \rightsquigarrow t$ exists in the unmasked graph.
  2. Temporarily severing the designated bridge relation breaks reachability, guaranteeing that the bridge edge represents the true bottleneck relation for Condition E adversarial masking.
- **AWS Semantic Capability Compliance**: Every generated IAM policy statement across all 6 motifs is validated against `CapabilityModel` from Week 1 to guarantee zero invalid ARN scoping or illegal wildcard actions.
- **Unit & Integration Tests**: Implemented [`tests/test_motifs.py`](file:///Users/duke/IAM/tests/test_motifs.py) with 14 comprehensive tests:
  - Graph indexing, serialization round-trips, cloning, edge removal, and ego-graph slicing.
  - Independent injection and verification for all 4 P0 and 2 P1 motifs.
  - Multi-motif coexistence test injecting all 6 motifs simultaneously into a single graph with zero cross-chain interference.
  - **Test Suite Results**: **64 / 64 passed** across the entire repository (94% coverage, 91% on `motifs.py`, 90% on `graph.py`).
  - **Tooling Verification**: Clean passes for `ruff`, `ruff-format`, `mypy --strict`, and all `pre-commit` hooks.

---

## Step 2.2: Parameterized Enterprise Topology Generator (6.5h)

**Objective**: Build a scalable graph generator that simulates real-world enterprise AWS environments with configurable complexity.

**Implementation Instructions**:
1. **Organizational Hierarchy**:
   - [ ] Implement realistic departmental boundaries (e.g., DevOps, Data/BI, SecOps, QA, Billing, Interns) utilizing efficient subgraph combination methods.
2. **Identity Distribution & Workflows**:
   - [ ] Model power-law identity distribution (small core of Admin/SecOps, moderate DevOps, large long-tail of restricted read-only roles). Use fast NumPy/SciPy distributions to sample scales.
   - [ ] Incorporate benign business workflows (e.g., CI/CD build runners with scoped S3/ECR permissions, read-only analysts querying Athena with strict KMS access).
3. **Graph Scaling Parameters**:
   - [ ] Support parameterization: $N \in [500, 5000]$ nodes with configurable edge density.
   - [ ] Optimize the edge generation using vectorized probability checks rather than `for` loop bottlenecks where possible.
4. **PE Chain Injection**:
   - [ ] Inject positive privilege escalation chains with configurable path lengths (2-hop, 3-hop, 4-hop, and branching paths).
   - [ ] Ensure injected chains are deeply embedded within the noise of benign workflows.

**Execution Log & Completed Artifacts**:
- *[To be filled during execution]*

---

## Step 2.3: Ground-Truth Verification & Environment Export (3.5h)

**Objective**: Label the generated graphs accurately and export them efficiently for downstream PyG/ML pipelines.

**Implementation Instructions**:
1. **Automated Ground-Truth Labeler**:
   - [ ] Implement a highly optimized ground-truth labeler that outputs all reachable $(s, t)$ privilege escalation pairs.
   - [ ] Record the exact sequence of intermediate edges for each target pair. Use fast path-finding algorithms (e.g., NetworkX `all_simple_paths` with depth limit, or custom BFS).
2. **Topological Sanity Checks**:
   - [ ] Compute structural metrics: in/out-degree distributions, clustering coefficients, and connected components.
3. **Scalable Export Pipeline**:
   - [ ] Export labeled graph datasets into structured JSON / NetworkX schemas.
   - [ ] Ensure serialization is memory-efficient (e.g., using `orjson` or incremental file writing) to handle large graphs.

**Execution Log & Completed Artifacts**:
- *[To be filled during execution]*

---

## Step 2.4: Generator Sanity Tests (2.0h)

**Objective**: Provide a robust test suite to guarantee the generator's mathematical and semantic correctness before it produces the training corpus.

**Implementation Instructions**:
1. **Syntactic & Traversable Verification**:
   - [ ] Verify that injected PE chains are syntactically valid according to the parser and capability model from Week 1.
   - [ ] Ensure the paths are actually traversable under formal IAM evaluation rules.
2. **False Positive Checks**:
   - [ ] Ensure benign identities without escalation paths are not falsely annotated as positive escalation targets.
3. **Performance Profiling**:
   - [ ] Measure and document generator runtime and memory usage across node scales (500, 1,000, 2,500, 5,000 nodes). Enforce runtime limits via assertions to ensure scaling limits are respected.

**Execution Log & Completed Artifacts**:
- *[To be filled during execution]*
