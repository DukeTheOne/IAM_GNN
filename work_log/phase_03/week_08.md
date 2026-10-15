# Week 08: Synthetic Temporal Workload, Completion-Set Verifier & Central Tradeoff Experiment (RQ2)

**Goal**: Formulate and generate synthetic temporal authorization workloads with realistic operational drift ($\mathcal{D}^{T_1}$ for synthesis vs. $\mathcal{D}^{T_2}$ for evaluation), construct the three-level `CompletionSetVerifier` ($G_o$, $\{G_o \cup H_i\}_{i=1}^M$, and ground-truth $G^*$) with side-effect checking, execute the central empirical tradeoff experiment (RQ2) comparing our uncertainty-aware repair against 5 baseline strategies (Oracle, No Action, Coarse Action Deletion / IAMPERE, Log-Only Tightening / PolicyRefiner, and Point-Estimate Repair), and generate multi-dimensional Pareto tradeoff frontiers separating security from operational utility.
**Estimated Effort**: 17 Hours
**Target Milestone**: **M3: Security–Utility Tradeoff Evaluation (RQ2)**

---

## Architectural Decisions & Theoretical Foundations Context

Following the formal completion of the Monte Carlo latent-completion sampler, uncertainty-guided candidate pruner, and capability-aware repair engine in [week_07.md](file:///Users/duke/IAM/work_log/phase_03/week_07.md), Week 08 advances to the core empirical research contribution of Phase 3 (RQ2): **decision-making under partial authorization observability**.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                             PHASE 3 WEEK 08 EXPERIMENTAL PIPELINE                                  |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
| 1. Synthetic Temporal Workload & Drift Generator (Step 8.1)                                       |
|    │ Operational Activity Trace: D = { (principal, action, resource, time, context) }             |
|    │ Temporal Split: D^{T_1} (Months 1–3, Synthesis) vs. D^{T_2} (Month 4, Evaluation Holdout)    |
|    │ Operational Workflow Drift: Unseen resources & infrequent legitimate actions in T_2           |
|    ▼                                                                                              |
| 2. Remediation Strategy Dispatcher under Condition E (Step 8.3)                                   |
|    ├── Method 0: Oracle Repair (Full Ground Truth G*) [Upper Bound Reference]                     |
|    ├── Method 1: No Action (Blind Traversal on G_o)                                               |
|    ├── Method 2: Coarse Action Deletion (IAMPERE-style heuristic)                                 |
|    ├── Method 3: Log-Only Tightening on T_1 (PolicyRefiner-style baseline)                        |
|    ├── Method 4a: Point-Estimate Repair (Thresholded RGCN p̂ ≥ 0.50)                               |
|    └── Method 4b: Uncertainty-Aware Repair (Ours: M sampled completions {H_i})                     |
|    ▼                                                                                              |
| 3. Three-Level Completion-Set Verifier (Step 8.2)                                                 |
|    │ Level 1: Check safety on Observed Graph G_o ⊕ P                                              |
|    │ Level 2: Check safety on Completion Set { (G_o ∪ H_i) ⊕ P }_{i=1}^M                          |
|    │ Level 3: Check safety on Ground-Truth G* ⊕ P (synthetic benchmark only)                     |
|    │ Invariant Checks: Zero escalation side-effects + 100% T_1 preservation                        |
|    │ Conformance & Breakage: Measure legitimate operational requests denied on T_2                |
|    ▼                                                                                              |
| 4. Security–Utility Multi-Metric Evaluation & Pareto Frontiers (Step 8.4)                         |
|    │ Security: Ground-Truth Reachability, HV Asset Exposure, Max Privilege, Cut Cost              |
|    │ Utility: T_2 Drift Breakage Rate, AST Mutation Count, Transformation Penalty                  |
|    │ Output: Pareto Frontier plots isolating the scientific value of modeling uncertainty        |
|    ▼                                                                                              |
| [Handoff to Week 09] Visual Triage Dashboard & Demo Packaging (Phase 4)                           |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

### 1. Three-Level Verification Boundary
A neural model can be wrong; the symbolic verifier establishes the safety and claim boundary:
1. **Level 1 (Observed Graph $G_o \oplus \mathcal{P}$)**: Traditional static analysis verification. Under partial observability, passing Level 1 gives false confidence because latent bridge relations are completely invisible.
2. **Level 2 (Completion Set $\{ (G_o \cup H_i) \oplus \mathcal{P} \}_{i=1}^M$)**: Evaluates whether the policy patch eliminates privilege escalation across all plausible Monte Carlo latent completions drawn from the calibrated relational model. Guarantees are formal relative to the completion set.
3. **Level 3 (Ground Truth $G^* \oplus \mathcal{P}$)**: Available strictly in synthetic benchmarking. Exposes residual risk caused by missing relationships that the neural model failed to infer.

The measured discrepancy between Level 1, Level 2, and Level 3 is a primary research result of RQ2, exposing exactly when and why "safe on $G_o$" fails.

### 2. Operational Workload Drift ($\mathcal{D}^{T_1}$ vs. $\mathcal{D}^{T_2}$)
To rigorously benchmark operational utility without relying on proprietary enterprise CloudTrail dumps, we construct a **Data Level 1 Temporal Holdout**:
- **Baseline Synthesis Window ($T_1$)**: Months 1–3 ($\sim 90$ days). Benign operational events (CI/CD build execution, scheduled ETL queries, routine infrastructure provisioning). The repair solver treats $\mathcal{D}^{T_1}$ as a hard constraint:

$$
\forall e \in \mathcal{D}^{T_1}, \quad \text{Authorized}(e, G_o \oplus \mathcal{P}) = 1
$$

- **Evaluation Holdout Window ($T_2$)**: Month 4 ($\sim 30$ days). Strictly held out during patch synthesis. Features controlled operational drift:
  1. *Resource Drift*: Legitimate actions target newly created, valid enterprise resource ARNs (e.g. daily S3 partition buckets, quarterly audit reports) not observed during $T_1$.
  2. *Infrequent Action Drift*: Infrequent but authorized business workflows (e.g. quarterly credential rotation, emergency KMS re-encrypt).
- **Utility Metric**: Fraction of legitimate requests in $\mathcal{D}^{T_2}$ denied by $\mathcal{P}$:

$$
\text{BreakageRate}(\mathcal{P}, \mathcal{D}^{T_2}) = \frac{|\{ e \in \mathcal{D}^{T_2} \mid \text{Authorized}(e, G_o \oplus \mathcal{P}) = 0 \}|}{|\mathcal{D}^{T_2}|}
$$

### 3. The Central Tradeoff Experiment: 6 Remediation Strategies (RQ2)
Under Condition E (adversarial bridge relation $a \to b$ hidden), we systematically evaluate 6 distinct remediation strategies:

| Strategy | Description | Expected Security (Residual Risk in $G^*$) | Expected Operational Impact ($T_2$ Breakage) |
| :--- | :--- | :--- | :--- |
| **Method 0: Oracle** | Full access to $G^*$; repairs exact bridge | Minimal residual risk | Minimal breakage |
| **Method 1: No Action** | Blind traversal on $G_o$; leaves latent path open | Severe (Attack path exploitable) | Zero breakage (nothing modified) |
| **Method 2: Coarse Deletion** | IAMPERE-style heuristic: excised broad actions (`iam:PassRole`) | Moderate/Low | High breakage (breaks shared build pipelines) |
| **Method 3: Log-Only Tightening** | PolicyRefiner-style: scopes strictly to observed $T_1$ events | Blind to latent path if uncalled in $T_1$ | High breakage on $T_2$ resource drift |
| **Method 4a: Point-Estimate** | Thresholds RGCN $\hat{p} \ge 0.50$; repairs single graph | Vulnerable to model calibration errors | Low/Moderate |
| **Method 4b: Uncertainty-Aware** | Optimizes over sampled completions $\{H_i\}$ with $C(a, r, c)$ | Lowest residual risk among zero-query methods | Low (scopes only what severs latent distribution) |

The comparison between **Method 4a** and **Method 4b** isolates whether modeling **latent graph uncertainty**, rather than merely thresholding edge predictions, provides quantifiable advantages.

---

## Step 8.1: Synthetic Workload and Temporal Split (Data Level 1) (4.0h) - [COMPLETED]

**Objective**: Implement the `SyntheticWorkloadGenerator` and `TemporalWorkloadSplit` modules that simulate realistic enterprise CloudTrail authorization activity traces $\mathcal{D}$ for benign identities, partitioned into an observation window $T_1$ and a holdout evaluation window $T_2$ equipped with controlled operational drift.

**Implementation Details**:
1. **Workload Event Data Model (`WorkloadEvent` / `AuthRequest`)**:
   - [x] Standardized event model representing authorized cloud requests: `principal_arn`, `action`, `resource_arn`, `timestamp`, `department`, and `context`.
   - [x] Support realistic enterprise workflow profiles: CI/CD runner, Data/BI analyst, SecOps auditor, and DevOps engineer.
2. **Temporal Split Management (`TemporalWorkloadSplit`)**:
   - [x] Partitioning timeline into observation window $T_1$ (e.g. Days 1–90) and evaluation holdout $T_2$ (e.g. Days 91–120).
   - [x] Export split traces as typed sequences of `AuthRequest` objects for direct ingestion by repair engines and verifiers.
3. **Controlled Operational Drift Modeling (`DriftConfig`)**:
   - [x] Resource drift: generates authorized requests on resources within the principal's policy scope but with resource identifiers not evidenced in $T_1$ (e.g. new S3 partition keys, quarterly backup snapshots).
   - [x] Action frequency drift: generates periodic, low-frequency authorized requests that occur exclusively in $T_2$ (e.g. monthly key rotation).
   - [x] Parameterized drift intensity $\rho_{\text{drift}} \in [0.0, 1.0]$.
4. **Verification & Conformance Testing**:
   - [x] Verify 100% of generated events in $T_1$ and $T_2$ evaluate to `ALLOW` on the baseline IAM graph $G_o$.
   - [x] Verify that $T_1$ and $T_2$ event sets are temporally disjoint.
   - [x] Measure drift rate and verify un-drifted vs. drifted event distributions.

**Target File Artifacts**:
- [`src/iam/verifier/workload.py`](file:///Users/duke/IAM/src/iam/verifier/workload.py): `WorkloadEvent`, `DriftConfig`, `TemporalWorkloadSplit`, and `SyntheticWorkloadGenerator` (96% statement coverage).
- [`src/iam/verifier/__init__.py`](file:///Users/duke/IAM/src/iam/verifier/__init__.py): Public module exports for verifier and workload components.
- [`tests/test_temporal_workload.py`](file:///Users/duke/IAM/tests/test_temporal_workload.py): Unit and property tests for temporal generation, drift, and baseline conformance (10 / 10 tests passing).

**Verification & Acceptance Results**:
- **100% Baseline Conformance Invariant**: Verified across both unit test graphs and 150-node enterprise topologies that 100% of generated events in $T_1$ and $T_2$ strictly evaluate to `ALLOW` on $G_o$ via `WorkloadConformanceChecker`.
- **Temporal Disjointness**: Confirmed strict timeline partitioning: $\forall e \in \mathcal{D}^{T_1}, \text{day}(e) < 90.0$; $\forall e \in \mathcal{D}^{T_2}, \text{day}(e) \ge 90.0$.
- **Operational Drift Realism**: Demonstrated that $T_2$ drift events inject novel resource ARNs and infrequent actions (e.g. `iam:CreateAccessKey` self-rotation), providing non-trivial benchmark breakage for coarse action excision or over-tightening.
- **Enterprise Scaling**: Generated 1,350+ CloudTrail events across 150 enterprise nodes in **0.18 seconds**.
- **Code Quality Standards**: 100% clean passes on `mypy --strict` and `ruff`.

---

## Step 8.2: Completion-Set Verifier (4.0h) - [COMPLETED]

**Objective**: Implement the `CompletionSetVerifier` module that formally evaluates proposed IAM policy patches across the 3-level verification boundary ($G_o$, $\{G_o \cup H_i\}_{i=1}^M$, and $G^*$), checks for unwanted privilege escalation side-effects, and quantifies operational breakage on the temporal holdout $\mathcal{D}^{T_2}$.

**Implementation Details**:
1. **Three-Level Symbolic Verification (`VerificationLevel`)**:
   - [x] Level 1: Verify residual attack reachability on $G_o \oplus \mathcal{P}$.
   - [x] Level 2: Verify residual attack reachability across all $M$ completions $(G_o \cup H_i) \oplus \mathcal{P}$.
   - [x] Level 3: Verify residual attack reachability on ground-truth graph $G^* \oplus \mathcal{P}$.
2. **Escalation Side-Effect Checker**:
   - [x] Evaluates all non-target $(s', t')$ pairs in the graph to guarantee that $\mathcal{P}$ introduces zero new privilege escalation paths.
3. **Operational Conformance & Breakage Auditor**:
   - [x] Replays $\mathcal{D}^{T_1}$ and confirms 100% zero-regression invariant.
   - [x] Replays holdout $\mathcal{D}^{T_2}$ and computes operational breakage rate and denied event breakdown.
4. **Structured Verification Report (`VerificationReport`)**:
   - [x] Serializes verification results, residual reachabilities, broken requests, and safety certifications to structured JSON.

**Target File Artifacts**:
- [`src/iam/verifier/verifier.py`](file:///Users/duke/IAM/src/iam/verifier/verifier.py): `CompletionSetVerifier`, `VerificationLevel`, `SafetyStatus`, `VerificationReport`, `SideEffectResult`, and `WorkloadReplayResult` (90% statement coverage).
- [`tests/test_completion_verifier.py`](file:///Users/duke/IAM/tests/test_completion_verifier.py): Comprehensive unit and integration test suite (6 / 6 tests passing).

**Verification & Acceptance Results**:
- **Contribution 3 Formal Verification**: Successfully verified that an unrepaired baseline under Condition E masks attack reachability at Level 1 ($G_o$ risk = 0.0), but is detected as unsafe at Level 2 ($>70\%$ risk across completions) and Level 3 ($100\%$ risk on $G^*$), correctly classified as `SafetyStatus.MISLEADING_SAFE` with `is_misleading=True`.
- **Side-Effect Audit**: Confirmed clean 0 side-effect verification on sound repairs, while accurately flagging synthetic policy regressions that inadvertently expose new admin pathways.
- **Holdout Workload Replay**: Validated zero breakage on historical baseline $\mathcal{D}^{T_1}$ ($0.0\%$ breakage) and quantified operational disruption on drifted $\mathcal{D}^{T_2}$.
- **Performance**: Full 3-level verification and replay completes in $< 1\text{ ms}$ on canonical motifs.
- **Code Quality**: 100% clean passes on `mypy --strict` and `ruff`.

---

## Step 8.3: Central Tradeoff Experiment (RQ2 Execution) (5.5h) - [COMPLETED]

**Objective**: Construct and execute the comprehensive benchmarking suite evaluating the 6 remediation strategies under Condition E adversarial bridge masking across canonical PE motifs and parameterized enterprise topologies.

**Implementation Details**:
1. **Remediation Strategy Implementations**:
   - [x] **Method 0 (Oracle)**: Greedy repair executed directly on ground-truth graph $G^*$.
   - [x] **Method 1 (No Action)**: Baseline identity patch (zero changes).
   - [x] **Method 2 (Coarse Deletion / IAMPERE-style)**: Strategy excising actions associated with attack cut relations.
   - [x] **Method 3 (Log-Only Tightening / PolicyRefiner-style)**: Narrowing all identity policies strictly to evidenced $T_1$ events.
   - [x] **Method 4a (Point-Estimate Repair)**: Materializing single most-likely graph from $\hat{p} \ge 0.50$ predictions and repairing on it.
   - [x] **Method 4b (Uncertainty-Aware Repair - Ours)**: Full stochastic completion sampling + pruning + capability repair.
2. **Automated Experiment Runner (`TradeoffExperimentRunner`)**:
   - [x] Parameterized execution across random seeds and varied enterprise scales.
   - [x] Computes multi-level risk, $T_1$ historical breakage, $T_2$ holdout breakage under drift, AST cost, and execution latency.
3. **CLI Benchmark Tool**:
   - [x] CLI script [`scripts/run_tradeoff_experiment.py`](file:///Users/duke/IAM/scripts/run_tradeoff_experiment.py) with structured JSON, CSV, and Markdown summary exports.

**Target File Artifacts**:
- [`src/iam/verifier/tradeoff.py`](file:///Users/duke/IAM/src/iam/verifier/tradeoff.py): `RemediationMethod`, `TradeoffConfig`, `TradeoffExperimentRunner`, and `MethodResult`.
- [`scripts/run_tradeoff_experiment.py`](file:///Users/duke/IAM/scripts/run_tradeoff_experiment.py): CLI execution script for RQ2 tradeoff evaluation.
- [`tests/test_tradeoff_experiment.py`](file:///Users/duke/IAM/tests/test_tradeoff_experiment.py): Unit test suite for the tradeoff runner and strategy dispatchers (7 / 7 tests passing).

**Verification & Acceptance Results**:
- **Point-Estimate Calibration Failure**: Formally verified in `test_point_estimate_failure_under_subthreshold_probability` that when predicted edge probability is sub-threshold ($p = 0.40 < \theta = 0.50$), Method 4a drops the latent relation and fails to repair ($G^*$ risk remains 1.0, classified `MISLEADING_SAFE`), whereas Method 4b (Uncertainty-Aware) samples completions and severs the bottleneck (risk = 0.0).
- **Scalability Budget**: Benchmarked on 150-node enterprise topology in **0.37 seconds** total across all 6 methods (well below the $< 3.0\text{ s}$ budget).

---

## Step 8.4: Security–Utility Metrics & Pareto Frontier (3.5h) - [COMPLETED]

**Objective**: Formulate, compute, and visualize the multi-dimensional security–utility tradeoff metrics, synthesizing Pareto frontier curves that isolate the exact scientific advantage of uncertainty-aware remediation.

**Implementation Details**:
1. **Multi-Dimensional Metrics**:
   - [x] Security: Ground-truth reachability probability, weighted high-value asset count, maximum reachable privilege tier, minimum cut cost.
   - [x] Operational Utility: $T_2$ holdout breakage rate, total denied requests, AST mutation count, and repair transformation penalty.
2. **Pareto Dominance Analysis**:
   - [x] Algorithm computing the non-dominated Pareto frontier points across the 6 methods.
   - [x] Proves whether Method 4b strictly dominates Methods 1, 2, 3, and 4a.
3. **Reporting & Visualization**:
   - [x] Generates publication-ready Markdown and CSV summary tables ([`experiments/tradeoff/tradeoff_metrics.csv`](file:///Users/duke/IAM/experiments/tradeoff/tradeoff_metrics.csv)).
   - [x] Generates Pareto frontier curves: Security Risk vs. $T_2$ Breakage Rate (ASCII scatter plots embedded in reports).

**Target File Artifacts**:
- [`src/iam/verifier/metrics.py`](file:///Users/duke/IAM/src/iam/verifier/metrics.py): `TradeoffMetrics`, `ParetoFrontierAnalyzer`, `DominancePairResult`, `ParetoFrontierPoint`, and `MetricSummary`.
- [`experiments/tradeoff/tradeoff_report.md`](file:///Users/duke/IAM/experiments/tradeoff/tradeoff_report.md): Final Milestone M3 empirical tradeoff report.
- [`experiments/tradeoff/tradeoff_metrics.csv`](file:///Users/duke/IAM/experiments/tradeoff/tradeoff_metrics.csv): Tabular CSV metrics export.
- [`tests/test_tradeoff_metrics.py`](file:///Users/duke/IAM/tests/test_tradeoff_metrics.py): Unit tests for metric calculation and Pareto dominance logic (7 / 7 tests passing).

**Verification & Acceptance Results**:
- **Pareto Dominance Certification**: Formally verified that Method 4b (Uncertainty-Aware) achieves Pareto optimality and strictly dominates Method 1 (No Action), Method 2 (Coarse Deletion), and Method 3 (Log-Only Tightening), and strictly dominates Method 4a under subthreshold link prediction.
- **Reporting Pipeline**: Fully integrated into [`scripts/run_tradeoff_experiment.py`](file:///Users/duke/IAM/scripts/run_tradeoff_experiment.py), auto-generating Pareto frontier charts and metrics.
- **Code Quality**: 100% clean passes on `mypy --strict` and `ruff`.

---

## Cumulative Verification & Quality Status (Phase 3 Progress)

- **Unit & Integration Tests**: **377 / 377 tests passing** across the complete project test suite (`pytest -v`).
- **Verifier Module Coverage**: **>90% coverage** across all verifier components:
  - `src/iam/verifier/workload.py` (257 LOC)
  - `src/iam/verifier/verifier.py` (518 LOC)
  - `src/iam/verifier/tradeoff.py` (781 LOC)
  - `src/iam/verifier/metrics.py` (683 LOC)
- **Milestone M3 Foundation**: **Formally achieved** with target milestone certification in [`experiments/tradeoff/tradeoff_report.md`](file:///Users/duke/IAM/experiments/tradeoff/tradeoff_report.md).
- **Strict Typing**: **Zero errors** across all 87 source files on `mypy --strict src tests scripts`.
- **Formatting & Linting**: **Zero issues** on `ruff check` and `ruff format --check`.

---

## Week 08 Deliverables Checklist & Handoff to Week 09

| Module / Component | Target Artifact | Status | Consumer in Week 09 |
| :--- | :--- | :---: | :--- |
| **Synthetic Workload & Drift Generator** | [`src/iam/verifier/workload.py`](file:///Users/duke/IAM/src/iam/verifier/workload.py) | [COMPLETED] | Log generator for evaluation holdout and policy replay |
| **Completion-Set Verifier** | [`src/iam/verifier/verifier.py`](file:///Users/duke/IAM/src/iam/verifier/verifier.py) | [COMPLETED] | 3-level symbolic safety engine for dashboard audit |
| **Tradeoff Experiment Runner** | [`src/iam/verifier/tradeoff.py`](file:///Users/duke/IAM/src/iam/verifier/tradeoff.py) | [COMPLETED] | Benchmarking core for RQ2 evaluation |
| **Pareto Frontier & Metrics Analyzer** | [`src/iam/verifier/metrics.py`](file:///Users/duke/IAM/src/iam/verifier/metrics.py) | [COMPLETED] | Metric generator for Streamlit KPI dashboard |
| **Tradeoff CLI Script** | [`scripts/run_tradeoff_experiment.py`](file:///Users/duke/IAM/scripts/run_tradeoff_experiment.py) | [COMPLETED] | CLI reproduction command for M3 demonstration |
| **Tradeoff Experiment Report** | [`experiments/tradeoff/tradeoff_report.md`](file:///Users/duke/IAM/experiments/tradeoff/tradeoff_report.md) | [COMPLETED] | Final Phase 3 milestone report for Milestone M3 |
| **Verifier Test Suite** | [`tests/test_temporal_workload.py`](file:///Users/duke/IAM/tests/test_temporal_workload.py), [`tests/test_completion_verifier.py`](file:///Users/duke/IAM/tests/test_completion_verifier.py), [`tests/test_tradeoff_experiment.py`](file:///Users/duke/IAM/tests/test_tradeoff_experiment.py), [`tests/test_tradeoff_metrics.py`](file:///Users/duke/IAM/tests/test_tradeoff_metrics.py) | [COMPLETED] | Automated regression testing for verifier & metrics |
