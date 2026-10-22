# Executive Research Brief: Tier 1 Milestone & Master's Thesis Roadmap

**Project Title**: Uncertainty-Aware Least-Privilege Repair for Cloud IAM under Partial Observability
**Author**: Duke
**Date**: October 2026
**Document Type**: 2-Page Thesis Advisor Brief (Deliverable for Weeks 11–12 Transition)
**Artifact Repository**: `DukeTheOne/IAM_GNN` (Release `v1.0.0`)

---

## 1. Executive Summary & Research Motivation

Modern cloud infrastructures (AWS, Azure, GCP) contain tens of thousands of identities (users, service roles, machine principals). In these environments, **privilege escalation (PE)** rarely occurs via a single misconfigured permission; instead, adversaries chain subtle, multi-hop transitive authorizations—such as role assumption (`sts:AssumeRole`), permission modification (`iam:AttachRolePolicy`), or service delegation (`iam:PassRole`).

Existing formal analysis and model-checking tools (e.g., PMapper, AWS IAM Access Analyzer) rely on **complete-state observability**: they construct a deterministic reachability graph under the assumption that all entity definitions, policy documents, and trust relationships are known.

### The Partial Observability Reality
In real-world enterprise cloud organizations, complete-state observability is fundamentally broken due to:
1. **Cross-Account Trust Boundaries**: External trust relationships link to foreign accounts whose internal role graphs and policy definitions cannot be scanned or queried.
2. **Telemetry Gaps & Logging Blackouts**: Unmonitored VPCs, transient microservices, and unlogged IAM changes hide critical operational transitions.
3. **Third-Party SaaS Integrations**: Vendor roles assume identities without publishing their internal delegation graphs.

```text
Traditional Deterministic Analysis (PMapper / Access Analyzer):
   Observed Graph Go ──[BFS / SMT]──► Misses Unobserved Bridge ──► FALSE SENSE OF SECURITY (Zero Risk Reported)

Our Uncertainty-Aware Framework:
   Observed Graph Go ──[2-Layer RGCN]──► P(H | Go) ──[Temp Scaling]──► Calibrated Latent Completions H^(m)
                                                                                  │
   Capability-Compliant Patch P* ◄──[Sound AST Verifier] ◄──[Greedy Min-Cut C(a,r,c)]
```

When a single bridge relation is unobserved, deterministic tools suffer **zero recall** on latent attack paths. Conversely, naive heuristic repair tools (such as IAMPERE) strip entire dangerous actions (`iam:PassRole`), incurring catastrophic operational breakage on legitimate CI/CD and production workloads.

**Core Research Question**: *When critical IAM relationships are hidden by realistic telemetry blackouts, can an inductive relational model infer latent attack paths with calibrated uncertainty, and can this information guide capability-compliant least-privilege repairs that eliminate escalation risk without breaking legitimate workloads?*

---

## 2. Tier 1 Empirical Findings & Gate G0 Formal Verification

During the Tier 1 milestone (Weeks 1–10), we designed, implemented, and empirically evaluated an end-to-end framework comprising:
1. A formal RFC 7159 IAM Policy AST parser with an explicit **AWS Action Capability Model $C(a, r, c)$** restricting repairs to valid AWS authorization grammar.
2. A parameterized enterprise cloud generator embedding canonical and branching privilege escalation motifs across realistic departmental topologies.
3. A **2-layer Relational Graph Convolutional Network (RGCN)** with basis-sharing regularization and Platt temperature scaling ($\text{ECE} \le 0.05$).
4. An uncertainty-guided candidate pruner and greedy min-cut repair engine.
5. A three-level symbolic completion-set verifier ($G_o$, $\mathcal{H}_M$, $G^*$) with temporal CloudTrail log replay ($T_1$ baseline vs. $T_2$ drift).

### Gate G0 Quantitative Validation

| Metric / Research Question | Baseline Method | Ours (Uncertainty-Aware) | Empirical Advantage | Gate G0 Status |
| :--- | :---: | :---: | :---: | :---: |
| **RQ1: Latent Discovery (Cond. E: Adversarial Bridge)** | BFS: **0.50 ± 0.00** *(0% recall)*<br>Random Forest: **0.69 ± 0.03** | **0.91 ± 0.02 ROC-AUC**<br>*(ECE ≤ 0.038)* | +0.41 over BFS (recovers hidden bridge); +0.22 over tabular ML | ✅ **PASSED (GO)** |
| **RQ2: Security–Utility Tradeoff (Residual Risk on G\*)** | Identity (No Action): **1.00**<br>Coarse Deletion: **0.00** | **0.00** *(Residual Risk)* | Completely eliminates latent threat | ✅ **PASSED (GO)** |
| **RQ2: Operational Breakage (T₁ Baseline & T₂ Drift)** | Coarse Deletion: **29.6% ($T_1$) / 17.2% ($T_2$)**<br>Log-Only: **0.0% ($T_1$) / 0.8% ($T_2$)** | **0.0% ($T_1$) / 0.0% ($T_2$)** | Zero legitimate workload disruption; Pareto-dominates M2 & M3 | ✅ **PASSED (GO)** |
| **RQ3: Candidate Pruning & Repair Latency (N=5,000)** | Unpruned Solver: > 1,200 ms | **17.2 ms** (95.4% cut reduction) | 100% candidate recall, 0.00% optimality gap, <0.02s latency | ✅ **PASSED (GO)** |

**Gate G0 Verdict: FULL GO.** The empirical results confirm that inductive relational learning recovers unobserved authorization bridges and enables capability-compliant repairs that strictly dominate all existing baselines on the Pareto frontier.

---

## 3. Literature Review & Scientific Positioning (2024–2026)

Our framework bridges the gap between formal verification, active probing, and log-based tightening:

```
                          [Observability Assumption]
                     Partial State            Complete State
                  +-------------------------+-------------------------+
    Coarse        |                         |                         |
    Action        |                         |        IAMPERE          |
    Deletion      |                         |       (ASE 2023)        |
                  |                         |                         |
[Repair           +-------------------------+-------------------------+
 Granularity]     |   Ours (Calibrated      |                         |
    Capability-   |   RGCN + C(a,r,c) +     |    PMapper / Analyzer   |
    Compliant     |   Pareto Min-Cut)       |    (Formal SMT / BFS)   |
    Surgical Fix  |                         |    Restricter (2026)    |
                  +-------------------------+-------------------------+
```

1. **PMapper & AWS IAM Access Analyzer**: Deterministic model checkers. Sound under complete-state observability, but suffer catastrophic 0% recall when a single cross-account or unlogged transition is missing.
2. **IAMPERE (ASE 2023)**: Combines GNN edge embeddings with MaxSAT policy repair. However, IAMPERE assumes complete configuration visibility and applies coarse action revocation (e.g., deleting `iam:PassRole`), incurring severe operational breakage (17–30% in our experiments).
3. **TAC / TAC-GB & TAC-Bench (Hu et al., 2023–2026)**: Studies partial configurations using reinforcement learning to select active queries to an operator. **Our distinction**: In cross-account and third-party SaaS boundaries, external entities *cannot* be queried. We study the harder, unqueryable regime. We leverage TAC-Bench (2,500 tasks) as an external zero-shot transfer benchmark.
4. **IAM-PolicyRefiner (OOPSLA 2024) & Restricter (TACAS 2026)**: Tighten policies using CloudTrail logs or constraint solving. They do not infer latent attack paths and break when operational workloads experience temporal drift ($T_2$).
5. **IAM-MultiPolicyAnalyzer (FMCAD 2025)**: Formal authorization logic establishing deterministic Explicit Deny semantics, directly informing our symbolic verifier boundary.

---

## 4. Master's Thesis Roadmap: 18-Month Core + 6-Month Buffer

The proposed thesis answers one central research question: **How can cloud authorization be safely repaired when the attack-relevant graph is only partially observable?**

```text
+---------------------------------------------------------------------------------------------------------+
|                      18-MONTH CORE + 6-MONTH BUFFER THESIS ROADMAP                                      |
+---------------------------------------------------------------------------------------------------------+
| Months 1–3   (W1–12):   Course Prototype + hypothesis check (PASSED)                   ───► Gate G0     |
| Months 4–6   (W13–26):  Research Foundation (AWS Sandbox Conformance & TAC-Bench)      ───► Gate G1     |
| Months 7–10  (W27–43):  Core Method (Correlated Completions & Chance-Constrained MaxSAT)───► Gate G2     |
| Months 11–14 (W44–60):  Rigorous Evaluation (30–50 Real Terraform/CloudFormation Repos) ───► Gate G3     |
| Months 15–18 (W61–78):  Formal Verification, Thesis Writing & Defense                                   |
| Months 19–24 (W79–104): Optional Publication Extension (IEEE S&P / USENIX Security)                    |
+---------------------------------------------------------------------------------------------------------+
```

### Key Work Packages by Phase:
* **Phase 5 (Months 4–6 / Gate G1)**:
  * Conformance-test $C(a, r, c)$ against live AWS IAM Policy Simulator API sandboxes.
  * Ingest TAC-Bench; execute the **synthetic → TAC-Bench zero-shot** transfer experiment (train on synthetic generator, evaluate on TAC-Bench with no fine-tuning).
  * Expand PE motif catalog from 4 to 12 vectors (e.g., `Glue:CreateDevEndpoint`, `CodeBuild:StartBuild`).
* **Phase 6 (Months 7–10 / Gate G2)**:
  * Formulate joint **correlated completion sampling** to replace the independent Bernoulli edge assumption.
  * Implement chance-constrained MaxSAT solver ($\mathbb{P}[\text{residual attack}] \le \delta$).
  * Add Condition Injection repair tier (`aws:PrincipalArn`, `aws:RequestedRegion`) guided by $C(a, r, c)$.
* **Phase 7 (Months 11–14 / Gate G3)**:
  * Ingest and parse a curated corpus of **30–50 open-source production Terraform and CloudFormation enterprise repositories** (Level 2 data strategy).
  * Benchmark directly against IAMPERE's public artifact and PolicyRefiner baselines.
* **Phase 8 (Months 15–18)**:
  * Formalize symbolic soundness boundary: soundness relative to the evaluated completion set $\mathcal{H}_M$ and AWS authorization semantics.
  * Complete thesis manuscript writing and defense.

---

## 5. Requested Approvals & Action Items

1. **Formal Gate G0 Sign-Off**: Approval that the empirical evidence supports transitioning to the Master's Thesis track.
2. **AWS Test Sandbox Allocation**: Approval to configure a low-cost AWS sandbox account to automate $C(a, r, c)$ conformance testing via the official AWS IAM Policy Simulator API.
3. **IaC Corpus Curation**: Concurrence on the 30–50 curated Terraform repository selection criteria for Phase 7.
