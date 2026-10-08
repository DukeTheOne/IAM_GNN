# Project Proposal Plan

### (1) Title

**Uncertainty-Aware Least-Privilege Repair for Cloud IAM under Partial Observability**

### (2) Team Member

Currently just me and myself.

---

### (3) Abstract

Cloud Identity and Access Management (IAM) governs identity boundaries in modern multi-cloud architectures. However, developer team velocity and identity sprawl lead to complex permission misconfigurations, allowing adversaries to chain permissions across intermediate roles and escalate to administrative control. Many existing exact and repair-oriented approaches—deterministic graph reachability tools (e.g., PMapper), model-checking analyses, and hybrid GNN–MaxSAT repair frameworks (e.g., IAMPERE)—assume complete configuration visibility, whereas TAC specifically addresses partial visibility through adaptive querying of the cloud operator. In modern enterprise environments, however, some authorization evidence is structurally unavailable to any single auditor: it lies across cross-account boundaries, behind federated Identity Providers (IdPs), or in ephemeral STS session state. This work studies the complementary setting in which additional configuration information cannot be obtained, and the system must infer and repair under residual uncertainty.

This project proposes an **Uncertainty-Aware Least-Privilege Repair framework** for cloud access control under partial observability. Rather than treating link prediction and policy repair as disconnected modules, we unify them into a single research problem: *inferring attack-relevant latent authorization relationships with calibrated uncertainty, and synthesizing minimal, capability-compliant IAM policy repairs that eliminate escalation paths while preserving legitimate operational workflows despite topological uncertainty.*

Our architecture operates in four stages: (1) constructing a heterogeneous permission graph with a data-driven AWS Action–Resource–Condition capability model derived from AWS's Service Authorization Reference; (2) using a Relational Graph Convolutional Network (RGCN) to perform zero-query inference of hidden authorization relationships with calibrated probability estimates, from which plausible latent graph completions are sampled; (3) employing an uncertainty-aware repair engine that evaluates security risk across the sampled completions and applies only capability-valid transformations (resource scoping and action removal in the course prototype; condition injection in the thesis) while preserving historical workload evidence; and (4) verifying the patch against the observed graph and the completion set, and evaluating operational conformance on a temporal holdout of workload logs. The neural model may be wrong; the symbolic verifier defines the safety boundary, and its guarantees are stated explicitly relative to the formal model and the completion set. A Streamlit and PyVis triage dashboard is a deliverable for explainability and policy diffs, not a research contribution.

---

### (4) Introduction

#### 4.1 Motivation & Practical Relevance

In enterprise cloud platforms such as Amazon Web Services (AWS), Azure, and Google Cloud Platform (GCP), traditional network perimeter firewalls have been replaced by identity-centric authorization models. An enterprise cloud deployment routinely contains tens of thousands of interacting entities: human developers, federated SSO profiles, automated CI/CD pipelines, and compute resources like Lambda functions or EC2 instances. Each entity operates under declarative access control policies specifying granted permissions, target resources, and conditional context.

In real-world operations, permissions are granted iteratively to unblock immediate technical requirements, but privileges are rarely revoked when tasks conclude. This dynamic creates severe **identity sprawl**. Adversaries rarely attempt brute-force compromises against root accounts; instead, they compromise an edge machine identity (such as a forgotten test server or a junior contractor profile) and exploit subtle, multi-hop permission chains—such as passing roles to serverless execution environments or updating policy versions—to progressively elevate their privileges to full organization administrators.

#### 4.2 Problem Formulation & The Partial Observability Threat Model

Privilege Escalation (PE) in cloud IAM is fundamentally a relational graph reachability problem. However, solving it algorithmically in real-world enterprise environments faces three fundamental practical obstacles:

1. **Transitive Combinatorial Permissiveness:** Privilege escalation is rarely a single-step edge. It relies on multi-hop chains spanning identity groups, role-assumption paths, and permission-delegation primitives (e.g., passing roles to compute resources or reconfiguring policy versions).
2. **The Incomplete Telemetry Reality (Persistent Blind Spots):**
   A standard question in graph-based cloud security is: *"Why would edges ever be missing if cloud providers offer configuration APIs like AWS `GetAccountAuthorizationDetails`?"*
   In reality, `GetAccountAuthorizationDetails` is strictly single-account scoped and snapshots only static local policies. Global configuration visibility in modern enterprise multi-cloud architectures is fundamentally fragmented by three persistent industrial blind spots:
   * *Blind Spot 1: Cross-Account Organization Silos:* Enterprise cloud deployments partition workloads across dozens of isolated AWS accounts. An auditor inspecting Account A cannot evaluate whether an assumed role in Account B can pivot further into Account C if read access to Account B or C is restricted by departmental, organizational, or compliance boundaries.
   * *Blind Spot 2: Third-Party CI/CD & Federated IdP Boundaries:* Modern workloads authenticate via OpenID Connect (OIDC) and SAML federations (e.g., GitHub Actions, GitLab CI, Okta). The cloud provider only sees an incoming trust policy condition (e.g., matching a repository string); the internal user-to-group mappings and token issuance workflows inside the external SaaS provider remain completely opaque to cloud IAM scanners.
   * *Blind Spot 3: Ephemeral STS Session Delegations:* Dynamic credentials generated on the fly via AWS Security Token Service (`sts:AssumeRoleWithWebIdentity`, temporary session tokens) establish short-lived runtime trust relationships that leave no persistent configuration trail in static IAM snapshots.
3. **The Least-Privilege Remediation Dilemma (Operational Fragility vs. Over-Reaction):** Identifying an attack chain is insufficient; security engineers must remediate it without breaking production business services. When faced with missing telemetry, security teams face a dilemma: *do nothing* (leaving critical latent attack vectors unaddressed) or *coarsely revoke broad permissions* (e.g., deleting `iam:PassRole`), which severs legitimate deployment pipelines and causes operators to reject automated security patches.

#### 4.3 Existing Solutions and Research Gaps

Current approaches in cloud security and academic literature divide into three distinct paradigms, each exhibiting critical limitations when addressing partial observability:

* **1. Deterministic Graph Traversals & SMT Reachability Solvers (e.g., PMapper, Shevrin et al. [USENIX Security 2023]):**
  Tools like PMapper convert cloud configurations into graph structures and run deterministic search algorithms (BFS/Dijkstra) to identify reachability. Formal model-checking methods (e.g., Shevrin et al.) have demonstrated sound analysis on complex enterprise configurations. However, **deterministic methods require complete observability of the relevant state**. If an intermediate bridge edge across accounts or federated providers is unobserved in telemetry, deterministic graph search suffers total reachability failure, reporting high-value targets as unreachable despite an active multi-hop attack vector existing in reality.
* **2. Hybrid GNN–MaxSAT Solvers (e.g., IAMPERE [Hu et al., ASE 2023]):**
  IAMPERE pioneered using Graph Neural Networks (GNNs) on permission-flow graphs to predict intermediate repair candidates and prune search spaces for MaxSAT solvers. However:
  * *Assumes Full White-Box Configuration:* IAMPERE assumes complete visibility of the IAM topology and does not model latent link prediction or reasoning under missing telemetry.
  * *Coarse Action Revocation:* IAMPERE focuses predominantly on binary action removal. In enterprise production, revoking shared actions like `iam:PassRole` breaks legitimate workflows. It lacks multi-granularity repair that respects AWS action-level resource capabilities.
* **3. Partial-Configuration Detection via Adaptive Querying (TAC / TAC-GB [Hu et al.]):**
  TAC studies privilege-escalation detection under partial configurations (TAC-GB), combining GNN representations with reinforcement-learning-based query selection, and released TAC-Bench (2,500 tasks). TAC is both the closest work on partial observability and a **baseline** for this thesis, not a gap to be claimed.
  * *Different operating assumption and objective:* TAC focuses on adaptive information acquisition under partial observability (*"which missing information should be queried?"*). We study the complementary setting in which additional information is unavailable (e.g., another organization's account, an external IdP's internal mappings) and the system must reason under residual uncertainty. Our objective also differs: not detection alone, but a **repair whose security and operational utility are evaluated under that uncertainty**.
* **4. Log-Preserving Policy Tightening (e.g., IAM-PolicyRefiner [D'Antoni et al., OOPSLA 2024], Quantitative Policy Repair, Restricter [Wu et al., TACAS 2026, Cedar]):**
  Prior work reduces permissions while preserving permissions evidenced by access logs, and IAM-PolicyRefiner evaluates refined policies on left-out requests. Combining logs, formal synthesis, and fine-grained policy tightening is therefore **not claimed as novel**. These tools tighten *observed* configurations against *observed* workloads; they do not reason about latent, security-relevant relationships or about how topological uncertainty should shape the repair. The remaining gap is the combination of **latent security relationships + partial observability + repair**.

#### 4.4 Proposed Solution: Uncertainty-Aware Least-Privilege Repair

Rather than treating latent link prediction and policy repair as disconnected modules, this project unifies them into a coherent pipeline:

$$
G_o \xrightarrow{\text{Calibrated RGCN}} \mathbb{P}(H \mid G_o) \xrightarrow{\text{sample}} \{H_1,\dots,H_M\} \xrightarrow{\text{Uncertainty-Aware Repair}} \mathcal{P}^* \xrightarrow{\text{Verify on } G_o,\, G_o \cup H_i\text{; Temporal Replay}} \text{Repaired Policy + Stated Guarantee}
$$

The framework introduces three primary technical contributions:

* **Contribution 1: Zero-Query Latent Relation Inference with Calibrated Uncertainty:**
  Instead of requiring complete configuration graphs or interactive operator queries, we model the missing authorization evidence as a latent set $H$ and use an RGCN to estimate $\mathbb{P}((u, r, v) \in H \mid G_o)$ for hidden relations under structured blind spots (cross-account silos, federated IdP boundaries, ephemeral delegations). Calibration (Expected Calibration Error, reliability diagrams) is evaluated so that predicted probabilities carry meaning for the downstream repair. We keep two prediction problems explicit and separate: (i) *hidden-relation prediction* and (ii) *attacker–target reachability*, the latter derived from sampled completions rather than from a product of edge probabilities.
* **Contribution 2: Capability-Aware Repair under Uncertainty over Latent Completions:**
  Edge probabilities do not define path probabilities, because hidden relations can be correlated. We therefore sample latent completions $H\_1,\dots,H\_M \sim \mathbb{P}(H \mid G_o)$ and estimate residual risk by Monte Carlo:

$$
\min_{\mathcal{P}} \text{RepairCost}(\mathcal{P}) \quad \text{s.t.} \quad \frac{1}{M}\sum_{i=1}^{M} \mathbf{1}\Big[\text{Attack}(s,t;\,(G_o \cup H_i) \oplus \mathcal{P})\Big] \le \delta,\quad \text{LogConformance}(\mathcal{P}, \mathcal{D}^{T_1}) = 1,\quad \text{CapabilityValid}(\mathcal{P})
$$

  Every transformation must be valid under the AWS Action–Resource–Condition capability model $C(a, r, c)$ (Section 5.1).
* **Contribution 3: Completion-Set Verification and Temporal-Holdout Validation:**
  A symbolic verifier checks the patch against the observed graph *and* the latent completion set, so that "safe" is not merely "safe on $G_o$." Operational safety is evaluated on a **temporal holdout**: repairs are synthesized from workload history $T_1$ and conformance is measured on unseen later activity $T_2$. We claim conformance on the replayed workload under stated logging assumptions, not zero production breakage. The Streamlit + PyVis dashboard is a delivery vehicle for these results, not a contribution in itself.

#### 4.5 Formal Research Questions (RQs)

This research addresses three central, tightly scoped questions:

* **RQ1 (Inference under Structured Missingness):** Can a zero-query RGCN improve latent attack-path recall under structured missingness (random, cross-account, federated, ephemeral, and adversarial-bridge masking) while producing calibrated probability estimates? *Comparisons:* versus deterministic BFS / exact traversal on recall and false-negative rate; versus classical ML (Random Forest/XGBoost) and a rule-based motif-completion heuristic on PR-AUC and Recall@K; calibration (ECE) is assessed only for probabilistic models, since BFS yields 0/1 reachability.
* **RQ2 (Repair under Uncertainty):** When a critical attack-path relationship is hidden, does repair that accounts for latent-completion uncertainty achieve a better security–utility–repair-cost tradeoff than (1) no action on the observed graph, (2) coarse action deletion (IAMPERE-style), (3) log-only tightening (PolicyRefiner-style), and (4) point-estimate repair that thresholds the model's predictions?
* **RQ3 (Candidate Pruning Quality and Safety):** Can uncertainty-guided candidate pruning reduce symbolic repair cost while retaining the quality and safety of the resulting repairs? *Measured by:* runtime, candidate recall, repair optimality gap relative to the unpruned solver, and security-guarantee coverage. An interactive-latency target ($\lt 3\text{ s}$) is an engineering goal, not the hypothesis.

---

### (5) Method

```
+---------------------------------------------------------------------------------------------------------+
|                                    END-TO-END SYSTEM ARCHITECTURE                                       |
+---------------------------------------------------------------------------------------------------------+
| [Observed Telemetry] (AWS IAM JSON, Roles, Policies, Cross-Account Trusts, CloudTrail Logs)             |
|                                                    │                                                    |
|                                                    ▼                                                    |
| 1. Symbolic Parsing & Capability Model ──► Builds Heterogeneous Graph G_o + Action Capability C(a,r,c) |
|                                                    │                                                    |
|                                                    ▼                                                    |
| 2. Relational GNN (RGCN) + Calibration ──► Infers Latent Authorization Links P(H | G_o); ECE reported   |
|                                                    │                                                    |
|                                                    ▼                                                    |
| 3. Candidate Pruning & Uncertainty Repair ──► Synthesizes Capability-Aware Minimal Patches (Greedy / Z3)|
|                                                    │                                                    |
|                                                    ▼                                                    |
| 4. Completion-Set Verifier & Temporal Replay ──► Checks Safety on G_o and G_o ∪ H_i; Future Conformance  |
|                                                    │                                                    |
|                                                    ▼                                                    |
| 5. Streamlit / PyVis Triage Dashboard ──► Renders Ego-Networks, Confidence Weights & JSON Policy Diffs  |
+---------------------------------------------------------------------------------------------------------+
```

#### 5.1 Symbolic Permission Graph Construction & AWS Capability Model

We formalize cloud access control as a heterogeneous directed multigraph $G = (\mathcal{V}, \mathcal{E}, \tau_v, \tau_e)$:

* **Node Types $\mathcal{V}$:**
  * `User`: Human accounts and corporate credentials.
  * `Role`: Assumable IAM roles, service execution roles.
  * `Policy`: Declarative JSON policy documents containing statements, action permissions, and resource constraints.
  * `Resource`: High-value assets, including S3 buckets, Lambda functions, RDS instances, and KMS keys.
* **Typed Edge Relations $\mathcal{E}$:**
  * `MemberOf`: Associates users with identity groups.
  * `AssumesRole`: Models authentication and role assumption trust boundaries.
  * `AttachedWith`: Captures policy attachments to principals.
  * `ActsOn`: Maps permissions granted over concrete resources.
  * `PassesTo`: Represents execution boundary delegation (e.g., `iam:PassRole`).
* **AWS Action–Resource–Condition Capability Model $C(a, r, c)$:**
  Unlike naive graph models that assume any wildcard can be scoped to an ARN, repairs are checked against AWS authorization semantics. $C$ is a **data layer, not hand-coded rules**: it is generated from AWS's machine-readable Service Authorization Reference as a versioned snapshot (snapshot date recorded), restricted to the services used (IAM, STS, Lambda, EC2, S3, KMS) for the prototype and extended in the thesis.
  * Per action, $C$ records supported resource types (e.g., `iam:PassRole` targets a role resource; `iam:CreateAccessKey` targets a user resource, so it can be scoped to a user ARN), whether the action is wildcard-only (e.g., `iam:ListRoles`, `iam:GetAccountAuthorizationDetails`; verified against the snapshot), and the condition keys applicable to the action and resource.
  * A transformation (ARN scoping, condition injection, action removal) is admissible only if $C(a, r, c)$ permits it; inadmissible patches are rejected before reaching the solver output.
* **Symbolic Pre-Filtering:** Deterministically evaluates explicit `Deny` statements (which override `Allow` across all scopes) and eliminates syntactically invalid edges prior to neural message passing.

#### 5.2 Ground-Truth Data Diversity & Labeled Corpus

To ensure rigorous evaluation and avoid synthetic circularity:

* **Two-Level Problem Formulation:** The learning task is explicitly split into:
  1. *Latent Relation Prediction:* Predicting probability $\mathbb{P}((u, r, v) \in H \mid G_o)$ of unobserved authorization edges.
  2. *Attack Reachability under Latent Graphs:* Evaluating multi-hop reachability $\mathbb{P}(s \xrightarrow{*} t \mid G_o, H)$ over plausible completed topologies.
* **Environment-Level Graph Splits:** To prevent the model from simply memorizing synthetic generator artifacts, dataset splits are performed **strictly by environment/organization**, rather than randomly splitting edges within the same graph. The model is trained on Organization topologies $\mathcal{D}\_{\text{train}}$ and evaluated on completely unseen topologies $\mathcal{D}\_{\text{test}}$ with different departmental structures, branch factors, and identity distributions.
* **Tier 1 (Course Prototype - 10 Weeks):** Correctness over coverage. **P0 motifs (required):**
  1. `iam:PassRole` + `lambda:CreateFunction` / `lambda:InvokeFunction`
  2. `iam:CreateAccessKey` on a high-privilege user
  3. `iam:AttachRolePolicy` / `iam:AttachUserPolicy` (`AdministratorAccess`)
  4. `sts:AssumeRole` multi-hop chaining

  **P1 motifs (if on schedule):** `iam:PassRole` + `ec2:RunInstances` (instance profile elevation) and `iam:SetDefaultPolicyVersion`.
* **Tier 2 (M.S. Thesis - 18-Month Core + 6-Month Buffer):** Expands the motif catalog, evaluates **synthetic → TAC-Bench zero-shot** transfer (train on our generator, no fine-tuning, test on TAC-Bench), and ingests real Infrastructure-as-Code (Terraform, CloudFormation) from a curated set of open-source repositories. See the data strategy in Section 6.4.

#### 5.3 Relational Graph Convolutional Network (RGCN) Engine & Calibration

* **Node Feature Representation:** Nodes are initialized with multi-hot permission feature vectors, structural node-type one-hot encodings, and local graph centrality metrics (in/out-degree, PageRank).
* **Message Passing:** A 2-layer RGCN aggregates neighborhood representations independently across edge relations with basis-sharing regularization:

$$
h_i^{(l+1)} = \sigma \left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} \right), \quad W_r^{(l)} = \sum_{b=1}^{B} a_{r,b}^{(l)} V_b^{(l)}
$$

* **Bilinear Scoring Head:** Latent edge probability is decoded via:

$$
\hat{y}_{uv} = \sigma \left( h_u^T W_r h_v + b_r \right)
$$

* **Uncertainty Calibration:** Raw sigmoid outputs are calibrated using temperature scaling on a held-out validation set, minimizing Expected Calibration Error (ECE) to ensure that a predicted probability of 0.85 corresponds to an ~85% empirical presence of the latent link.
* **From Edge Probabilities to Graph and Path Uncertainty:** $\mathbb{P}(e\_1 \wedge e\_2) \neq \mathbb{P}(e\_1) \cdot \mathbb{P}(e\_2)$ in general, since hidden relations may share a cause (e.g., all relations inside one unobservable account). Path- and graph-level risk is therefore estimated over *sampled latent completions* $H_i \sim \mathbb{P}(H \mid G_o)$:

$$
\widehat{\mathbb{P}}\big[\text{attack path exists} \mid G_o\big] = \frac{1}{M}\sum_{i=1}^{M}\mathbf{1}\big[\text{Attack}(s,t;\,G_o \cup H_i)\big]
$$

  Sampling strategies, in increasing fidelity: (a) independent Bernoulli sampling on calibrated edge probabilities (a labeled approximation used in the prototype), (b) group-correlated sampling in which relations hidden by the same blind-spot mechanism are sampled jointly (thesis), and (c) a direct comparison against the naive product-of-edge-probabilities estimator to quantify the error that the independence assumption introduces.

#### 5.4 Uncertainty-Aware Symbolic Policy Remediation

* **Neural Candidate Space Pruning:** Calibrated model outputs and the sampled completions rank candidate bridge relations and cut points, reducing the repair search to a focused candidate subgraph (target $\sim 50$ edges). Pruning is treated as hypothesis generation: its quality is measured (candidate recall, optimality gap versus the unpruned solver) and it never bypasses the verifier.
* **Uncertainty-Aware Repair Formulation:**

$$
\min_{\mathcal{P}} \Big[\text{Cost}(\mathcal{P}) + \lambda \cdot \frac{1}{M}\sum_{i=1}^{M}\mathbf{1}\big[\text{Attack}(s,t;\,(G_o \cup H_i)\oplus\mathcal{P})\big]\Big]
$$

  subject to capability validity $\forall a \in \mathcal{P}: C(a, r, c)$ admits the transformation, and log conformance $\forall \text{event} \in \mathcal{D}^{T_1}: \text{Authorized}(\text{event}, G_o \oplus \mathcal{P})$. The thesis additionally studies the chance-constrained form (residual attack frequency $\le \delta$).
* **Repair Operations (in order of operational preference, subject to $C$):**
  1. *Resource ARN Scoping:* If $C$ shows the action supports resource-level permissions, narrow `Resource: "*"` to ARNs evidenced by the workload logs.
  2. *Action Subtraction:* Remove the offending action when no less disruptive admissible transformation exists.
  3. *Condition Injection (thesis phase):* Add contextual constraints (e.g., `aws:MultiFactorAuthPresent`, `iam:PassedToService`) only where $C$ lists the key as valid for the action and resource.
* **Verification and Claim Boundary:** The neural model can be wrong; the verifier defines the safety boundary. A patch is checked at three levels, and results are reported separately: (i) the observed graph $G_o$ only, (ii) $G_o \cup H_i$ for every sampled completion (the completion set), and (iii) the ground-truth graph, available only in synthetic evaluation, which exposes cases where the model missed a hidden relation. Guarantees are stated relative to the formal model and the verified completion set; hidden relations outside that set are quantified empirically as residual risk, not assumed away.
* **Temporal Replay:** Operational conformance is evaluated by replaying later workload activity $\mathcal{D}^{T_2}$ that was not used to synthesize the patch. We claim that the observed/replayed workload remains authorized under the stated logging assumptions, not that production breakage is impossible.

---

### (6) Experimental Design & Evaluation Plan

#### 6.1 Structured Masking Evaluation Protocol (RQ1)

Random edge deletion is a controlled baseline but not a model of how visibility is actually lost. We therefore mask **authorization evidence and delegation relationships**, not only literal static IAM edges, and document for each condition exactly which AWS object is hidden and how it is represented in the graph:

* **Condition A (Random Masking - Baseline):** Randomly drop $p \in \{10\text{\%}, 20\text{\%}, 30\text{\%}, 40\text{\%}\}$ of intermediate trust/delegation relations.
* **Condition B (Cross-Account Masking):** Hide all relations whose evidence lives in a designated unobservable account (its role trust policies and identity policies cannot be read).
* **Condition C (Federated IdP Masking):** The cloud-side OIDC/SAML trust policy remains visible; what is hidden is the external-principal-to-role delegation evidence, modeled as a distinct federated-delegation relation rather than as an `AssumesRole` edge.
* **Condition D (Ephemeral STS Masking):** Hide runtime, session-derived delegation (e.g., role-chaining sessions). This is runtime authorization state, modeled in the workload layer rather than as a persistent configuration edge.
* **Condition E (Adversarial Bridge Masking):** Hide the critical bottleneck relation along the target attack path.

**Baselines:** exact oracle (full graph, upper bound), deterministic BFS (PMapper proxy), Random Forest / XGBoost, and a rule-based motif-completion heuristic. Under Condition E, BFS fails by construction, so the heuristic and classical ML are the informative comparators for the RGCN.

**Metrics Reported:**
* Primary: PR-AUC, Recall@K, False Negative Rate, and attack-path recovery rate (positives are extremely sparse, so ROC-AUC is secondary).
* Calibration (probabilistic models only): ECE and reliability diagrams.
* Degradation curves with mean, variance, and 95% confidence intervals across 5 seeds, evaluated on held-out environments. No performance thresholds are fixed in advance; success is defined by the measured curves relative to baselines.

#### 6.2 The Central Tradeoff Experiment: Remediation under Uncertainty (RQ2)

To demonstrate the scientific necessity of combining neural inference with symbolic repair, we evaluate on attack paths where a critical bridge edge $a \to b$ is hidden:

| Remediation Strategy | Description | Security (Residual Reachability in Ground Truth) | Operational Impact (Unseen T₂ Requests Broken) |
| :--- | :--- | :--- | :--- |
| **Method 0: Oracle Repair (reference)** | Repair with full knowledge of the hidden relation | Lowest achievable | Reference |
| **Method 1: No Action (Blind Traversal)** | Treats the unobserved relation as non-existent | Attack path remains exploitable | None (nothing modified) |
| **Method 2: Coarse Action Deletion (IAMPERE-style)** | Revokes broad actions (e.g., `iam:PassRole`) | Low | Expected high (breaks shared pipelines) |
| **Method 3: Log-Only Tightening (PolicyRefiner-style)** | Tightens to observed history; blind to the latent path | Latent path remains | Minimal on past logs |
| **Method 4a: Point-Estimate Repair** | Thresholds model output (e.g., p̂ ≥ 0.5), repairs that single graph | Depends on model recall | Low |
| **Method 4b: Uncertainty-Aware Repair (Ours)** | Optimizes over sampled completions with capability-valid, log-conforming transformations | Lower residual risk than 4a if uncertainty matters | Low |

Methods 3 and 2 are simplified re-implementations of the *strategies* of PolicyRefiner and IAMPERE in the prototype; the thesis compares against their public artifacts where available. The 4a vs 4b comparison isolates whether modeling uncertainty, rather than merely predicting edges, improves repair.

**Metrics Reported (kept separate, not collapsed into one score):**
* Weighted reachable high-value asset count, maximum reachable privilege, attack-path count, and minimum-cut cost.
* Policy semantic change and repair minimality (statements modified, actions removed).
* Observed legitimate requests broken on the temporal holdout $T_2$.
* Security–utility–cost Pareto curves.

#### 6.3 Candidate Pruning Quality, Safety, and Latency (RQ3)

* Compare unpruned repair against uncertainty-guided pruned repair on graphs of 500 to 5,000 nodes.
* Report **candidate recall** (does the pruned set contain the repair edges the unpruned solver uses?), **optimality gap**, **security-guarantee coverage** (fraction of verified-safe patches that remain safe in the ground truth), and wall-clock runtime.
* An interactive-latency target ($\lt 3\text{ s}$) is tracked as an engineering goal only; speed is meaningless if pruning discards the correct repair.

#### 6.4 Data Strategy and Claim Boundaries

* **Configuration data:** Level 1 synthetic generator with environment-level splits; Level 2 real IaC-derived configurations (thesis); TAC-Bench for a **synthetic → TAC-Bench zero-shot** transfer test (train on our generator, no fine-tuning). IAMPERE's public implementation and dataset serve as comparison material.
* **Workload / CloudTrail data is hierarchical so that the thesis does not depend on obtaining real logs:**
  * *Level 1:* synthetic IAM configuration with a synthetic temporal workload (course prototype).
  * *Level 2:* real IaC-derived configuration with synthetically generated workload traces.
  * *Level 3:* real IaC/IAM with real CloudTrail, pursued only if a lab or industry partner can provide it.
* **Claim boundaries:** conformance is claimed on replayed workload under stated logging assumptions; safety is claimed relative to the formal model and the verified completion set; generalization is claimed only to the extent shown by held-out environments and TAC-Bench transfer.
* **Venue philosophy:** the target hierarchy is a good thesis, then a publishable paper, then a strong venue if results warrant it. A top security venue is an aspiration, not a design requirement.

#### 6.5 Visual Interface & Operational Deliverable

The deliverable includes an interactive Streamlit + PyVis triage dashboard rendering ego-network attack paths ($k \le 2$ hops), uncertainty scores, side-by-side JSON policy diffs, and simulated impact metrics. It is a delivery vehicle, scoped as lowest priority within the 10-week prototype (static graph plus JSON diff is an acceptable fallback).

---

### (7) References

1. **Hu, Y., Wang, W., Khurshid, S., McMillan, K. L., & Tiwari, M.** "Fixing Privilege Escalations in Cloud Access Control with MaxSAT and Graph Neural Networks." *Proceedings of the 38th IEEE/ACM International Conference on Automated Software Engineering (ASE)*, pp. 104–115, 2023.
2. **Hu, Y., et al.** "TAC: Hybrid IAM Privilege Escalation Detection (TAC-GB & TAC-Bench)." *arXiv:2304.14540*. Cite the latest revision at time of writing and state the revision history accurately.
3. **Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M.** "Modeling Relational Data with Graph Convolutional Networks." *European Semantic Web Conference (ESWC)*, pp. 593–607, Springer, 2018.
4. **Shevrin, I., & Margalit, O.** "Detecting Multi-Step IAM Attacks in AWS Environments via Model Checking." *USENIX Security Symposium*, 2023.
5. **D'Antoni, L., Ding, S., Goel, A., Ramesh, M., Rungta, N., & Sung, C.** "Automatically Reducing Privilege for Access Control Policies (IAM-PolicyRefiner)." *Proceedings of the ACM on Programming Languages (OOPSLA)*, 2024.
6. **Barnett, L. A., D'Antoni, L., Goel, A., Kıcı, R. G., Rungta, N., Southern, M., & Sung, C.** "Modeling the AWS Authorization Engine (IAM-MultiPolicyAnalyzer)." *Formal Methods in Computer-Aided Design (FMCAD)*, 2025.
7. **Wu, K. L., Jenkins, C., Stoller, S. D., & Chowdhury, O.** "Automatically Tightening Access Control Policies with Restricter." *Tools and Algorithms for the Construction and Analysis of Systems (TACAS)*, 2026. (Targets Amazon Cedar rather than AWS IAM.)
8. **Amazon Web Services.** "Service Authorization Reference: Actions, Resources, and Condition Keys for AWS Services." *AWS Documentation* (record snapshot date used).
9. **Rhino Security Labs.** "AWS IAM Privilege Escalation Methods: A Systematic Taxonomy." *Open Cloud Security Exploit Corpus*, 2021.
10. **NCC Group.** "Principal Mapper (PMapper): An Open-Source Graph Analysis Tool for IAM Reachability in Cloud Environments." *Software Tool Repository*, 2021.
