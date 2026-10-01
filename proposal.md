# Project Proposal Plan

### (1) Title

**Hybrid Symbolic–Relational Graph Learning for Latent Attack-Path Discovery and Least-Privilege Optimization in Cloud IAM**

### (2) Team Member

Currently just me and myself.

---

### (3) Abstract

Cloud Identity and Access Management (IAM) governs identity boundaries in modern multi-cloud architectures. However, developer team velocity and identity sprawl lead to complex permission misconfigurations, allowing adversaries to chain permissions across intermediate roles and escalate to administrative control. Existing defense paradigms suffer from a fundamental dichotomy: deterministic formal methods and graph traversal tools (such as Breadth-First Search or SMT/SAT solvers) require fully observed topology and suffer from state-space explosion, breaking down when configurations are incomplete or noisy; conversely, pure deep learning approaches lack rigid domain semantics and frequently produce syntactically unviable attack paths.

To bridge this gap, this project proposes a **Hybrid Symbolic–Relational Graph Learning framework** for cloud access control. We model enterprise infrastructure as a heterogeneous permission flow graph where declarative IAM syntax and policy rules provide symbolic priors to constrain the learning space. A Relational Graph Convolutional Network (RGCN) is then deployed to learn entity embeddings and perform link prediction, discovering latent, unmanifested privilege escalation paths even under masked telemetry. Finally, a symbolic optimization layer refines these probabilistic scores to compute formally valid, approximately minimal policy patches that sever attack chains. The final deliverable will feature an interactive auditing and visual triage dashboard, providing explainable multi-hop risk discovery and actionable least-privilege remediation.

---

### (4) Introduction

#### 4.1 Motivation & Practical Relevance

In enterprise cloud platforms such as Amazon Web Services (AWS), Azure, and Google Cloud Platform (GCP), traditional network perimeter firewalls have been replaced by identity-centric authorization models. An enterprise cloud deployment routinely contains tens of thousands of interacting entities: human developers, federated SSO profiles, automated CI/CD pipelines, and compute resources like Lambda functions or EC2 instances. Each entity operates under declarative access control policies specifying granted permissions, target resources, and conditional context.

In real-world operations, permissions are granted iteratively to unblock immediate technical requirements, but privileges are rarely revoked when tasks conclude. This dynamic creates severe **identity sprawl**. Adversaries rarely attempt brute-force compromises against root accounts; instead, they compromise an edge machine identity (such as a forgotten test server or a junior contractor profile) and exploit subtle, multi-hop permission chains—such as passing roles to serverless execution environments or updating policy versions—to progressively elevate their privileges to full organization administrators.

#### 4.2 Problem Formulation

#### 4.2 Problem Formulation & The Incomplete Telemetry Threat Model

Privilege Escalation (PE) in cloud IAM is fundamentally a relational graph reachability problem. However, solving it algorithmically in real-world enterprise environments faces three practical obstacles:

1. **Transitive Combinatorial Permissiveness:** Privilege escalation is rarely a single-step edge. It relies on multi-hop chains spanning identity groups, role-assumption paths, and permission-delegation primitives (e.g., passing roles to compute resources or reconfiguring policy versions).
2. **The Incomplete Telemetry Reality (Industrial Blind Spots):**
   A standard critique of graph-based cloud security is: *"Why would edges ever be missing if cloud providers offer configuration APIs like AWS `GetAccountAuthorizationDetails`?"* 
   In reality, `GetAccountAuthorizationDetails` is strictly single-account scoped and snapshots only static local policies. Global configuration visibility in modern multi-cloud architectures is fundamentally fragmented by three ubiquitous industrial blind spots:
   * *Blind Spot 1: Cross-Account AWS Organization Silos:* Enterprise cloud deployments routinely partition workloads across dozens of isolated AWS accounts. An auditor inspecting Account A cannot evaluate whether an assumed role in Account B can pivot further into Account C if read access to Account B or C is restricted by departmental or compliance boundaries.
   * *Blind Spot 2: Third-Party CI/CD & Federated IdP Boundaries:* Modern workloads authenticate via OpenID Connect (OIDC) and SAML federations (e.g., GitHub Actions, GitLab CI, Okta, Azure AD). The cloud provider only sees a trust policy condition (e.g., matching a repository string); the internal user-to-group mappings and token issuance workflows inside the external SaaS provider remain completely opaque to the cloud IAM scanner.
   * *Blind Spot 3: Ephemeral STS Session Delegations:* Dynamic credentials generated on the fly via AWS Security Token Service (`sts:AssumeRoleWithWebIdentity`, temporary session tokens) establish short-lived runtime trust relationships that leave no persistent configuration trail in static IAM snapshots.
3. **The Least-Privilege Remediation Tradeoff (Operational Fragility):** Identifying an attack chain is insufficient; security engineers must remediate it without breaking production business services. Coarsely revoking core permissions (such as `iam:PassRole`) inevitably severs active deployment pipelines, causing engineers to reject automated security patches.

#### 4.3 Existing Solutions and Their Gaps

Current approaches in cloud security and academic literature divide into three paradigms, all exhibiting key operational and scientific limitations:

* **1. Deterministic Graph Traversals & Reachability Solvers (e.g., PMapper, BloodHound, Pure SMT/SAT):**
  Tools like PMapper convert cloud configurations into graph databases and run deterministic search algorithms (BFS/Dijkstra) to identify reachability. While mathematically sound on clean, fully observed graphs, **they possess zero tolerance for incomplete data**. If an intermediate bridge edge across accounts or federated providers is unobserved in telemetry, deterministic graph search fails completely, reporting high-value targets as unreachable when an active multi-hop attack vector exists. Furthermore, pure MaxSAT approaches suffer from severe combinatorial state explosion when evaluating complex enterprise graphs with thousands of interacting policies.
* **2. First-Generation Hybrid GNN–MaxSAT Solvers (e.g., IAMPERE [Hu et al., ASE 2023], TAC [Hu et al., 2023]):**
  Recent pioneering work such as IAMPERE (ASE '23) demonstrated that GNNs can generate intermediate patches to prune the search space for MaxSAT solvers in AWS IAM privilege escalation repair. Similarly, TAC introduced active querying for partial configurations. However, existing hybrid solvers still exhibit two critical limitations:
  * **Assumption of Global Observability:** IAMPERE assumes a static, fully observed white-box graph. It does not perform *latent link prediction* to infer unmanifested escalation routes across fragmented, cross-account boundaries where edges are unobserved.
  * **Coarse Permission Revocation (Production Fragility):** Prior tools focus primarily on binary action removal. In enterprise production, revoking a shared action like `iam:PassRole` often severs legitimate automated deployment pipelines. Existing work lacks *multi-granularity repair*—such as scoping wildcard `Resource: "*"` down to target ARNs or injecting Attribute-Based Access Control (ABAC) conditions—which preserves operational utility while neutralizing escalation paths.
* **3. Pure Deep Learning & Heuristic Classification:**
  Standard neural models attempt to score entities using continuous embeddings without formal logical grounding. They frequently hallucinate or suggest attack paths that violate hard IAM logical constraints (e.g., proposing an escalation step across an explicit `Deny` policy or attempting actions prohibited by service boundaries).

#### 4.4 Proposed Hybrid Solution & Novel Breakthroughs

To advance beyond the state of the art, this project designs a **Hybrid Symbolic–Relational Graph Learning Architecture** with three core differentiators:

* **Breakthrough 1: Latent Multi-Hop Attack Path Discovery under Fragmented Telemetry:**
  Instead of requiring fully observed graphs, we formulate privilege escalation detection as **latent multi-relational link prediction** using a Relational Graph Convolutional Network (RGCN). By learning structural entity embeddings and topological neighborhood motifs across heterogeneous cloud relations, the model infers missing or obscured trust edges across the three industrial blind spots (cross-account silos, federated IdP boundaries, and ephemeral delegations) where deterministic tools suffer total reachability failure.
* **Breakthrough 2: Multi-Granularity Policy Remediation with Operational Log Conformance:**
  Rather than simply revoking coarse action permissions, our neural-guided symbolic optimization layer evaluates three repair tiers:
  1. *Action Subtraction:* Revoking high-risk standalone actions where safe.
  2. *Resource ARN Scoping:* Narrowing permissive wildcards (`Resource: "*"`) down to specific allowed resource ARNs.
  3. *ABAC Condition Injection:* Appending contextual constraints (e.g., MFA requirements, source IP restrictions, or `aws:PrincipalTag` validation).
  We formalize a **dual optimization objective**: severing all latent attack reachability while maximizing historical operational utility conformance (measured via CloudTrail API log replay, proving that 100% of historically executed benign actions remain authorized).
* **Breakthrough 3: Explainable Visual Auditing & Operational Triage:**
  We bridge academic formal methods with day-to-day cloud security operations through an interactive visual triage dashboard (Streamlit + PyVis), rendering high-risk latent paths, confidence weights, and side-by-side JSON policy diffs for rapid human-in-the-loop remediation.

#### 4.5 Formal Research Questions (RQs)

This research addresses three central questions:

* **RQ1 (Incomplete Telemetry Robustness):** How effectively can an RGCN perform latent link prediction to recover multi-hop privilege escalation reachability under realistic industrial telemetry blind spots (cross-account silos, federated IdP boundaries, 10% to 40% masked edges), compared to deterministic BFS traversals and tabular machine learning baselines?
* **RQ2 (Multi-Granularity Repair & Operational Conformance):** Does a multi-tier symbolic repair formulation (Action Subtraction vs. Resource Scoping vs. Condition Injection) preserve 100% of historically executed benign CloudTrail actions while achieving equivalent blast-radius reduction compared to coarse action-deletion baselines like IAMPERE?
* **RQ3 (Computational Scalability & Real-Time Triage):** Can GNN-driven candidate search space pruning constrain MaxSAT / min-cut solving times to interactive latencies ($< 3.0$ seconds) on enterprise graphs spanning 500 to 5,000 heterogeneous nodes?

---

### (5) Method

#### 5.1 Symbolic Permission Flow Graph Construction

We formalize cloud access control as a heterogeneous directed graph where access rules and entities are explicitly structured:

* **Node Types:**
  * `User`: Human accounts and corporate credentials.
  * `Role`: Assumable IAM roles, service execution roles.
  * `Policy`: Declarative JSON policy documents containing statements, action permissions, and resource constraints.
  * `Resource`: High-value assets, including S3 data buckets, Lambda compute functions, RDS database instances, and KMS encryption keys.

* **Typed Edge Relations:**
  * `MemberOf`: Associates users with identity groups.
  * `AssumesRole`: Models authentication and role assumption trust boundaries.
  * `AttachedWith`: Captures policy attachments to principals.
  * `ActsOn`: Maps permissions granted over concrete resources.
  * `PassesTo`: Represents execution boundary delegation (e.g., `iam:PassRole`).

* **Symbolic Pre-Filtering:** Before feeding the graph into the neural model, a deterministic symbolic parsing layer analyzes explicit `Deny` statements, condition blocks, and resource ARNs, eliminating syntactically invalid edges and initializing relational adjacency masks.

#### 5.2 Ground-Truth Data Diversity & Labeled Corpus

To satisfy both rapid course prototyping and peer-reviewed publication rigor, the data corpus follows a two-tier evaluation strategy:

* **Tier 1 (Course MVP Synthetic Generator):** Because raw corporate cloud graphs contain proprietary security configurations, we develop a parameterized synthetic graph generator. We seed the topology using the recognized catalog of 28+ IAM privilege escalation vectors documented by Rhino Security Labs (e.g., `iam:CreateAccessKey`, `iam:SetDefaultPolicyVersion`, and `iam:PassRole` combined with compute execution). The generator synthesizes environments ranging from 500 to 5,000 nodes, combining benign least-privilege departmental clusters (DevOps, QA, Analytics) with positive escalation chain motifs.
* **Tier 2 (48-Week Thesis Ecological Validity & Real-World IaC Ingestion):** To defeat the classic reviewer objection regarding synthetic circularity (*"training a model on synthetically injected motifs simply tests synthetic memorization"*), the full thesis incorporates real-world Infrastructure-as-Code (IaC) configurations:
  1. *Open-Source IaC Repositories:* Mining and parsing production-grade Terraform, CloudFormation, and AWS CDK templates from high-star GitHub repositories to construct realistic multi-principal trust graphs.
  2. *Enterprise Benchmarks:* Ingesting TAC-Bench (2,500 real-world inspired IAM misconfiguration benchmarks) to evaluate real-world generalization across heterogeneous cloud architectures.

#### 5.3 Relational Graph Convolutional Network (RGCN) Engine

To model multi-relational interactions across distinct entity and edge types:

* **Node Initialization:** Nodes are initialized with multi-hot permission feature vectors, structural node-type encodings, and local graph centrality metrics.
* **Message Passing:** We implement a 2-layer Relational Graph Convolutional Network (RGCN) using PyTorch Geometric. The layer-wise message passing aggregates neighborhood information independently across each edge type, using basis-sharing regularization to maintain parameter efficiency across diverse cloud relations.
* **Link-Prediction Head:** The resulting latent node representations are passed through a bilinear scoring decoder to predict the probability of an emergent escalation link between arbitrary low-privilege principals and administrative roles. Training is guided by Binary Cross-Entropy loss with balanced negative edge sampling.

#### 5.4 Symbolic Optimization for Multi-Granularity Policy Remediation

Once the GNN identifies high-risk latent attack paths, the pipeline transitions from the neural representation back to symbolic optimization:

* **Constrained Candidate Search:** The GNN's predicted edge probabilities and attention weights serve as priority scores to prune the full graph down to a compact candidate subgraph ($< 50$ edges), mitigating MaxSAT combinatorial state explosion.
* **Multi-Granularity MaxSAT Formulation:** Unlike prior tools that strictly delete entire permission statements, our symbolic solver (implemented in Z3) optimizes across three distinct repair tiers:
  1. *Action Subtraction:* Revoking high-risk standalone actions where safe (e.g., removing `iam:CreateAccessKey`).
  2. *Resource ARN Scoping:* Restricting wildcard `Resource: "*"` statements to designated least-privilege resource ARNs (e.g., binding `iam:PassRole` strictly to trusted service worker roles).
  3. *ABAC Condition Injection:* Injecting contextual constraints (e.g., enforcing `aws:MultiFactorAuthPresent: true` or IP restrictions).
* **Dual Optimization Objective & CloudTrail Conformance:** The solver optimizes a formal dual objective:
  $$\min_{\mathcal{P}_{patch}} \text{BlastRadius}(G \oplus \mathcal{P}_{patch}) \quad \text{subject to} \quad \text{LogConformance}(\mathcal{P}_{patch}, \mathcal{D}_{\text{CloudTrail}}) = 1.0$$
  By replaying historical AWS CloudTrail API event logs, the optimizer guarantees that **100% of historically executed benign actions remain authorized**, completely eliminating production breakage.
* **Actionable Policy Diff:** The system automatically outputs a human-readable JSON remediation snippet with strict syntax validation, enabling one-click deployment or PR generation in infrastructure-as-code (Terraform/CloudFormation) workflows.

---

### (6) Expected Results

#### 6.1 Quantitative Performance Expectations

We will evaluate our hybrid framework against two baselines: a deterministic BFS reachability engine (representative of tools like PMapper) and a classical machine learning model (Random Forest trained purely on graph centrality statistics).

* **Latent Link Prediction:** On a held-out test split of complex privilege escalation paths, our relational model is expected to achieve strong predictive performance (**ROC-AUC $\ge$ 0.90** and **Precision-Recall AUC $\ge$ 0.85**).
* **Incomplete Graph Robustness (Masked Telemetry Experiment):** In experiments where 20% to 40% of intermediate trust edges are randomly masked from the input graph, deterministic BFS detection accuracy will collapse toward zero because single missing hops break path traversals. In contrast, our hybrid framework is expected to maintain robust detection (AUC $\ge$ 0.80) by inferring latent trust through structural neighborhood embeddings.
* **Remediation Minimality:** The symbolic repair layer is expected to generate policy patches that reduce the targeted identity blast radius by over 80% while removing fewer than 15% of the total assigned permissions, demonstrating strict least-privilege efficiency.

#### 6.2 Visual Interface & Remediation Deliverable

The anticipated final artifact will include an interactive Streamlit and PyVis web dashboard demonstrating end-to-end operational triage:

```
+-----------------------------------------------------------------------------------------+
|                HYBRID SYMBOLIC-GNN CLOUD IAM AUDIT DASHBOARD                            |
+-----------------------------------------------------------------------------------------+
| [System Telemetry Metrics]                                                              |
| Scanned Identities: 1,248 | Evaluated Policies: 3,410 | High-Risk Latent Paths: 12      |
|                                                                                         |
| [Identities Risk Ranking Table]                                                         |
| Identity               Assigned Role       Blast Radius    Predicted Escalation Target  |
| usr-test-contractor    Contractor-Dev      0.96 [CRITICAL] role-org-admin               |
| srv-lambda-pipeline    CI-Builder          0.88 [HIGH]     role-data-admin              |
| usr-data-analyst       Read-Only-Analyst   0.09 [BENIGN]   None                         |
|                                                                                         |
| [Interactive Attack-Path Visualizer (PyVis Graph Canvas)]                               |
|                                                                                         |
|       (usr-test-contractor)                                                             |
|                │                                                                        |
|                ▼ [AttachedPolicy: LambdaWorker] (Weight: 0.92)                          |
|       (role-sandbox-executor)                                                           |
|                │                                                                        |
|                ▼ [PassesTo: iam:PassRole] (Weight: 0.97) <--- Flagged Critical Link     |
|       [role-org-admin] <--- High-Value Target Breach Identified                         |
|                                                                                         |
| [Symbolic Least-Privilege Remediation Recommendation]                                   |
| Status: Validated by Symbolic Verifier                                                  |
| Action: Strip 'iam:PassRole' action from Policy 'LambdaWorker' on resource '*'          |
| Target Patch Diff:                                                                      |
| - "Action": ["iam:PassRole", "lambda:CreateFunction"]                                   |
| + "Action": ["lambda:CreateFunction"]                                                   |
| Expected Blast Radius Reduction: 92.4% (No dependent developer pipelines broken)        |
+-----------------------------------------------------------------------------------------+
```

---

### (7) Reference

1. **Hu, Y., Wang, W., Khurshid, S., McMillan, K. L., & Tiwari, M.** "Fixing Privilege Escalations in Cloud Access Control with MaxSAT and Graph Neural Networks." *Proceedings of the 38th IEEE/ACM International Conference on Automated Software Engineering (ASE)*, pp. 104–115, 2023.
2. **Hu, Y., et al.** "TAC: Hybrid IAM Privilege Escalation Detection." *arXiv preprint arXiv:2304.14540*, 2023.
3. **Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M.** "Modeling Relational Data with Graph Convolutional Networks." *European Semantic Web Conference (ESWC)*, pp. 593–607, Springer, 2018.
4. **Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J.** "GNNExplainer: Generating Explanations for Graph Neural Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 2019.
5. **Rhino Security Labs.** "AWS IAM Privilege Escalation Methods: A Systematic Taxonomy." *Technical Research and Open Exploit Corpus*, 2021.
6. **NCC Group.** "Principal Mapper (PMapper): An Open-Source Graph Analysis Tool for IAM Reachability in Cloud Environments." *Software Tool Repository*, 2021.
