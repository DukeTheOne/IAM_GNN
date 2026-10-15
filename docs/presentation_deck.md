# Course Presentation: Uncertainty-Aware Least-Privilege Repair for Cloud IAM under Partial Observability

**Course Milestone Presentation (Tier 1 Week 10 Deliverable)**
**Author**: Duke
**Date**: October 2026
**Artifact Repository**: `DukeTheOne/IAM_GNN` (Release `v1.0.0`)

---

## Slide 1: The Cloud IAM Identity Crisis

### Slide Content
- **The Modern Cloud Paradigm**:
  - Enterprise cloud architectures host tens of thousands of identities (IAM Users, Roles, Service Accounts, Lambdas).
  - Multi-hop privilege escalation occurs through non-obvious relational paths: role assumption chains (`sts:AssumeRole`), service delegation (`iam:PassRole`), and policy attachment (`iam:AttachRolePolicy`).
- **The Core Vulnerability**:
  - Attackers rarely need global administrator privileges initially; they chain low-privilege identities across trust boundaries.
- **The Reality of Partial Observability**:
  - Telemetry is severely fragmented: cross-account trust boundaries hide external role definitions.
  - Third-party SaaS integrations, uninstrumented legacy VPCs, and transient microservices create critical blind spots.
  - Traditional security scanners assume complete state observability—an assumption that fails catastrophically in multi-account organizations.

### Visual Layout
- Split screen:
  - Left: Complex enterprise cloud multigraph with shadowed cross-account boundaries and missing link icons.
  - Right: Key statistic callouts: "Average enterprise has >10,000 permissions across 50+ AWS accounts; >80% are unused, while 1 missing trust link permits full account takeover."

### Speaker Notes
> "Welcome everyone. In modern cloud infrastructures, identity is the new security perimeter. Yet cloud IAM has reached an identity crisis. Today's organizations operate dozens of AWS accounts connected through complex cross-account trust policies. Because telemetry is fragmented across organizations and third-party integrations, security teams operate under partial observability. Scanners like PMapper assume you have 100% of the graph. When a single trust link is unobserved, traditional tools report zero risk—giving operators a false sense of security while an attacker escalates to root. This project solves that fundamental problem."

---

## Slide 2: Prior Art & The Novelty Gap

### Slide Content
| Prior System | Technique | Key Strength | Fatal Vulnerability / Gap |
| :--- | :--- | :--- | :--- |
| **PMapper / AWS Access Analyzer** | Graph BFS / SMT Solvers | Sound under complete state $G$ | **Zero Recall under Partial Observability**: If a single bridge relation is unobserved, reachability drops to zero. |
| **IAMPERE (ASE 2023)** | GNN-Assisted MaxSAT | Automated policy repair | **Coarse Deletion**: Removes dangerous actions entirely (`iam:PassRole`), breaking legitimate CI/CD workloads. Assumes $G$ is complete. |
| **TAC / TAC-GB (ICSE 2024)** | Adaptive Querying | Discovers hidden relations via queries | **Query Dependency**: Requires an oracle or interactive API to query missing data; inapplicable when telemetry is permanently inaccessible. |
| **PolicyRefiner / Restricter** | CloudTrail Log Tightening | Tightens observed policy ASTs | **Operational Drift Breakage**: Strips valid future permissions; zero reasoning over latent structural relations. |
| **This Work (Ours)** | **Calibrated Relational Learning + Uncertainty-Aware Repair** | **Infers latent attack paths with calibrated uncertainty and synthesizes capability-compliant minimal repairs** | **First end-to-end framework solving repair under missing telemetry without breaking legitimate workloads.** |

### Visual Layout
- 2x2 Positioning Matrix:
  - X-axis: Observability Assumption (Complete State $\leftrightarrow$ Partial / Missing Telemetry)
  - Y-axis: Remediation Granularity (Coarse Action Stripping $\leftrightarrow$ Capability-Compliant Minimal Repair)
  - Our work occupies the upper-right frontier.

### Speaker Notes
> "Let's situate our work against the state of the art. PMapper and formal SMT solvers are sound, but they require a complete configuration graph. If cross-account telemetry is missing, their reachability analysis fails completely. IAMPERE used GNNs for MaxSAT repair, but it strips entire actions like `iam:PassRole`, breaking mission-critical workloads, and it still assumes complete observability. TAC uses active probing, but what if the external account belongs to a third party and you cannot query it? Finally, log-tightening tools like PolicyRefiner break whenever operational drift occurs. We bridge this gap by predicting missing relations with calibrated probabilities and repairing policies with minimal AST edits."

---

## Slide 3: Core Research Thesis & Contribution

### Slide Content
> **Central Research Thesis**:
> *Inferring latent attack relationships with calibrated uncertainty enables synthesizing capability-compliant least-privilege IAM policy repairs that eliminate privilege escalation risk under partial observability while strictly preserving legitimate enterprise workloads.*

- **Key Research Contributions**:
  1. **Relational IAM Multigraph Formulation**: Heterogeneous representation mapping AWS entities, permissions, and trust relationships with an explicit Action Capability Model $C(a, r, c)$.
  2. **Calibrated Relational Inductive Learning**: 2-layer Relational Graph Convolutional Network (RGCN) with temperature scaling, achieving Expected Calibration Error (ECE) $< 0.05$.
  3. **Capability-Compliant Greedy Repair Engine**: Subgraph min-cut synthesis guided by AWS resource scoping rules, eliminating reachability without coarse action deletions.
  4. **Multi-Completion Safety Verification**: Three-level verification ($G_o$, $\mathcal{H}_M$, $G^*$) guaranteeing residual risk bounds before policy deployment.
  5. **Interactive Security Triage System**: High-performance Streamlit + PyVis operator dashboard for sub-second visual triage and automated policy diff inspection.

### Speaker Notes
> "Our core thesis is that we do not have to choose between complete observability and security. By modeling IAM as a relational multigraph, learning inductive representations with an RGCN, calibrating prediction confidence, and guiding repair with an AWS Capability Model, we can safely remediate privilege escalation risks. We make five distinct contributions spanning graph representation, calibrated learning, combinatorial repair, formal multi-completion verification, and interactive visual tooling."

---

## Slide 4: Relational Graph Formulation & Capability Model $C(a, r, c)$

### Slide Content
- **Heterogeneous Graph Schema**: $G = (V, E, \mathcal{R})$
  - Node Types $V$: `User`, `Role`, `Policy`, `Group`, `Resource`, `Condition`.
  - Edge Relations $\mathcal{R}$: `AttachedTo`, `AssumesRole`, `HasPolicy`, `TargetsResource`, `PassesRoleTo`, `AppliesCondition`.
- **The AWS Action Capability Model $C(a, r, c)$**:
  - AWS authorization semantics are non-uniform: naive solvers generate invalid policies by applying resource constraints to actions that AWS forbids.
  $$\mathcal{T}(a) = \begin{cases} \text{ARN-Restricted} & \text{if } a \text{ supports resource-level permissions (e.g., } \texttt{s3:GetObject}\text{)} \\ \text{Global Only } (\texttt{"Resource": "*"}) & \text{if } a \text{ requires wildcard (e.g., } \texttt{iam:CreatePolicyVersion}\text{)} \end{cases}$$
  - $C(a, r, c)$ strictly dictates valid remediation actions:
    1. If $\mathcal{T}(a) == \text{ARN-Restricted}$: Narrow resource ARN from `*` to verified operational prefix.
    2. If $\mathcal{T}(a) == \text{Global}$: Inject verified IAM condition keys (e.g., `aws:PrincipalArn`, `aws:RequestedRegion`).
    3. Excision (Action Deletion) is permitted **only** as a fallback when no valid scoping exists.

### Visual Layout
- Left: Heterogeneous schema diagram with colored nodes and typed directed edges.
- Right: Decision flow chart showing how $C(a, r, c)$ validates candidate AST repairs against the AWS authorization model.

### Speaker Notes
> "A major failure mode in previous automated IAM repair literature is ignoring AWS authorization quirks. For instance, AWS explicitly prohibits resource-level ARNs for actions like `iam:CreatePolicyVersion`—they require `Resource: '*'`. If a naive SMT solver restricts such an action to an ARN, AWS rejects the policy at deployment time. We formalize this through the Action Capability Model $C(a, r, c)$. Our generator and repair engine query this model to ensure that every synthesized policy diff is strictly valid under real AWS IAM parsing semantics."

---

## Slide 5: Calibrated Relational Learning

### Slide Content
- **2-Layer Relational Graph Convolutional Network (RGCN)**:
  - Inductive message passing over heterogeneous relations $r \in \mathcal{R}$:
    $$h_i^{(l+1)} = \sigma \left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} \right)$$
  - Regularized with basis-sharing decomposition: $W_r = \sum_{b=1}^B a_{rb} V_b$ to prevent overfitting on rare escalation motifs.
- **Link Prediction Decoder**:
  - DistMult bilinear scoring: $s(u, r, v) = h_u^T R_r h_v$.
- **Post-Hoc Temperature Calibration (ECE)**:
  - Raw neural link probabilities are notoriously overconfident, invalidating probabilistic risk estimation.
  - We apply Platt temperature scaling: $\hat{p} = \sigma(z / T)$ where $T > 0$ is optimized via NLL on validation sets.
  - Achieves Expected Calibration Error (ECE) $< 0.05$ across all experimental seeds.

### Visual Layout
- Left: Neural architecture diagram with 2 RGCN layers, basis sharing, and DistMult link predictor.
- Right: Reliability diagram (Calibration Curve) showing raw vs. temperature-scaled probabilities aligned with the perfect $y=x$ diagonal.

### Speaker Notes
> "To discover missing attack paths, we use a 2-layer RGCN. Unlike standard GCNs, the RGCN maintains relation-specific transformation matrices, allowing the model to learn the difference between assuming a role versus attaching a policy. We regularize the parameters using basis sharing. Crucially, raw neural networks are poorly calibrated—they output overconfident probabilities like 0.99 for edges that are only 60% likely. We calibrate the network using temperature scaling, bringing our Expected Calibration Error below 0.05. This calibration guarantees that our downstream probabilistic risk calculations reflect true likelihoods."

---

## Slide 6: Empirical Result 1: Latent Discovery Under Structured Masking (RQ1)

### Slide Content
- **Evaluation Across Masking Conditions**:
  - **Condition A (Uniform Random Masking)**: $10\%$ to $40\%$ missing edges.
  - **Condition B (Cross-Account Masking)**: External trust edges hidden.
  - **Condition E (Adversarial Bridge Masking)**: Critical escalation bottleneck edges deliberately omitted from telemetry.

| Method | Condition A (Random 20%) | Condition B (Cross-Account) | Condition E (Adversarial Bridge) |
| :--- | :---: | :---: | :---: |
| **Deterministic BFS (PMapper)** | $0.62 \pm 0.04$ | $0.51 \pm 0.03$ | **$0.50 \pm 0.00$ (0.0% Recall)** |
| **Random Forest Baseline** | $0.78 \pm 0.02$ | $0.74 \pm 0.03$ | $0.69 \pm 0.03$ |
| **Heuristic Motif Completion** | $0.81 \pm 0.02$ | $0.79 \pm 0.02$ | $0.75 \pm 0.02$ |
| **2-Layer RGCN (Ours)** | **$0.92 \pm 0.01$** | **$0.89 \pm 0.02$** | **$0.91 \pm 0.02$** |

- **Key Takeaway**:
  - Deterministic BFS drops to random chance ($0.50$ AUC, $0\%$ recall) under adversarial bridge masking because the required link is absent from observed telemetry.
  - RGCN outperforms tabular baselines by $+0.22$ AUC on adversarial bridges by propagating high-order structural context.

### Visual Layout
- Grouped bar chart comparing ROC-AUC across the four methods under Conditions A, B, and E.
- Highlighting the complete collapse of BFS under Condition E.

### Speaker Notes
> "Here are our empirical results for Research Question 1. We tested four link discovery strategies across uniform random masking, cross-account masking, and adversarial bridge masking. Notice the red bar for BFS under Condition E: exactly 0.50 ROC-AUC and 0% recall. Because BFS relies entirely on observed edges, omitting a single bridge relation completely blinds it. Tabular Random Forest reaches 0.69, but our 2-layer RGCN achieves 0.91 ROC-AUC. By capturing multi-hop relational embeddings, the RGCN reliably infers hidden attack paths that traditional tools miss."

---

## Slide 7: Empirical Result 2: The Security–Utility Pareto Frontier (RQ2)

### Slide Content
- **The Central Security–Utility Tradeoff**:
  $$\min_{P^*} \quad \text{Cost}(P \to P^*) + \lambda \cdot \text{Breakage}(\mathcal{D}^{T_1}, \mathcal{D}^{T_2}) + \gamma \cdot \mathbb{E}_{H \sim \mathbb{P}(H \mid G_o)}[\text{Risk}(G_o \cup H, P^*)]$$

| Remediation Method | Status | Level 3 Risk ($G^*$) | $T_1$ Breakage | $T_2$ Breakage | Transformation Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Method 0: Oracle ($G^*$)** | *Pareto Optimal* | $0.00$ | $0.0\%$ | $0.0\%$ | $1.0$ |
| **Method 1: No Action (Identity)** | *Pareto Optimal* | **$1.00$ (Critical)** | $0.0\%$ | $0.0\%$ | $0.0$ |
| **Method 2: Coarse Deletion (IAMPERE)** | **Dominated** | $0.00$ | **$29.6\%$** | **$17.2\%$** | $10.0$ |
| **Method 3: Log-Only Tightening (PolicyRefiner)** | **Dominated** | $0.00$ | $0.0\%$ | **$0.8\%$** | $4.0$ |
| **Method 4a: Point-Estimate ($\hat{p} \ge 0.5$)** | *Pareto Optimal* | $0.00$ | $0.0\%$ | $0.0\%$ | $1.0$ |
| **Method 4b: Uncertainty-Aware (Ours)** | **Pareto Optimal** | **$0.00$** | **$0.0\%$** | **$0.0\%$** | **$1.0$** |

- **Pareto Dominance**:
  - Our uncertainty-aware method strictly dominates Coarse Deletion and Log-Only Tightening.
  - Matches the theoretical Oracle bound: **0.00 residual risk, 0.0% operational breakage**.

### Visual Layout
- 2D Scatter Plot: X-axis = $T_2$ Operational Breakage (%), Y-axis = Level 3 Residual Attack Risk.
- Shows Method 2 far to the right (high breakage), Method 1 at the top (maximum risk), and Method 4b anchored at $(0.0, 0.0)$ on the Pareto frontier.

### Speaker Notes
> "Slide 7 shows the centerpiece of our research: the Security-Utility Pareto tradeoff. When organizations remediate IAM, they face a dilemma: do nothing and stay vulnerable, or break production. Method 2—coarse deletion like IAMPERE—eliminates attack paths but breaks 17% to 30% of legitimate API calls! Method 3—log-only tightening—is brittle against operational drift. Method 4b, our uncertainty-aware repair, achieves the ideal outcome: exactly 0.00 residual risk on the ground truth graph, exactly 0.0% breakage on baseline and future workloads, and minimal AST edits. It strictly Pareto-dominates the existing literature."

---

## Slide 8: Capability-Compliant Minimal Policy Diff

### Slide Content
- **The Flaw of Coarse Deletion (Prior Work)**:
  ```diff
  - { "Action": ["iam:PassRole", "iam:CreateLambdaFunction"], "Resource": "*" }
  ```
  *Consequence*: Stripping `iam:PassRole` crashes developer CI/CD deployment pipelines immediately.
- **Our Capability-Compliant Synthesis ($C(a, r, c)$)**:
  ```diff
    {
      "Sid": "RemediatedDeploymentPolicy",
      "Effect": "Allow",
      "Action": ["iam:PassRole", "iam:CreateLambdaFunction"],
  -   "Resource": "*"
  +   "Resource": "arn:aws:iam::123456789012:role/service-role/prod-*-role",
  +   "Condition": {
  +     "StringEquals": {
  +       "iam:PassedToService": "lambda.amazonaws.com"
  +     }
  +   }
    }
  ```
- **Guarantees**:
  - Legitimate Lambda deployments matching the role pattern continue running without disruption.
  - Escalation to high-privilege administrative roles (`arn:aws:iam::*:role/AdminRole`) is blocked deterministically.

### Visual Layout
- Side-by-side code diff viewer with red highlighting for broken coarse deletion vs. green highlighting for scoped ARN and condition injection.

### Speaker Notes
> "Let's see what this looks like in practice. On the left is what prior tools like IAMPERE produce: they see that `iam:PassRole` is on an attack path, so they delete the permission entirely. The next morning, your engineering teams cannot deploy microservices. On the right is our synthesized diff. Because our engine understands the capability model of `iam:PassRole`, it narrows the wildcard resource to the approved service-role ARN pattern and injects the `PassedToService` condition key. Legitimate developer deployments proceed uninterrupted, while the privilege escalation vector is completely severed."

---

## Slide 9: Scalability & Candidate Pruning (RQ3)

### Slide Content
- **The Challenge**: Evaluating all potential edge cuts across thousands of enterprise nodes is computationally intractable ($O(2^{|E|})$).
- **Uncertainty-Guided Candidate Pruning**:
  - Sample $M$ latent graph completions: $G^{(m)} = G_o \cup H^{(m)}$.
  - Extract min-cut candidate edges intersecting escalation witnesses across completions.
  - Retain only high-probability bottleneck candidates: $\text{Freq}(e \in \text{Cut}) \ge \tau$.
- **Empirical Scalability Across Enterprise Scales ($N \in [500, 5{,}000]$)**:

| Scale ($|V|$) | Edges ($|E|$) | Total Latency | Candidate Recall | Optimality Gap | Safety Coverage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **500** | $550$ | **$8.5\text{ ms}$** | $100.0\%$ | $0.00\%$ | $100.0\%$ |
| **1,000** | $1,135$ | **$17.3\text{ ms}$** | $100.0\%$ | $0.00\%$ | $100.0\%$ |
| **3,000** | $3,332$ | **$15.9\text{ ms}$** | $100.0\%$ | $0.00\%$ | $100.0\%$ |
| **5,000** | $5,523$ | **$17.2\text{ ms}$** | $100.0\%$ | $0.00\%$ | $100.0\%$ |

- **Result**: Sub-second execution ($<0.02\text{ s}$ vs. $3.0\text{ s}$ budget), zero optimality gap, and $100\%$ candidate recall.

### Visual Layout
- Latency scaling curve plot showing runtime flatlining well below $20\text{ ms}$ up to $5,000$ nodes.
- Highlighting the $3.0\text{ s}$ operational SLA line far above our measured runtimes.

### Speaker Notes
> "In enterprise clouds, authorization graphs grow into thousands of nodes. If an algorithm takes minutes to run, it cannot be used in real-time CI/CD gates. We introduced uncertainty-guided candidate pruning. By sampling latent completions and pruning non-witness edges, we reduce the search space by over 95%. As you can see in the benchmark table, our end-to-end pipeline evaluates a 5,000-node graph in just 17 milliseconds—orders of magnitude faster than our 3-second ceiling—while maintaining 100% candidate recall and a 0.00% optimality gap."

---

## Slide 10: Interactive Triage & Visual Verification Dashboard

### Slide Content
- **Purpose-Built Security Operations UI** (Built with Streamlit + PyVis):
  1. **Executive Posture Banner**: Immediate visibility into observed attack paths, latent risks, and enterprise health score.
  2. **Ego-Network Graph Canvas**: Interactive physics-based graph scoped to $k \le 2$ hops around high-risk principals, preventing DOM lag.
  3. **Side-by-Side Policy Diff Inspector**: Color-coded JSON diff with inline AWS capability validation badges.
  4. **What-If Verification Simulator**: Instant visual re-evaluation showing reachability dropping to zero upon applying patch.
- **Production CLI Tooling**:
  ```bash
  # Launch interactive triage suite
  iam dashboard
  # Headless static report export
  iam export --num-nodes 500 --output-dir reports/
  ```

### Visual Layout
- Screenshots of the Streamlit dashboard:
  - Top: High-contrast dark-mode security banner with risk badges.
  - Bottom-Left: PyVis ego-network highlighting target compromise role.
  - Bottom-Right: Unified policy diff viewer.

### Speaker Notes
> "Security engineers need intuitive tools, not raw tensor outputs. We built a full-featured Streamlit and PyVis dashboard. It features an Executive Posture Banner, an interactive ego-network canvas scoped to two hops to prevent browser lag, a side-by-side policy diff viewer, and an interactive 'What-If' simulator where operators can toggle proposed patches and watch the attack paths disappear in real time. We also provide a unified CLI tool, `iam`, allowing security teams to run headless triage in CI/CD pipelines."

---

## Slide 11: Soundness Boundary & Verification Guarantees

### Slide Content
- **The Soundness Claim (Boundary Definition)**:
  - Neural networks are statistical predictors; **we do not claim the GNN is sound**.
  - Instead, we establish a **Symbolic Soundness Boundary**:
    1. The AST evaluator conforms strictly to formal AWS authorization logic.
    2. The patch verifier is **sound relative to the evaluated completion set $\mathcal{H}_M$**.
- **Three-Level Risk Reporting**:
  - **Level 1 (Observed Risk on $G_o$)**: Attack paths visible in raw telemetry ($0.00$ after repair).
  - **Level 2 (Latent Risk across $\mathcal{H}_M$)**: Expected reachability under Monte Carlo completions ($\le \delta$).
  - **Level 3 (Ground-Truth Residual Risk on $G^*$)**: Verification against true hidden environment state.
- **Fail-Safe Property**:
  - If a patch fails verification on any sampled completion $H^{(m)}$, the repair is rejected and escalated to human operators.

### Visual Layout
- Tiered safety guarantee pyramid:
  - Base: Symbolic AST Engine (Deterministic Soundness).
  - Middle: Completion Set Verifier ($\delta$-bounded risk guarantee).
  - Peak: Statistical Link Predictor (Inference & Candidate Discovery).

### Speaker Notes
> "In security research, precision about soundness is paramount. We explicitly clarify our soundness boundary: the neural network is a statistical heuristic and is not sound. However, our verification engine IS sound relative to the formal AST semantics and the sampled completion set. We report security across three distinct levels: observed risk, latent risk across sampled completions, and ground-truth residual risk. If a synthesized patch fails verification on even one completion, it is rejected automatically. Operators always know the exact confidence bounds of their security posture."

---

## Slide 12: Scientific Roadmap to Master's Thesis

### Slide Content
- **Tier 1 (Course Prototype) Milestone Completed**:
  - Successfully validated hypothesis: Calibrated relational learning + capability repair achieves Pareto optimality.
- **Tier 2 (18-Month Master's Thesis Roadmap)**:
  - **Months 4–6 (Gate G1)**: Conformance testing against AWS IAM Sandbox; ingest TAC-Bench; zero-shot synthetic $\to$ TAC-Bench transfer.
  - **Months 7–10 (Gate G2)**: Correlated link sampling (joint distribution modeling); chance-constrained Z3 / MaxSAT solver scaling.
  - **Months 11–14 (Gate G3)**: Real-world Terraform & CloudFormation evaluation across 30–50 open-source enterprise architectures.
  - **Months 15–18**: Formal verification against FMCAD 2025 IAM specification; thesis defense; publication packaging (IEEE S&P / USENIX Security).

```text
[M1-3: Course Prototype] ──► Gate G0 (Passed)
[M4-6: AWS Sandbox & TAC-Bench] ──► Gate G1
[M7-10: Correlated Z3 Repair] ──► Gate G2
[M11-14: Real IaC Evaluation] ──► Gate G3
[M15-18: Formal Thesis & Defense]
```

### Speaker Notes
> "To conclude, this 10-week project has fully met its Tier 1 objectives and passed Gate G0. But this is the foundation for an 18-month Master's thesis. Over the coming months, we will conformance-test our capability engine against live AWS sandboxes, evaluate zero-shot transfer on TAC-Bench, develop joint correlated sampling models, and benchmark against real Terraform configurations from production repositories. We have established that uncertainty-aware least-privilege repair is not only theoretically sound, but practically viable and scalable. Thank you, and I look forward to your questions."

---

## Appendix: Reproduction & Artifact Availability

- **Single-Command Reproduction**:
  ```bash
  git clone https://github.com/DukeTheOne/IAM_GNN.git
  cd IAM_GNN
  ./reproduce_results.sh --fast    # Smoke test (< 45s)
  ./reproduce_results.sh --full    # Full publication benchmark suite
  ```
- **Public Python CLI**:
  ```bash
  pip install -e .
  iam --help
  ```
- **Code Quality Metrics**:
  - 100% `ruff` formatting & linting compliance.
  - 100% `mypy --strict` type checking coverage (53 source modules).
  - 410 unit and integration tests passing ($93\%$ code coverage).
