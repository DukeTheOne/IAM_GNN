# Week 09: Visual Triage Dashboard, Ego-Network Explainability & High-Throughput Scalability Engine

**Goal**: Formulate, architect, and deliver the production-grade visual triage and explainability layer for Phase 4. Integrate Streamlit and PyVis into an interactive, multi-page security operations dashboard that ingests calibrated latent link inferences ($\mathbb{P}(H \mid G_o)$) and three-level completion-set verifications. Implement strict $k \le 2$ ego-network canvas rendering to eliminate DOM freezing on enterprise graphs ($500 \to 50{,}000$ nodes), build side-by-side AST policy diffs with live in-memory patch simulation, design a multi-tiered caching architecture (`@st.cache_data`, `@st.cache_resource`, stateful pre-indexing) guaranteeing $< 3.0\text{ s}$ interactive latency, and provide a headless CLI triage exporter ensuring automated CI/CD compliance and robust offline fallback.
**Estimated Effort**: 17 Hours
**Target Milestone**: **M4 Foundation: Visual Triage UI & Interactive Explainability Engine**

---

## Architectural Decisions & Future Scaling Context

Following the formal completion of Milestone M3 (Synthetic Temporal Workload, Completion-Set Verifier & Central Tradeoff Experiment) in [week_08.md](file:///Users/duke/IAM/work_log/phase_03/week_08.md), Phase 4 transitions from backend algorithmic synthesis to **operator explainability, visual triage, and production packaging**.

As established in [proposal.md](file:///Users/duke/IAM/proposal.md) and [blueprint.md](file:///Users/duke/IAM/blueprint.md), the dashboard is a **deliverable for human-in-the-loop explainability**, while the symbolic verifier remains the sole mathematical arbiter of security and safety.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                                 PHASE 4 WEEK 09 VISUAL TRIAGE ARCHITECTURE                         │
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│ [Core Backend Inputs]                                                                             │
│   ├── Observed Multigraph G_o (NetworkX / HeteroData) & Ground Truth G*                           │
│   ├── Calibrated RGCN Link Predictions P̂(H | G_o) with ECE Metrics                                │
│   ├── Sampled Latent Completions {H_i}_{i=1}^M & Attack Reachabilities                            │
│   ├── Synthesized Minimal Repairs P* (Greedy / Z3 MaxSAT) with C(a, r, c) Admissibility           │
│   └── Verification Reports (Level 1 G_o, Level 2 {G_o ∪ H_i}, Level 3 G*, Temporal Replay)        │
│   ▼                                                                                               │
│ [High-Throughput Caching & State Management Layer] (Step 9.1)                                     │
│   ├── @st.cache_resource: In-memory graph topologies, model weights, and capability models        │
│   ├── @st.cache_data: Pre-computed posture KPIs, triage catalogs, and Pareto tradeoff tables     │
│   └── DashboardState: Reactive session store with O(1) principal lookup and filter indices        │
│   ▼                                                                                               │
│ [Dual-Channel Explainability Presentation Engine]                                                  │
│   ├── Channel 1: Interactive Streamlit Web UI (Steps 9.1 – 9.3)                                   │
│   │   ├── Posture Banner: Scanned Identities, Evaluated Policies, Flagged Latent Attack Vectors  │
│   │   ├── Triage Table: Multi-column sorting, severity tiers, and calibrated confidence badges    │
│   │   ├── Ego-Network Canvas (Step 9.2): k ≤ 2 hop scoped force-directed graph (vis.js)           │
│   │   │   ├── Observed edges: Solid blue/slate lines with relation annotations                    │
│   │   │   ├── Latent GNN bridges: Dashed crimson lines with P̂ probability badges                  │
│   │   │   ├── Crown jewel assets: Red diamond nodes with attack reachability highlights           │
│   │   │   └── Physics freezing: Post-layout stabilization yielding 0% browser idle CPU load       │
│   │   └── Policy Diff & Patch Simulator (Step 9.3):                                              │
│   │       ├── Side-by-side colorized JSON AST diffs highlighting resource ARN scoping             │
│   │       ├── Capability Admissibility Badge: C(a, r, c) = Valid confirmation                     │
│   │       └── "Simulate Patch" Toggle: Real-time in-memory edge severing on ego-canvas            │
│   └── Channel 2: Headless CLI Static Exporter (Step 9.4 - P2 Fallback)                            │
│       ├── scripts/export_triage_report.py: Zero-browser standalone batch exporter                 │
│       └── Output: Self-contained interactive HTML PyVis files + Markdown diff audit logs          │
│   ▼                                                                                               │
│ [Handoff to Week 10] Automated Reproduction Package, Full Pipeline Benchmarks & Demo (M4 Release) │
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

### 1. The Explainability & Safety Claim Boundary
A graphical interface can illustrate an attack path, but it cannot certify security. We maintain strict separation of responsibilities:
1. **The Symbolic Verifier is the Safety Authority**: All safety claims, residual attack reachability probabilities, and zero-regression guarantees are computed exclusively by the `CompletionSetVerifier` ([`src/iam/verifier/verifier.py`](file:///Users/duke/IAM/src/iam/verifier/verifier.py)) across the three verification boundaries ($G_o$, $\{G_o \cup H_i\}_{i=1}^M$, and $G^*$).
2. **The Dashboard is an Operational Explainability Tool**: Its purpose is cognitive triage—translating complex multi-hop relational dependencies into clear, intuitive ego-networks that allow cloud security operators to understand *why* a latent path was flagged, *what* capability-compliant patch is proposed, and *how* the patch severs the bottleneck without disrupting business operations.

### 2. Ego-Network Scoping as a Fundamental Scaling Invariant ($k \le 2$)
Enterprise cloud environments contain thousands to tens of thousands of identities, roles, groups, and policies ($N \in [500, 50{,}000]$).
- **The Browser DOM Collapse Trap**: Force-directed physics algorithms (Barnes-Hut, ForceAtlas2) in vis.js/PyVis exhibit $\mathcal{O}(N^2)$ computational complexity. Rendering full graphs with $>500$ nodes directly in the browser causes severe DOM lag, frame rate collapse ($<5\text{ fps}$), and browser tab crashes (Risk **R-05**).
- **The $k \le 2$ Hop Boundary**: For any selected high-risk principal $s$, the dashboard extracts strictly the $k \le 2$ hop directed ego-network $\mathcal{G}_{\text{ego}}(s, k=2)$ spanning active escalation chains and intermediate policies reaching toward crown-jewel assets.
- **Complexity Bound**: This bounds the rendered canvas to $N_{\text{ego}} \le 30$ nodes and $E_{\text{ego}} \le 60$ edges, guaranteeing smooth **60fps interactive exploration** regardless of whether the underlying global graph contains 500 or 50,000 nodes.

### 3. Dual-Channel Delivery: Rich Streamlit UI + Headless CLI Exporter
Per the project fallback rules codified in [blueprint.md](file:///Users/duke/IAM/blueprint.md) (Section 1.1):
> *"Fallback rules: if the dashboard is not ready, present a static graph and a JSON policy diff."*

To ensure full resilience against headless server environments, containerized evaluation pipelines, and CI/CD automated gates, Phase 4 implements **Dual-Channel Delivery**:
1. **Interactive Channel**: A dynamic, multi-page Streamlit web application with real-time PyVis canvas embedding, search filters, and live patch simulation.
2. **Headless Channel**: A standalone CLI command (`scripts/export_triage_report.py`) that exports self-contained HTML PyVis ego-subgraphs and formatted Markdown policy diffs to disk without requiring a running browser, GUI display, or web server.

---

## Step 9.1: Streamlit Dashboard Skeleton & Posture KPIs (4.0h) - [PLANNED]

**Objective**: Formulate and implement the modular Streamlit application architecture, the centralized reactive state store (`DashboardState`), the global security posture KPI banner, and the high-risk identity triage table equipped with multi-column sorting, calibrated confidence badges, and severity filtering.

**Implementation Details**:
1. **Modular Dashboard Package Scaffolding (`src/iam/dashboard/`)**:
   - [ ] Establish clean separation between presentation, state, and rendering:
     ```
     src/iam/dashboard/
     ├── __init__.py           # Public module interface and exports
     ├── app.py                # Main Streamlit application entry point
     ├── state.py              # Centralized reactive session state & cache manager
     ├── ego_graph.py          # k ≤ 2 ego-network extractor & PyVis HTML synthesizer
     ├── diff_viewer.py        # Colorized side-by-side JSON AST policy diff generator
     └── metrics_banner.py     # Posture KPI cards and Pareto tradeoff visualization
     ```
   - [ ] Multi-page navigation structure:
     - `Overview & Posture`: Global security KPIs, risk distributions, and Pareto tradeoff frontiers.
     - `Identity Triage`: Searchable high-risk principal table and calibrated latent path discovery.
     - `Interactive Remediation`: Ego-network canvas, capability checks, and side-by-side policy diffs.
2. **Centralized Reactive State Manager (`DashboardState` in `state.py`)**:
   - [ ] Encapsulate dataset loading, GNN link predictions, Monte Carlo completions, and verification reports into a typed state object.
   - [ ] Implement `@st.cache_resource` for immutable heavy objects: underlying `IAMGraph` topologies, PyG embeddings, and trained RGCN weights.
   - [ ] Implement `@st.cache_data` for computational results: pre-computed risk rankings, blast radius aggregations, and holdout workload verification reports.
   - [ ] Provide atomic state mutation helpers: `select_principal(principal_arn: str)`, `toggle_simulation(enabled: bool)`, and `filter_identities(severity: str, department: str)`.
3. **Global Security Posture Banner (`metrics_banner.py`)**:
   - [ ] Render responsive metric KPI cards across the top viewport:
     - **Scanned Identities**: Total principals evaluated ($N_{\text{principals}}$).
     - **Evaluated Policies**: Total attached and inline JSON policies.
     - **Latent Risks Flagged**: Count of attack paths unobserved in $G_o$ but flagged by calibrated RGCN ($\hat{p} \ge \theta$).
     - **Max Privilege Tier Exposed**: Highest administrative tier reachable across completions.
     - **Pareto Optimality Status**: Confirmation of uncertainty-aware dominance over coarse deletion and log-only tightening.
4. **High-Risk Identity Triage Table (`app.py` / `state.py`)**:
   - [ ] Tabular view of principals exhibiting privilege escalation vectors:
     - Principal ARN, Department, Observed $G_o$ Risk (0.0 under Condition E), Calibrated Latent Attack Probability ($\widehat{\mathbb{P}}_{\text{MC}} \in [0.0, 1.0]$), High-Value Target Assets Exposed, Recommended Remediation Action.
   - [ ] Dynamic multi-attribute filtering:
     - Filter by Severity Tier (`CRITICAL`: $\widehat{\mathbb{P}} \ge 0.70$, `HIGH`: $0.40 \le \widehat{\mathbb{P}} < 0.70$, `MEDIUM`: $\widehat{\mathbb{P}} < 0.40$).
     - Filter by Department (`DevOps`, `Data/BI`, `SecOps`, `QA`, `Interns`).
     - Free-text search matching principal ARNs and target asset names.
   - [ ] Interactive row selection linking directly to the ego-network remediation canvas.

**Future Scaling & Performance Optimization Considerations**:
- **Vectorized Pre-Sorting**: Pre-compute triage ranking tables as Pandas/Polars DataFrames during initialization so interactive filtering executes in $<10\text{ ms}$ without traversing the raw graph multigraph.
- **Lazy Completion Materialization**: Defer generating the full completion set for benign identities; compute Monte Carlo reachability only for identities flagged by the candidate pruner ($|E_{\text{cand}}| \le 50$).

**Target File Artifacts**:
- [`src/iam/dashboard/state.py`](file:///Users/duke/IAM/src/iam/dashboard/state.py): `DashboardState`, `TriageRecord`, `load_dashboard_state()`, and state caching decorators.
- [`src/iam/dashboard/metrics_banner.py`](file:///Users/duke/IAM/src/iam/dashboard/metrics_banner.py): `render_posture_banner()` and `render_pareto_kpi_card()`.
- [`src/iam/dashboard/app.py`](file:///Users/duke/IAM/src/iam/dashboard/app.py): Main Streamlit dashboard application script.
- [`tests/test_dashboard_state.py`](file:///Users/duke/IAM/tests/test_dashboard_state.py): Unit tests for state initialization, filtering, caching, and serialization.

**Verification & Acceptance Criteria**:
- **Clean Ingestion**: Dashboard successfully loads synthetic enterprise topologies (500–1,500 nodes) with associated RGCN predictions and verification reports in $<1.5\text{ s}$.
- **Zero-Latency Filtering**: Department and severity filtering on a 1,000-identity triage table completes in $<20\text{ ms}$.
- **Condition E Visibility Contrast**: The triage table clearly displays $G_o\text{ Risk} = 0.0$ alongside $\widehat{\mathbb{P}}_{\text{latent}} > 0.70$, visually exposing the "misleading safe" static analysis blind spot.

---

## Step 9.2: PyVis Interactive Canvas & Ego-Network Scoping Engine (5.5h) - [PLANNED]

**Objective**: Implement the `EgoNetworkExtractor` engine that extracts strictly bounded $k \le 2$ hop subgraphs around high-risk identities, and build the PyVis HTML synthesis pipeline that generates color-coded, force-directed graph canvases featuring distinct visual semantics for observed edges, latent GNN bridges, and crown-jewel assets.

**Implementation Details**:
1. **$k \le 2$ Ego-Network Extraction Engine (`EgoNetworkExtractor` in `ego_graph.py`)**:
   - [ ] Parameterized extraction: `extract_ego_subgraph(graph: IAMGraph, source_arn: str, max_depth: int = 2, target_arns: Optional[set[str]] = None) -> EgoSubgraph`.
   - [ ] Traverse directed authorization and delegation relationships starting from `source_arn`:
     - Hop 1: Direct group memberships (`MemberOf`), role assumptions (`AssumesRole`), and attached policies (`AttachedWith`).
     - Hop 2: Intermediate role-chaining delegations (`PassesTo`, service executions) and resource actions (`ActsOn`).
   - [ ] Bridge Injection: Merge latent GNN-inferred edges originating from or incident to nodes within the $k$-hop neighborhood.
   - [ ] Prune irrelevant dead-end nodes that do not participate in paths toward monitored resources or escalation targets.
   - [ ] Strict node bounding: Enforce hard ceiling $|V_{\text{ego}}| \le 30$ to guarantee smooth 60fps browser rendering.
2. **Visual Semantic Encoding System**:
   - [ ] **Node Types & Aesthetic Formatting**:
     - `Principal` (`User` / `Role`): Circular nodes color-coded by department; source identity highlighted with an amber border.
     - `Policy`: Rectangular slate-blue nodes labeled with policy names.
     - `High-Value Resource` (Crown Jewels): Red diamond nodes (`shape="diamond"`, glowing red border, bold label).
     - `Intermediate Services` (e.g. Lambda, EC2): Hexagonal cyan nodes.
   - [ ] **Edge Types & Relationship Styling**:
     - `Observed Static Edges` ($e \in E_{\text{obs}}$): Solid slate-gray arrows with clear relation labels (`AssumesRole`, `PassesTo`).
     - `Latent Inferred Bridges` ($e \in H$): **Dashed vibrant crimson arrows**, line width scaled by calibrated probability $\hat{p} \in [0.0, 1.0]$, annotated with confidence badges (e.g. `[GNN: p̂=0.88]`).
     - `Severed Edges` (Simulated Patches): Dashed semi-transparent gray with strike-through styling.
3. **PyVis HTML Canvas Generation & Physics Stabilization**:
   - [ ] Configure `pyvis.network.Network` with custom CDN-independent styling and embedded CSS for dark-mode alignment.
   - [ ] **Physics Stabilization Safeguard**:
     ```javascript
     physics: {
       enabled: true,
       solver: 'forceAtlas2Based',
       forceAtlas2Based: { gravitationalConstant: -50, centralGravity: 0.01, springLength: 100 },
       stabilization: { enabled: true, iterations: 100, updateInterval: 25 }
     }
     ```
   - [ ] Post-stabilization freeze: Set `physics.enabled = false` after initial layout stabilization so browser CPU utilization drops to **0.0% at rest**.
4. **Streamlit Canvas Embedding**:
   - [ ] Render PyVis HTML output directly inside Streamlit using `st.components.v1.html(html_data, height=550, scrolling=False)`.
   - [ ] Implement seamless iframe resizing and container responsiveness.

**Future Scaling & Graph Rendering Optimization Considerations**:
- **Zero DOM Bloat**: By guaranteeing $|V_{\text{ego}}| \le 30$ and $|E_{\text{ego}}| \le 60$, HTML string generation takes $<15\text{ ms}$, and client-side DOM layout takes $<100\text{ ms}$.
- **Canvas vs. SVG**: PyVis renders via HTML5 Canvas (hardware accelerated), avoiding the heavy DOM tree overhead of pure SVG renderers when interacting with node tooltips.
- **Thesis Scale Tier 2**: The ego-network extractor operates on graph adjacency lists in $\mathcal{O}(d^k)$ time (where $d$ is average node degree), remaining sub-millisecond even when querying an enterprise graph of $50{,}000+$ nodes.

**Target File Artifacts**:
- [`src/iam/dashboard/ego_graph.py`](file:///Users/duke/IAM/src/iam/dashboard/ego_graph.py): `EgoNetworkExtractor`, `EgoSubgraph`, `PyVisGraphSynthesizer`, and `render_ego_canvas()`.
- [`tests/test_ego_graph.py`](file:///Users/duke/IAM/tests/test_ego_graph.py): Unit tests for $k$-hop traversal, node bounding, bridge injection, and HTML synthesis.

**Verification & Acceptance Criteria**:
- **Bounded Canvas Size**: Verified on a 2,500-node graph that extracting the ego-network for any principal yields $\le 30$ nodes and completes in $<5\text{ ms}$.
- **Visual Encoding Fidelity**: Latent bridge edges are distinctly styled with dashed crimson formatting and calibrated likelihood badges.
- **Zero Idle CPU Drain**: Verified that browser physics stabilization ceases after 100 iterations, maintaining stable 60fps interaction without CPU fan spin.

---

## Step 9.3: Remediation & Policy Diff Viewer with Live Patch Simulation (4.5h) - [PLANNED]

**Objective**: Build the colorized side-by-side JSON policy diff viewer, implement real-time validation badges confirming AWS Action Capability Model $C(a, r, c)$ admissibility, display expected security–utility tradeoff impact metrics, and provide a dynamic "Simulate Patch" toggle that severs attack edges on the PyVis canvas in real time.

**Implementation Details**:
1. **Side-by-Side Colorized JSON Policy Diff (`diff_viewer.py`)**:
   - [ ] Synthesize structured AST diffs contrasting the unrepaired baseline policy against the repaired least-privilege policy $\mathcal{P}^*$:
     - Added restrictions / scoped ARNs highlighted in **subtle emerald green**.
     - Excised overly broad wildcards (`"Resource": "*"`) or revoked actions highlighted in **soft crimson red**.
     - Unmodified statements preserved with muted syntax highlighting.
   - [ ] Generate both dual-pane side-by-side HTML diff tables and unified diff blocks compatible with GitHub Flavored Markdown.
2. **Capability Compliance Confirmation Badge ($C(a, r, c) = \text{Valid}$)**:
   - [ ] Display an explicit capability certification panel verifying that the proposed patch adheres to the versioned AWS Service Authorization Reference codified in Week 1:
     - Resource ARN Scoping is confirmed valid (e.g. `iam:PassRole` correctly scoped to `arn:aws:iam::...:role/...`).
     - Wildcard-only actions were not illegally scoped (e.g. `iam:ListRoles` maintained with `Resource: "*"`).
     - Applicable condition keys are validated against the capability snapshot.
3. **Security–Utility Impact Metrics Display**:
   - [ ] Display concise remediation impact statistics derived from the `CompletionSetVerifier`:
     - **Blast Radius Reduction**: $\Delta \text{Risk} = \text{Reachability}(G^*) - \text{Reachability}(G^* \oplus \mathcal{P}^*)$ (e.g. $100\% \to 0\%$).
     - **Observed Historical Breakage**: $0.0\%$ broken requests on $\mathcal{D}^{T_1}$ (guaranteed log conformance).
     - **Temporal Holdout Conformance**: Quantified operational breakage rate on drifted workload $\mathcal{D}^{T_2}$.
     - **AST Complexity Penalty**: Exact count of mutated statements.
4. **Interactive "Simulate Patch" Toggle**:
   - [ ] Streamlit widget `st.toggle("Simulate Patch on Graph Canvas")`:
     - When activated, the ego-network dynamically severs the remediated relation (e.g. removing the overly permissive `iam:PassRole` bridge or restricting target ARN).
     - Re-evaluates ego-reachability in memory, changing the target crown jewel asset status from "EXPOSED (Red)" to "PROTECTED (Green)".
     - Visualizes the severed edge as a muted dashed line with a red cross icon.

**Future Scaling & Dynamic Simulation Optimization Considerations**:
- **In-Memory Graph Masking**: Patch simulation does not re-parse JSON or re-instantiate NetworkX graphs from disk; it applies an ephemeral boolean edge mask $\mathbf{m} \in \{0, 1\}^{|E_{\text{ego}}|}$ in memory, recomputing reachability in $<1\text{ ms}$.
- **Differential Diff Generation**: Policy AST diffs are pre-computed during solver execution and cached in `DashboardState`, requiring zero computation during interactive tab switching.

**Target File Artifacts**:
- [`src/iam/dashboard/diff_viewer.py`](file:///Users/duke/IAM/src/iam/dashboard/diff_viewer.py): `PolicyDiffViewer`, `PolicyASTDiff`, `render_side_by_side_diff()`, and `render_capability_badge()`.
- [`tests/test_diff_viewer.py`](file:///Users/duke/IAM/tests/test_diff_viewer.py): Unit tests for AST diff generation, HTML formatting, and capability badge verification.

**Verification & Acceptance Criteria**:
- **Semantically Accurate Diffs**: Side-by-side diff cleanly isolates the exact resource scoping modification without altering unrelated policy statements.
- **Capability Badge Enforcement**: Correctly flags admissible transformations as `VALID` and synthetic malformed test patches as `INVALID`.
- **Instantaneous Patch Simulation**: Toggling "Simulate Patch" updates the ego-canvas and target reachability state in $<50\text{ ms}$.

---

## Step 9.4: Usability Testing, Headless CLI Exporter & Quality Verification (3.0h) - [PLANNED]

**Objective**: Implement the standalone headless CLI triage exporter (`scripts/export_triage_report.py`) enabling automated offline report generation, conduct comprehensive responsiveness profiling across enterprise graph scales ($500 \to 5{,}000$ nodes), polish UI aesthetics with modern dark-mode ergonomics, and achieve 100% test pass rates across all dashboard components.

**Implementation Details**:
1. **Headless CLI Triage Exporter (`scripts/export_triage_report.py`)**:
   - [ ] Standalone CLI entry point allowing automated evaluation in non-interactive CI/CD pipelines without Streamlit runtime:
     ```bash
     python scripts/export_triage_report.py \
       --topology-path data/enterprise_sample.json \
       --predictions-path experiments/sweep/predictions.json \
       --output-dir reports/triage_export/ \
       --top-k 5
     ```
   - [ ] Exports standalone HTML files for the top-$K$ highest-risk identity ego-networks (viewable in any browser without a web server).
   - [ ] Exports a Markdown summary table of all flagged identities, calibrated risk scores, and side-by-side policy diffs.
2. **Performance & Responsiveness Profiling**:
   - [ ] Benchmark dashboard state loading and ego-network rendering across scaling tiers:
     - 500 nodes (Standard Organization): $< 0.5\text{ s}$ initial load.
     - 1,000 nodes (Mid-Enterprise): $< 1.0\text{ s}$ initial load.
     - 2,500 nodes (Large Multi-Account): $< 2.0\text{ s}$ initial load.
     - 5,000 nodes (Stress Benchmark): $< 3.0\text{ s}$ initial load.
   - [ ] Validate memory consumption: Ensure dashboard memory footprint remains $< 350\text{ MB}$ under continuous interaction.
3. **Ergonomic & Aesthetic Polish**:
   - [ ] Apply sleek dark-mode styling aligned with modern security tools (CrowdStrike / AWS Security Hub inspired palette).
   - [ ] Incorporate clear risk typography, high-contrast badges, and intuitive guidance tooltips.
   - [ ] Validate cross-browser rendering compatibility (Chrome, Safari, Firefox).
4. **Automated Regression & Coverage Test Suite**:
   - [ ] Construct integration test suite testing dashboard state loading, ego-network extraction, diff rendering, and CLI script execution.
   - [ ] Maintain 100% clean passes on `mypy --strict src tests scripts` and `ruff`.

**Future Scaling & Enterprise Extensibility Considerations**:
- **CI/CD Security Gate Integration**: The headless exporter enables cloud engineering teams to run latent-privilege triage as a GitHub Action on pull requests that modify Terraform / IAM policies, blocking merges if an unobserved multi-hop escalation path is introduced.
- **Multi-Account Organization Ingestion (Tier 2 Thesis)**: The data layer is decoupled from AWS account boundaries, enabling seamless visualization of cross-account role assumption chains across dozens of linked accounts.

**Target File Artifacts**:
- [`scripts/export_triage_report.py`](file:///Users/duke/IAM/scripts/export_triage_report.py): Standalone CLI script for headless triage export.
- [`scripts/run_dashboard.py`](file:///Users/duke/IAM/scripts/run_dashboard.py): Convenience launcher script for the Streamlit dashboard.
- [`tests/test_dashboard_e2e.py`](file:///Users/duke/IAM/tests/test_dashboard_e2e.py): End-to-end integration tests for dashboard components and headless export.

**Verification & Acceptance Criteria**:
- **Headless Export Autonomy**: `scripts/export_triage_report.py` executes in $< 2.0\text{ s}$ on a 1,000-node graph, generating valid, standalone HTML and Markdown reports without GUI dependencies.
- **Scalability Budget**: Sub-3-second interactive latency budget strictly respected across all graph scales up to 5,000 nodes.
- **Code Quality**: Zero typing errors on `mypy --strict` and zero linting violations on `ruff`.

---

## Scalability, Latency & Optimization Blueprint (Tier 1 Prototype to Tier 2 Thesis)

To guarantee that the visual triage and explainability layer scales seamlessly from the 10-week course prototype ($500 \to 5{,}000$ synthetic nodes) into the 18-month Master's Thesis research ($50{,}000+$ nodes, real Terraform/CloudFormation IaC, and TAC-Bench 2,500 tasks), we codify the following explicit scalability and optimization contracts:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                       SCALABILITY, LATENCY & OPTIMIZATION ARCHITECTURE                             │
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│ Dimension             │ Tier 1 Prototype (10 Weeks)           │ Tier 2 Thesis Core (18 Months)    │
+───────────────────────┼───────────────────────────────────────┼───────────────────────────────────+
│ Graph Node Scale (N)  │ 500 – 5,000 Nodes                     │ 50,000+ Nodes (Multi-Account IaC) │
│ Graph Edge Scale (E)  │ 2,000 – 25,000 Edges                  │ 250,000+ Directed Edges           │
│ Canvas Scope          │ k ≤ 2 Ego-Network (|V_ego| ≤ 30)      │ Hierarchical Account Partitioning │
│ Rendering Engine      │ PyVis HTML5 Canvas (Vis.js)           │ WebGL Accelerated Graph Renderer  │
│ Physics Model         │ ForceAtlas2 with 100-iter freeze      │ Pre-computed static layout cache  │
│ State Caching         │ @st.cache_resource + @st.cache_data   │ Redis / LMDB persistent cache     │
│ Interactive Latency   │ < 3.0 s End-to-End                    │ < 1.0 s Subgraph Query Latency    │
│ Headless Automation   │ Python CLI (export_triage_report.py)  │ GitHub Action PR Review Bot       │
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

### 1. Computational Complexity & Algorithmic Bounds
1. **Ego-Network Extraction**:
   - The extraction algorithm traverses directed edges using breadth-first search with depth limit $k=2$.
   - Time Complexity: $\mathcal{O}(d^k)$, where $d$ is the maximum out-degree of visited identities. Because enterprise role assumptions and policy attachments rarely exceed $d \approx 15$, the visited frontier contains $\le 225$ candidate paths.
   - Pruning to monitored crown-jewel assets reduces the extracted subgraph to $|V_{\text{ego}}| \le 30$ in $\mathcal{O}(|V_{\text{frontier}}|)$ time ($< 2\text{ ms}$).
2. **Policy AST Diff Generation**:
   - Time Complexity: $\mathcal{O}(|\mathcal{S}|)$, where $|\mathcal{S}|$ is the number of statements in the target policy (typically $1 \le |\mathcal{S}| \le 10$).
   - String normalization and diff generation completes in $< 1\text{ ms}$.
3. **Candidate Pruning Bound (RQ3 Foundation)**:
   - Uncertainty-guided candidate pruning restricts solver and verifier search spaces to $|E_{\text{cand}}| \le 50$.
   - This ensures that interactive remediation and dynamic patch simulation never solve an NP-hard problem over the global multigraph.

### 2. Browser DOM & Client-Side Rendering Optimization
- **The Browser Memory Leak Barrier**: Naive Streamlit components that re-inject large HTML blobs cause cumulative browser memory leaks during extended triage sessions.
- **Mitigation Protocol**:
  1. All PyVis HTML canvases generated embed a unique, deterministic canvas container ID with complete CSS scoping.
  2. The vis.js physics engine is unconditionally destroyed and frozen after 100 stabilization iterations via `network.setOptions({ physics: { enabled: false } })`.
  3. No raw node or edge lists are stored in client-side DOM attributes; tooltips are rendered on-canvas using lightweight Canvas text drawing.

### 3. Multi-Tiered Caching Hierarchy & Latency Budget
Streamlit executes Python scripts top-to-bottom upon every user interaction (e.g. clicking a table row or adjusting a filter slider). Without caching, re-loading 5,000-node graphs and re-running GNN inference would take 5–10 seconds per click.
We enforce a strict 3-tier caching hierarchy:
1. **Tier A (Static Structural Cache - `@st.cache_resource`)**:
   - Stores loaded `IAMGraph` topologies, pre-trained `HeteroRGCNEncoder` model weights, and the `CapabilityModel` data layer.
   - Instantiated once per server process; shared across all user sessions; zero garbage collection thrashing.
2. **Tier B (Analytical Data Cache - `@st.cache_data`)**:
   - Stores computed `TriageRecord` tables, Pareto dominance frontiers, and baseline workload replay metrics.
   - Keyed on dataset hash and model checkpoint hash; invalidated only when the underlying topology file changes.
3. **Tier C (Reactive Session Cache - `st.session_state`)**:
   - Stores user-specific transient selections: selected principal ARN, active department filters, and "Simulate Patch" boolean toggle state.
   - Modifying a filter triggers instantaneous re-rendering in $< 30\text{ ms}$ without accessing disk or running neural forward passes.

### 4. Memory Footprint & Resource Management
- **PyTorch Tensor Detachment**: All GNN embeddings and probability tensors are explicitly detached from the PyTorch computation graph (`.detach().cpu().numpy()`) prior to caching in `DashboardState`. This eliminates gradient history retention and prevents GPU/MPS memory accumulation.
- **Garbage Collection Discipline**: Intermediate NetworkX subgraph copies created during ego-network extraction are explicitly scoped within local functions, allowing Python's reference counter to reclaim memory immediately.

### 5. Tier 2 Master's Thesis Expansion Roadmap
- **Scaling to 50,000+ Nodes**: In Tier 2, enterprise cloud environments with multi-account AWS Organizations will be partitioned hierarchically by AWS Account ID (`account_id`), allowing operators to view cross-account trust boundaries at macro scale and drill into $k \le 2$ ego-networks at micro scale.
- **TAC-Bench Zero-Shot Visualizer**: The dashboard architecture will directly ingest the 2,500 benchmark tasks from TAC-Bench, displaying ground-truth attack chains alongside zero-shot RGCN predictions.
- **IaC Pull Request Review Bot**: The headless exporter (`scripts/export_triage_report.py`) will be packaged as a standalone GitHub Action that scans Terraform / CloudFormation diffs in CI/CD pipelines, posting visual ego-network SVGs and capability-compliant policy patches directly to developer pull request comments.

---

## Cumulative Verification & Quality Status (Phase 4 Foundation)

- **Unit & Integration Tests**: **377 / 377 tests passing** across the complete project test suite (`pytest -v`).
- **Core Engine Coverage**: **>90% coverage** maintained across parser, generator, models, optimizer, and verifier modules:
  - `src/iam/parser/` (93% coverage)
  - `src/iam/generator/` (94% coverage)
  - `src/iam/models/` (92% coverage)
  - `src/iam/optimizer/` (91% coverage)
  - `src/iam/verifier/` (92% coverage)
- **Milestone Progression**:
  - Milestone M1: Environment-Split IAM Corpus & Structured Masking (Phase 1) - **Achieved**
  - Milestone M2: Latent Relation Benchmark & Gate G0 GO Decision (Phase 2) - **Achieved**
  - Milestone M3: Security–Utility Tradeoff Evaluation & Pareto Dominance (Phase 3) - **Achieved**
  - Milestone M4: Visual Triage UI, Explainability & Open-Source Release (Phase 4) - **In Progress**
- **Strict Typing Compliance**: **Zero errors** across all source files on `mypy --strict src tests scripts`.
- **Formatting & Linting**: **Zero issues** on `ruff check` and `ruff format --check`.

---

## Week 09 Deliverables Checklist & Handoff to Week 10

| Module / Component | Target Artifact | Status | Consumer in Week 10 |
| :--- | :--- | :---: | :--- |
| **Dashboard State & Cache Manager** | [`src/iam/dashboard/state.py`](file:///Users/duke/IAM/src/iam/dashboard/state.py) | [PLANNED] | Data loader for interactive UI & reproduction scripts |
| **Ego-Network Extraction Engine** | [`src/iam/dashboard/ego_graph.py`](file:///Users/duke/IAM/src/iam/dashboard/ego_graph.py) | [PLANNED] | Visual graph canvas for course presentation & demo |
| **Side-by-Side Policy Diff Viewer** | [`src/iam/dashboard/diff_viewer.py`](file:///Users/duke/IAM/src/iam/dashboard/diff_viewer.py) | [PLANNED] | Explainability diff generator for slide deck & report |
| **Posture KPIs & Pareto Banner** | [`src/iam/dashboard/metrics_banner.py`](file:///Users/duke/IAM/src/iam/dashboard/metrics_banner.py) | [PLANNED] | Executive summary cards for Milestone M4 presentation |
| **Streamlit Web Application** | [`src/iam/dashboard/app.py`](file:///Users/duke/IAM/src/iam/dashboard/app.py) | [PLANNED] | Interactive web UI for 3-minute video walkthrough |
| **Headless CLI Triage Exporter** | [`scripts/export_triage_report.py`](file:///Users/duke/IAM/scripts/export_triage_report.py) | [PLANNED] | Offline report generator for automated testing & CI/CD |
| **Dashboard Launcher Script** | [`scripts/run_dashboard.py`](file:///Users/duke/IAM/scripts/run_dashboard.py) | [PLANNED] | Single-command dashboard launcher for README quickstart |
| **Dashboard Test Suite** | [`tests/test_dashboard_state.py`](file:///Users/duke/IAM/tests/test_dashboard_state.py), [`tests/test_ego_graph.py`](file:///Users/duke/IAM/tests/test_ego_graph.py), [`tests/test_diff_viewer.py`](file:///Users/duke/IAM/tests/test_diff_viewer.py), [`tests/test_dashboard_e2e.py`](file:///Users/duke/IAM/tests/test_dashboard_e2e.py) | [PLANNED] | Automated regression testing for visual triage subsystem |
