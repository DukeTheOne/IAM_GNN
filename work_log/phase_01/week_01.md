# Week 01: Environment Setup, IAM Parsing & Action Capability Model

**Goal**: Establish the repository infrastructure, build a correct AWS IAM JSON AST parser, and codify the AWS Action Capability Model $C(a, r, c)$ to prevent semantically invalid repairs.
**Estimated Effort**: 16 Hours

---

## Step 1.1: Core Environment & Toolchain Initialization (3.0h) - [COMPLETED]

**Objective**: Set up a robust, modern Python development environment and project scaffolding.

**Implementation Instructions**:
1. **Virtual Environment & Dependency Management**:
   - [x] Use `poetry`, `pip-tools`, or standard virtual environment to manage dependencies.
   - [x] Initialize a Python 3.10+ project.
   - [x] Required dependencies: `torch`, `torch_geometric`, `networkx`, `scipy`, `z3-solver`.
   - [x] Development dependencies: `pytest`, `mypy`, `ruff`, `pre-commit`.

2. **Tooling Configuration**:
   - [x] Create a `pyproject.toml` configuring `ruff` (linter and formatter) and `mypy` (strict type checking).
   - [x] Set up `.pre-commit-config.yaml` to run `ruff` and `mypy` on every commit.

3. **Repository Structure**:
   - [x] Create the following modular layout:
     ```text
     iam-graph-learning/
     ├── data/                  # Synthetic schemas, generators, and split datasets
     ├── src/
     │   ├── parser/            # IAM AST, AWS action capability model C(a,r,c)
     │   ├── generator/         # Parameterized enterprise generator & PE motifs
     │   ├── models/            # PyG RGCN, calibration module, and baselines
     │   ├── optimizer/         # Candidate pruning, min-cut, and Z3 MaxSAT repair
     │   ├── verifier/          # Symbolic reachability checker & log replay
     │   └── dashboard/         # Streamlit and PyVis UI
     ├── tests/                 # Unit, integration, and benchmark tests
     └── scripts/               # Reproduction and benchmarking CLI scripts
     ```

**Execution Log & Completed Artifacts**:
- **Environment**: Initialized Python 3.13 (Apple Silicon ARM64) virtual environment at `.venv` with `pip install -e ".[dev]"`. Installed `torch` (2.14.1 with MPS/CPU backend), `torch_geometric` (2.8.0), `networkx` (3.7), `scipy` (1.18.1), `z3-solver` (5.1.0), `pydantic` (2.13.5), `pytest` (9.1.1), `mypy` (2.4.0), `ruff` (0.16.10), and `pre-commit` (4.6.2).
- **Tooling**: Created [`pyproject.toml`](file:///Users/duke/IAM/pyproject.toml) (PEP 621, Ruff, Mypy strict mode, Pytest), [`.pre-commit-config.yaml`](file:///Users/duke/IAM/.pre-commit-config.yaml) (hook installed at `.git/hooks/pre-commit`), [`.gitignore`](file:///Users/duke/IAM/.gitignore), and [`README.md`](file:///Users/duke/IAM/README.md).
- **Package Architecture**: Structured namespace under `src/iam/` (`parser`, `generator`, `models`, `optimizer`, `verifier`, `dashboard`) to eliminate stdlib naming conflicts.
- **Validation**: Smoke test suite [`tests/test_environment.py`](file:///Users/duke/IAM/tests/test_environment.py) passed all 7 tests (`torch` MPS/CPU, PyG `HeteroData`, `z3` SMT solver, NetworkX `MultiDiGraph`, `scipy.sparse`, `pydantic`). Ruff and Mypy passed with 0 errors.

---

## Step 1.2: Declarative IAM JSON Schema & AST Parser (5.0h) - [COMPLETED]

**Objective**: Build a parser to ingest raw AWS IAM JSON policies and convert them into an Abstract Syntax Tree (AST) suitable for graph construction and symbolic analysis.

**Implementation Instructions**:
1. **Data Models (Pydantic or Dataclasses)**:
   - [x] Define structures for `Statement`, `Effect`, `Action`/`NotAction`, `Resource`/`NotResource`, and `Condition`.
   - [x] Ensure the parser gracefully handles single strings vs. lists of strings for fields like `Action` and `Resource`.

2. **Wildcard Expansion Resolver**:
   - [x] Implement a utility to expand wildcards (e.g., `iam:*` or `s3:Get*`) into canonical sets of actions based on the capability model (from Step 1.3).
   - [x] Properly handle case-insensitivity in AWS actions.

3. **Explicit Deny Evaluation logic**:
   - [x] Create logic to enforce the AWS evaluation model where an explicit `Deny` overrides any `Allow` across all scopes.

4. **Condition Parsing**:
   - [x] Build support for extracting conditional logic components (e.g., `aws:PrincipalArn`, `aws:MultiFactorAuthPresent`, `iam:PassedToService`).

**Execution Log & Completed Artifacts**:
- **AST Data Models**: Implemented [`src/iam/parser/schema.py`](file:///Users/duke/IAM/src/iam/parser/schema.py) with `Effect`, `Principal`, `Statement`, and `PolicyDocument`. Automatic coercion for string-to-list, dictionary/list statements, serialization/deserialization (`from_dict`, `from_json`, `to_dict`, `to_json`), and validation preventing simultaneous `Action`/`NotAction` or `Resource`/`NotResource`.
- **Wildcard Resolution**: Implemented [`src/iam/parser/wildcard.py`](file:///Users/duke/IAM/src/iam/parser/wildcard.py) supporting case-insensitive glob matching (`action_matches_pattern`, `resource_matches_pattern`) and `WildcardResolver` for action universe expansion and `NotAction` complement calculation.
- **Condition Parsing & Evaluation**: Implemented [`src/iam/parser/conditions.py`](file:///Users/duke/IAM/src/iam/parser/conditions.py) supporting `StringEquals`, `StringLike`, `ArnEquals`, `ArnLike`, `Bool`, `Numeric*`, `Null`, `IfExists` suffixes, and Set modifiers (`ForAnyValue:*`, `ForAllValues:*`).
- **Symbolic Policy Evaluator**: Implemented [`src/iam/parser/evaluator.py`](file:///Users/duke/IAM/src/iam/parser/evaluator.py) with formal AWS authorization semantics (`ExplicitDeny` overriding `Allow`, `ImplicitDeny` default), `AuthRequest`, `EvalResult`, and `filter_authorized_actions`.
- **Public API**: Exported all components from [`src/iam/parser/__init__.py`](file:///Users/duke/IAM/src/iam/parser/__init__.py).
- **Unit Tests**: Implemented [`tests/test_parser.py`](file:///Users/duke/IAM/tests/test_parser.py) with 19 comprehensive unit tests (26 passed across the entire suite). Ruff and Mypy passed with 0 errors.

---

## Step 1.3: AWS Action–Resource–Condition Capability Model $C(a, r, c)$ (4.5h) - [COMPLETED]

**Objective**: Codify valid AWS authorization semantics so the repair engine never synthesizes illegal policies.

**Implementation Instructions**:
1. **Data Ingestion Strategy**:
   - [x] Extract capability data derived from AWS's machine-readable Service Authorization Reference.
   - [x] For the Tier 1 prototype, strictly limit the scope to: **IAM, STS, Lambda, EC2, S3, and KMS**.
   - [x] Store this as a versioned static data layer (e.g., a structured JSON file with a recorded snapshot date) rather than hand-coded Python rules.
   - [x] *Fallback Note*: If programmatic ingestion exceeds ~2 hours, manually transcribe the table for the ~20 actions heavily used in PE motifs using the same JSON schema.

2. **Schema of Capability Record**:
   - [x] For each action, record:
     - Supported resource types (e.g., `iam:PassRole` targets a role; `iam:CreateAccessKey` targets a user).
     - Whether it is a wildcard-only action (e.g., `iam:ListRoles` must use `Resource: "*"`).
     - Applicable condition keys (action-specific, resource-specific, and global keys).

3. **Capability Checker API**:
   - [x] Implement `is_valid_transformation(action, target_resource, condition) -> bool`.
   - [x] This function will validate whether a proposed policy patch is syntactically and semantically permissible by AWS before passing it to the solver.

**Execution Log & Completed Artifacts**:
- **Versioned Capability Data Snapshot**: Created [`data/schemas/capability_snapshot.json`](file:///Users/duke/IAM/data/schemas/capability_snapshot.json) recording snapshot date (`2026-10-07`, version `1.0.0`) covering all 6 required services (IAM, STS, Lambda, EC2, S3, KMS) and global condition keys.
- **Capability Model Engine**: Implemented [`src/iam/parser/capability.py`](file:///Users/duke/IAM/src/iam/parser/capability.py) with `ActionCapability`, `CapabilityModel`, `TransformationValidation`, `load_default_capability_model()`, and `is_valid_transformation(action, target_resource, condition)`.
- **Admissibility Constraints Enforced**:
  - Rejects narrowing `Resource: "*"` to specific ARNs for wildcard-only actions (e.g., `iam:ListRoles`, `iam:GetAccountAuthorizationDetails`, `sts:GetCallerIdentity`).
  - Enforces resource type conformance (e.g., `iam:PassRole` accepts `:role/` ARNs, rejects `:user/` ARNs).
  - Enforces condition key applicability (rejects unlisted action condition keys while permitting global keys like `aws:PrincipalArn` and `aws:PrincipalTag/*`).
- **Unit Tests**: Implemented [`tests/test_capability.py`](file:///Users/duke/IAM/tests/test_capability.py) covering snapshot loading, ARN scoping, wildcard-only rejections, type mismatches, and condition key validation (8 tests passed, 34 total in suite). Ruff and Mypy passed with 0 errors.

---

## Step 1.4: Unit Tests for IAM Parser & Capability Model (3.5h) - [COMPLETED]

**Objective**: Rigorously verify the correctness of the parser and capability checker to ensure solid foundations.

**Implementation Instructions**:
1. **Testing Wildcard Expansion**:
   - [x] Write tests for edge cases, such as `iam:*AccessKey*` resolving accurately to only relevant keys.
   - [x] Verify action case-insensitivity handling.

2. **Testing Explicit Deny**:
   - [x] Create mock policies where an `Allow` is contradicted by a `Deny` (on overlapping resources or conditions) and ensure the resolver suppresses the `Allow` effectively.

3. **Testing the Capability Model API**:
   - [x] **Valid patches**: Scoping `iam:PassRole` to a valid role ARN; scoping `iam:CreateAccessKey` to a valid user ARN.
   - [x] **Invalid patches (Should Reject)**: Scoping a wildcard-only action like `iam:ListRoles` to a specific ARN; injecting an unsupported condition key for a specific action.
   - [x] Use `pytest` for all unit testing, aiming for >85% coverage on the `src/parser/` module.

**Execution Log & Completed Artifacts**:
- **Comprehensive Test Harness**:
  - [`tests/test_parser.py`](file:///Users/duke/IAM/tests/test_parser.py) (19 tests): Schema validation, AST coercion, wildcard pattern matching, explicit Deny suppression, condition blocks, trust policies.
  - [`tests/test_capability.py`](file:///Users/duke/IAM/tests/test_capability.py) (8 tests): Snapshot loading, valid resource scoping, wildcard-only action rejection, resource type mismatch rejection, condition key applicability.
  - [`tests/test_parser_edge_cases.py`](file:///Users/duke/IAM/tests/test_parser_edge_cases.py) (16 tests): Single-character wildcards (`?`), universal `*`, NotPrincipal exclusion, conflicting resources, advanced numeric/string condition operators, EC2 multi-resource types, custom capability model creation.
  - [`tests/test_environment.py`](file:///Users/duke/IAM/tests/test_environment.py) (7 tests): Core dependency smoke tests (PyTorch MPS/CPU, PyG HeteroData, Z3, SciPy, NetworkX, Pydantic).
- **Test Metrics & Coverage**:
  - **Total Tests Passed**: **50 / 50 passed** in 1.23s.
  - **Code Coverage**: Achieved **95% coverage** across `src/iam/parser/` (exceeding the >85% target):
    - `capability.py`: 95%
    - `conditions.py`: 96%
    - `evaluator.py`: 94%
    - `schema.py`: 96%
    - `wildcard.py`: 94%
- **Tooling Verification**: `ruff` (0 lint/formatting errors), `mypy` strict mode (0 errors across all 17 source files).

---

## Phase 1 - Week 1 Milestone Status: [ALL STEPS COMPLETED]
All 4 steps for Week 1 (Step 1.1, Step 1.2, Step 1.3, Step 1.4) have been implemented, tested, and validated. The repository infrastructure, IAM JSON AST parser, and AWS Action Capability Model $C(a,r,c)$ are complete and verified. Ready to proceed to **Week 2: Canonical PE Motifs & Parameterized Enterprise Cloud Generator**.
