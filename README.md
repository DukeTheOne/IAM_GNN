# Uncertainty-Aware Least-Privilege Repair for Cloud IAM under Partial Observability

An enterprise cloud security research framework for inferring latent authorization relationships with calibrated uncertainty and synthesizing capability-compliant least-privilege IAM policy repairs under partial observability.

## Repository Layout

```text
iam-graph-learning/
├── data/                  # Synthetic schemas, generators, and split datasets
│   ├── schemas/           # AST JSON schemas and capability snapshots
│   └── splits/            # Environment-level inductive dataset splits
├── src/iam/               # Core library implementation
│   ├── parser/            # IAM AST, AWS action capability model C(a,r,c)
│   ├── generator/         # Parameterized enterprise generator & PE motifs
│   ├── models/            # PyG RGCN, calibration module, and baselines
│   ├── optimizer/         # Candidate pruning, min-cut, and Z3 MaxSAT repair
│   ├── verifier/          # Symbolic reachability checker & log replay
│   └── dashboard/         # Streamlit and PyVis UI
├── tests/                 # Unit, integration, and benchmark tests
└── scripts/               # Reproduction and benchmarking CLI scripts
```

## Quickstart

### 1. Setup & Environment

The repository uses `uv` for dependency management and deterministic locking (`uv.lock`).

**Option A: Using `uv` (Recommended on Windows & Linux)**
`uv` automatically executes commands inside the project `.venv` without needing manual shell activation:
```bash
# Sync all dependencies and optional dev dependencies
uv sync --all-extras

# List installed packages in the project environment
uv pip list
```

**Option B: Manual Virtual Environment Activation**
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

---

### 2. Interactive Demonstrations & Verification

Run the interactive CLI demonstration to test the IAM AST parser, explicit Deny resolution, and Action Capability Model:
```bash
uv run python scripts/demo_parser.py
# Or if environment is activated: python scripts/demo_parser.py
```

Run the end-to-end synthetic enterprise topology generator, PyG `HeteroData` conversion, Condition E adversarial bridge masking, and neural message-passing demo:
```bash
uv run python scripts/demo_pipeline.py
# Or if environment is activated: python scripts/demo_pipeline.py
```

---

### 3. Run Test Suite

```bash
# Full test suite with coverage
uv run pytest --cov=iam

# Specific test suite
uv run pytest tests/test_pipeline_m1.py -v
```

---

### 4. Code Quality & Type Checking

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

## Development & Git Workflow

Pre-commit hooks are configured (`.pre-commit-config.yaml`) to enforce strict formatting, linting, and type checking.

### 1. Hook Installation (One-Time)
```bash
pre-commit install
```

### 2. Recommended Workflow (Before Commit & Push)
To prevent commits from being rejected by hook auto-formatting or type errors:

```bash
# 1. Format and fix linting
ruff format .
ruff check --fix .

# 2. Run type checks and test suite
mypy src tests
pytest

# 3. Stage changes
git add <files>   # or git add -u

# 4. (Optional) Run hooks against staged changes
pre-commit run

# 5. Commit and push
git commit -m "feat: concise description of changes"
git push origin <branch>
```

> **Tip:** If `git commit` is blocked because `ruff-format` adjusted file formatting during the commit attempt, simply re-stage the reformatted files (`git add -u`) and rerun `git commit`.
