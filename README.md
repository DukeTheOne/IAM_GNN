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

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Linting & Type Checking
```bash
ruff check .
ruff format --check .
mypy src tests
```

### 3. Run Test Suite
```bash
pytest --cov=iam
```

### 4. Interactive Live Demo & Manual Verification
Run the interactive CLI demonstration to manually test and inspect the AST parser, explicit Deny resolution, and AWS Action Capability Model $C(a,r,c)$:
```bash
python scripts/demo_parser.py
```
