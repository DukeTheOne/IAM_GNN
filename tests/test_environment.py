"""Environment smoke test suite verifying core dependencies and hardware backend."""

from typing import Any

import networkx as nx
import scipy.sparse as sp
import torch
import z3
from pydantic import BaseModel
from torch_geometric.data import HeteroData

import iam


def test_package_import() -> None:
    """Verify that the iam package is importable and has a version string."""
    assert hasattr(iam, "__version__")
    assert isinstance(iam.__version__, str)


def test_torch_backend() -> None:
    """Verify PyTorch initialization and device availability."""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert tensor.shape == (3,)
    # On Apple Silicon macOS, MPS (Metal Performance Shaders) might be available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t_device = tensor.to(device)
    assert t_device.device.type in ("mps", "cpu")


def test_pyg_heterodata_initialization() -> None:
    """Verify PyTorch Geometric HeteroData construction for multi-relational graphs."""
    data = HeteroData()
    # Add dummy User and Role nodes
    data["User"].x = torch.zeros((5, 16), dtype=torch.float32)
    data["Role"].x = torch.zeros((3, 16), dtype=torch.float32)

    # Add ('User', 'AssumesRole', 'Role') edge index
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    data["User", "AssumesRole", "Role"].edge_index = edge_index

    assert data.num_nodes == 8
    assert data["User", "AssumesRole", "Role"].edge_index.shape == (2, 3)


def test_networkx_multigraph() -> None:
    """Verify NetworkX MultiDiGraph support for AWS IAM topology modeling."""
    graph = nx.MultiDiGraph()
    graph.add_node("user_alice", type="User", department="SecOps")
    graph.add_node("role_admin", type="Role")
    graph.add_edge("user_alice", "role_admin", relation="AssumesRole")

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge("user_alice", "role_admin")


def test_z3_solver() -> None:
    """Verify Z3 solver for SMT and MaxSAT constraint reasoning."""
    solver = z3.Solver()
    p = z3.Bool("p")
    q = z3.Bool("q")
    solver.add(z3.Or(p, q))
    solver.add(z3.Not(p))

    result = solver.check()
    assert result == z3.sat
    model = solver.model()
    assert bool(model.eval(q)) is True


def test_scipy_sparse() -> None:
    """Verify SciPy sparse matrix operations."""
    adj = sp.csr_matrix([[0, 1], [1, 0]])
    assert adj.nnz == 2
    assert adj.shape == (2, 2)


def test_pydantic_validation() -> None:
    """Verify Pydantic declarative data validation for IAM policy schemas."""

    class PolicyStatement(BaseModel):
        effect: str
        action: list[str]
        resource: list[str]

    raw_data: dict[str, Any] = {
        "effect": "Allow",
        "action": ["iam:PassRole"],
        "resource": ["arn:aws:iam::123456789012:role/DevRole"],
    }
    stmt = PolicyStatement.model_validate(raw_data)
    assert stmt.effect == "Allow"
    assert len(stmt.action) == 1
