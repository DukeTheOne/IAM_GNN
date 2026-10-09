"""Milestone M1 validation and end-to-end data pipeline benchmark.

Validates the full data pipeline from symbolic generation to masked PyG tensors,
executes a PyG neural message-passing dry run with backward gradient flow,
and profiles execution runtime and memory scaling across enterprise graph sizes.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    EnvironmentExporter,
    GroundTruthLabeler,
    IAMGraph,
)
from iam.models import (
    ActionVocabulary,
    IAMHeteroDataConverter,
    MaskingCondition,
    MaskingResult,
    StratifiedNegativeSampler,
    StructuredMaskingSuite,
    is_reverse_edge_type,
    load_default_action_vocabulary,
)
from iam.parser import CapabilityModel, load_default_capability_model


class DummyHeteroMessagePassingGNN(nn.Module):
    """Minimal 1-layer HeteroConv neural module for message-passing smoke testing."""

    def __init__(self, data: HeteroData, hidden_channels: int = 32) -> None:
        super().__init__()
        # 1. Project heterogeneous input feature dimensions to uniform hidden space
        self.lin_dict = nn.ModuleDict()
        for nt in data.node_types:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                in_dim = int(data[nt].x.size(1))
                self.lin_dict[nt] = nn.Linear(in_dim, hidden_channels)

        # 2. Relational message passing across all edge types present in G_o
        conv_dict: dict[tuple[str, str, str], nn.Module] = {}
        for et in data.edge_types:
            if data[et].edge_index.size(1) > 0:
                conv_dict[et] = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv = HeteroConv(conv_dict, aggr="sum")

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Execute feature projection and 1-hop multi-relational aggregation."""
        h_dict: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.lin_dict:
                h_dict[nt] = F.relu(self.lin_dict[nt](x))

        return cast(dict[str, torch.Tensor], self.conv(h_dict, edge_index_dict))


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def default_vocab() -> ActionVocabulary:
    """Fixture providing default ActionVocabulary."""
    return load_default_action_vocabulary()


class TestEndToEndPipelineIntegration:
    """Chains all Milestone M1 components into an automated end-to-end integration test."""

    def test_complete_data_pipeline_chain(
        self, capability_model: CapabilityModel, default_vocab: ActionVocabulary
    ) -> None:
        """Execute: EnterpriseTopologyGenerator -> Exporter -> Converter -> Sampler -> Masking."""
        # 1. Generate parameterized enterprise topology
        topo_cfg = EnterpriseTopologyConfig(
            num_nodes=250,
            seed=101,
            num_pe_chains=3,
            include_branching_chains=True,
            num_branching_chains=1,
            account_id="111122223333",
            organization_id="o-pipeline-test",
        )
        topo_gen = EnterpriseTopologyGenerator(config=topo_cfg, capability_model=capability_model)
        graph = topo_gen.generate()
        assert isinstance(graph, IAMGraph)
        assert graph.num_nodes >= 200

        # 2. Label ground-truth reachability and critical bottleneck bridges
        labeler = GroundTruthLabeler()
        label_set = labeler.label_graph(graph)
        assert len(label_set.pe_pairs) >= 1

        # 3. Export to pre-indexed PyG dictionary and verify format
        exporter = EnvironmentExporter()
        pyg_dict = exporter.to_pyg_ready_dict(graph, label_set=label_set)
        assert "edge_indices" in pyg_dict
        assert "bridge_edge_indices" in pyg_dict

        # 4. Convert IAMGraph to PyG HeteroData with reverse relations and features
        converter = IAMHeteroDataConverter(
            vocabulary=default_vocab,
            include_reverse_edges=True,
            include_edge_attributes=True,
        )
        hetero_data = converter.convert(graph, label_set=label_set)
        assert isinstance(hetero_data, HeteroData)
        hetero_data.validate()

        # Count unmasked bridge edges
        unmasked_bridges = sum(
            int(hetero_data[et].edge_is_bridge.sum().item())
            for et in hetero_data.edge_types
            if not is_reverse_edge_type(et) and hasattr(hetero_data[et], "edge_is_bridge")
        )
        assert unmasked_bridges >= 1

        # 5. Apply Condition E Adversarial Bridge Masking with Stratified Negative Supervision
        sampler = StratifiedNegativeSampler(
            intra_dept_ratio=0.50,
            inter_dept_ratio=0.30,
            privilege_boundary_ratio=0.20,
        )
        masked_res: MaskingResult = StructuredMaskingSuite.apply_condition(
            data=hetero_data,
            condition=MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
            seed=42,
            negative_sampler=sampler,
            negative_ratio=2.0,
        )

        obs_data = masked_res.observed_data
        obs_data.validate()

        # In observed graph, active bridges drop to 0
        observed_bridges = sum(
            int(obs_data[et].edge_is_bridge.sum().item())
            for et in obs_data.edge_types
            if not is_reverse_edge_type(et) and hasattr(obs_data[et], "edge_is_bridge")
        )
        assert observed_bridges == 0
        assert masked_res.total_hidden_edges == unmasked_bridges

        # Supervision tensors are populated
        assert len(masked_res.supervision_by_edge_type) > 0
        for triple, sup in masked_res.supervision_by_edge_type.items():
            assert sup.num_pos == int(masked_res.hidden_edges[triple].size(1))
            assert sup.num_neg > 0
            assert sup.edge_label_index.size(1) == sup.num_pos + sup.num_neg


class TestPyGMessagePassingSmokeTest:
    """Verifies that observed tensors support forward message passing and backward gradient flow."""

    def test_hetero_conv_dry_run_and_gradient_flow(
        self, capability_model: CapabilityModel, default_vocab: ActionVocabulary
    ) -> None:
        """Execute a 1-layer HeteroConv forward and backward pass over masked graph G_o."""
        # 1. Generate and mask environment
        cfg = EnterpriseTopologyConfig(num_nodes=150, seed=42, num_pe_chains=2)
        graph = EnterpriseTopologyGenerator(cfg, capability_model=capability_model).generate()
        labels = GroundTruthLabeler().label_graph(graph)
        converter = IAMHeteroDataConverter(vocabulary=default_vocab)
        hetero_data = converter.convert(graph, label_set=labels)

        sampler = StratifiedNegativeSampler()
        masked_res = StructuredMaskingSuite.apply_condition(
            data=hetero_data,
            condition=MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
            seed=42,
            negative_sampler=sampler,
            negative_ratio=1.0,
        )
        obs_data = masked_res.observed_data

        # 2. Instantiate Dummy HeteroGNN
        hidden_dim = 16
        model = DummyHeteroMessagePassingGNN(obs_data, hidden_channels=hidden_dim)

        # 3. Forward message passing
        x_dict = {
            nt: obs_data[nt].x
            for nt in obs_data.node_types
            if hasattr(obs_data[nt], "x") and obs_data[nt].x is not None
        }
        edge_index_dict = {
            et: obs_data[et].edge_index
            for et in obs_data.edge_types
            if obs_data[et].edge_index.size(1) > 0
        }

        out_dict = model(x_dict, edge_index_dict)
        assert isinstance(out_dict, dict)
        for nt, h in out_dict.items():
            assert h.size(0) == x_dict[nt].size(0)
            assert h.size(1) == hidden_dim

        # 4. Link prediction loss computation on supervision evaluation packet
        target_triple = next(iter(masked_res.supervision_by_edge_type.keys()))
        sup = masked_res.supervision_by_edge_type[target_triple]
        src_type, _rel, dst_type = target_triple

        src_emb = out_dict[src_type][sup.edge_label_index[0]]
        dst_emb = out_dict[dst_type][sup.edge_label_index[1]]

        # Bilinear dot-product logits
        logits = (src_emb * dst_emb).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, sup.edge_label)

        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        # 5. Backward gradient flow
        loss.backward()  # type: ignore[no-untyped-call]

        # Check gradients exist and are finite for model parameters
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                assert torch.isfinite(param.grad).all()
        assert has_grads, "Neural model parameters must receive valid gradients."


class TestScalabilityAndPerformanceProfiling:
    """Profiles runtime and memory consumption of conversion and masking across graph scales."""

    @pytest.mark.parametrize(
        ("num_nodes", "max_time_sec", "max_mem_mb"),
        [
            (500, 0.25, 20.0),
            (1000, 0.50, 40.0),
            (2500, 1.50, 80.0),
        ],
    )
    def test_pipeline_runtime_and_memory_scaling(
        self,
        num_nodes: int,
        max_time_sec: float,
        max_mem_mb: float,
        capability_model: CapabilityModel,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify tensor conversion and structured masking scale within latency and memory budgets."""
        cfg = EnterpriseTopologyConfig(num_nodes=num_nodes, seed=42, num_pe_chains=4)
        graph = EnterpriseTopologyGenerator(cfg, capability_model=capability_model).generate()
        converter = IAMHeteroDataConverter(vocabulary=default_vocab)

        tracemalloc.start()
        start_time = time.perf_counter()

        hetero_data = converter.convert(graph)
        masked_res = StructuredMaskingSuite.apply_condition(
            data=hetero_data,
            condition=MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
            seed=42,
        )

        elapsed = time.perf_counter() - start_time
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem / (1024 * 1024)

        assert elapsed < max_time_sec, (
            f"Scale {num_nodes}: Elapsed {elapsed:.3f}s exceeded {max_time_sec}s."
        )
        assert peak_mem_mb < max_mem_mb, (
            f"Scale {num_nodes}: Peak RAM {peak_mem_mb:.2f}MB exceeded {max_mem_mb}MB."
        )
        assert masked_res.observed_data.validate() is True
