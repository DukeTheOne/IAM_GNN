"""Unit and integration tests for Step 3.4: Formal Structured Masking Framework.

Validates partial cloud observability operators:
- Condition A: Random edge deletion (P0 baseline)
- Condition E: Adversarial bridge bottleneck masking (P0 core PE evaluation)
- Condition B: Cross-account silo boundary masking (P1 multi-account)
- Condition C: Federated IdP entrypoint masking (P2 SaaS/IdP)
- Condition D: Ephemeral STS session delegation masking (P2 dynamic credentials)
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    GroundTruthLabeler,
    IAMGraph,
)
from iam.models import (
    ActionVocabulary,
    AdversarialBridgeMaskingOperator,
    CrossAccountMaskingOperator,
    EphemeralSTSMaskingOperator,
    FederatedIdPMaskingOperator,
    IAMHeteroDataConverter,
    MaskingCondition,
    MaskingResult,
    RandomMaskingOperator,
    StratifiedNegativeSampler,
    StructuredMaskingSuite,
    get_reverse_edge_type,
    is_reverse_edge_type,
    load_default_action_vocabulary,
)
from iam.parser import CapabilityModel, load_default_capability_model


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def default_vocab() -> ActionVocabulary:
    """Fixture providing default ActionVocabulary."""
    return load_default_action_vocabulary()


@pytest.fixture
def enterprise_graph_and_hetero(
    capability_model: CapabilityModel, default_vocab: ActionVocabulary
) -> tuple[IAMGraph, HeteroData, Any]:
    """Fixture generating a medium enterprise graph, labels, and PyG HeteroData."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=180,
        seed=42,
        num_pe_chains=3,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    topo_gen = EnterpriseTopologyGenerator(config=cfg, capability_model=capability_model)
    graph = topo_gen.generate()
    labeler = GroundTruthLabeler()
    labels = labeler.label_graph(graph)

    converter = IAMHeteroDataConverter(vocabulary=default_vocab)
    hetero = converter.convert(graph, label_set=labels)
    return graph, hetero, labels


class TestStructuredMaskingInvariants:
    """Tests ensuring fundamental mathematical invariants across all 5 masking conditions."""

    @pytest.mark.parametrize(
        "condition",
        [
            MaskingCondition.CONDITION_A_RANDOM,
            MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
            MaskingCondition.CONDITION_B_CROSS_ACCOUNT,
            MaskingCondition.CONDITION_C_FEDERATED_IDP,
            MaskingCondition.CONDITION_D_EPHEMERAL_STS,
        ],
    )
    def test_edge_accounting_invariance(
        self,
        condition: MaskingCondition,
        enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any],
    ) -> None:
        """Verify: |E_observed| + |E_hidden| == |E_unmasked| for all operators."""
        _, hetero, _ = enterprise_graph_and_hetero

        unmasked_forward_edges = sum(
            int(hetero[et].edge_index.size(1))
            for et in hetero.edge_types
            if not is_reverse_edge_type(et)
        )

        result: MaskingResult = StructuredMaskingSuite.apply_condition(
            data=hetero,
            condition=condition,
            seed=42,
        )

        assert result.total_observed_edges + result.total_hidden_edges == unmasked_forward_edges

    @pytest.mark.parametrize(
        "condition",
        [
            MaskingCondition.CONDITION_A_RANDOM,
            MaskingCondition.CONDITION_E_ADVERSARIAL_BRIDGE,
            MaskingCondition.CONDITION_B_CROSS_ACCOUNT,
            MaskingCondition.CONDITION_C_FEDERATED_IDP,
            MaskingCondition.CONDITION_D_EPHEMERAL_STS,
        ],
    )
    def test_reverse_edge_symmetry_and_message_passing_consistency(
        self,
        condition: MaskingCondition,
        enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any],
    ) -> None:
        """Verify that reverse edges are synchronized: |E_obs_fwd| == |E_obs_rev| with 0 bridge flags."""
        _, hetero, _ = enterprise_graph_and_hetero

        result: MaskingResult = StructuredMaskingSuite.apply_condition(
            data=hetero,
            condition=condition,
            seed=42,
        )
        obs_data = result.observed_data

        for et in obs_data.edge_types:
            if not is_reverse_edge_type(et):
                rev_et = get_reverse_edge_type(et)
                if rev_et in obs_data.edge_types:
                    n_fwd = obs_data[et].edge_index.size(1)
                    n_rev = obs_data[rev_et].edge_index.size(1)
                    assert n_fwd == n_rev, (
                        f"Asymmetry between {et} ({n_fwd}) and {rev_et} ({n_rev})"
                    )
                    if hasattr(obs_data[rev_et], "edge_is_bridge"):
                        assert not obs_data[rev_et].edge_is_bridge.any(), (
                            f"Reverse relation {rev_et} must never carry active bridge flags."
                        )


class TestConditionARandomMasking:
    """Tests for Condition A: Uniform random edge deletion."""

    def test_random_masking_ratio_and_protection(
        self, enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any]
    ) -> None:
        """Verify random edge excision approximates mask_ratio and preserves MemberOf."""
        _, hetero, _ = enterprise_graph_and_hetero
        operator = RandomMaskingOperator(mask_ratio=0.30, protect_member_of=True)

        res = operator.apply(hetero, seed=42)

        # 1. MemberOf relations are completely protected
        member_of_type = ("User", "MemberOf", "Group")
        if member_of_type in hetero.edge_types:
            orig_member_count = hetero[member_of_type].edge_index.size(1)
            obs_member_count = res.observed_data[member_of_type].edge_index.size(1)
            assert obs_member_count == orig_member_count

        # 2. Total hidden edges is positive and reasonable
        assert res.total_hidden_edges > 0
        assert res.total_observed_edges > 0

        # 3. Node features remain intact and unchanged
        for nt in hetero.node_types:
            assert torch.equal(hetero[nt].x, res.observed_data[nt].x)

    def test_random_masking_reproducibility(
        self, enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any]
    ) -> None:
        """Verify identical seed produces identical masks."""
        _, hetero, _ = enterprise_graph_and_hetero
        operator = RandomMaskingOperator(mask_ratio=0.25)

        res1 = operator.apply(hetero, seed=99)
        res2 = operator.apply(hetero, seed=99)
        res3 = operator.apply(hetero, seed=123)

        assert res1.total_hidden_edges == res2.total_hidden_edges
        for triple in res1.hidden_edges:
            assert torch.equal(res1.hidden_edges[triple], res2.hidden_edges[triple])

        assert res1.total_hidden_edges != res3.total_hidden_edges or not torch.equal(
            next(iter(res1.hidden_edges.values())), next(iter(res3.hidden_edges.values()))
        )


class TestConditionEAdversarialBridgeMasking:
    """Tests for Condition E: Critical privilege escalation bottleneck masking."""

    def test_adversarial_bridge_masking_severs_bridges(
        self, enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any]
    ) -> None:
        """Verify Condition E excises 100% of injected bridge relations."""
        graph, hetero, labels = enterprise_graph_and_hetero

        initial_bridge_count = 0
        for et in hetero.edge_types:
            if not is_reverse_edge_type(et) and hasattr(hetero[et], "edge_is_bridge"):
                initial_bridge_count += int(hetero[et].edge_is_bridge.sum().item())

        assert initial_bridge_count > 0, "Graph must have injected PE bridges."

        operator = AdversarialBridgeMaskingOperator(bridge_mask_ratio=1.0)
        res = operator.apply(hetero, seed=42)

        # 1. In observed graph, active bridge count drops to zero
        observed_bridges = 0
        for et in res.observed_data.edge_types:
            if not is_reverse_edge_type(et) and hasattr(res.observed_data[et], "edge_is_bridge"):
                observed_bridges += int(res.observed_data[et].edge_is_bridge.sum().item())
        assert observed_bridges == 0

        # 2. Hidden edges match initial bridge count
        assert res.total_hidden_edges == initial_bridge_count

        # 3. Formal verification: BFS reachability drops to 0 across severed PE chains
        if labels.pe_pairs:
            pe_target = labels.pe_pairs[0]
            reachability_broken = StructuredMaskingSuite.verify_reachability_severed(
                original_graph=graph,
                masked_result=res,
                source_id=pe_target.source_id,
                target_id=pe_target.target_id,
            )
            assert reachability_broken is True

    def test_stochastic_bridge_masking(
        self, enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any]
    ) -> None:
        """Verify partial bridge masking with bridge_mask_ratio < 1.0."""
        _, hetero, _ = enterprise_graph_and_hetero
        operator = AdversarialBridgeMaskingOperator(bridge_mask_ratio=0.50)

        res = operator.apply(hetero, seed=42)
        assert 0 < res.total_hidden_edges < res.provenance["total_bridges_masked"] * 2


class TestConditionBCrossAccountMasking:
    """Tests for Condition B: Cross-account boundary and external silo masking."""

    def test_cross_account_masking_with_target_account(self) -> None:
        """Verify masking relations connecting to a designated secondary AWS account."""
        data = HeteroData()
        data.account_id = "111111111111"

        data["User"].num_nodes = 2
        data["User"].arns = [
            "arn:aws:iam::111111111111:user/primary-user",
            "arn:aws:iam::222222222222:user/partner-user",
        ]

        data["Role"].num_nodes = 2
        data["Role"].arns = [
            "arn:aws:iam::111111111111:role/primary-role",
            "arn:aws:iam::222222222222:role/partner-role",
        ]

        # Edge 0: internal (1111 -> 1111)
        # Edge 1: cross-account trust (1111 -> 2222)
        data["User", "AssumesRole", "Role"].edge_index = torch.tensor(
            [[0, 0], [0, 1]], dtype=torch.long
        )

        operator = CrossAccountMaskingOperator(target_account_id="222222222222")
        res = operator.apply(data, seed=42)

        # Edge 1 should be hidden, Edge 0 should remain observed
        assert res.total_hidden_edges == 1
        assert res.total_observed_edges == 1
        obs_edge = res.observed_data["User", "AssumesRole", "Role"].edge_index
        assert torch.equal(obs_edge, torch.tensor([[0], [0]], dtype=torch.long))


class TestConditionCFederatedIdPMasking:
    """Tests for Condition C: External IdP entrypoint binding masking."""

    def test_federated_idp_masking(self) -> None:
        """Verify masking of User -> Role and User -> Group entrypoints while retaining roles."""
        data = HeteroData()
        data["User"].num_nodes = 4
        data["Role"].num_nodes = 4
        data["Group"].num_nodes = 2
        data["Policy"].num_nodes = 2

        data["User", "AssumesRole", "Role"].edge_index = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
        )
        data["Role", "AttachedWith", "Policy"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )

        operator = FederatedIdPMaskingOperator()
        res = operator.apply(data, seed=42)

        # Internal policy attachments must remain untouched
        assert res.observed_data["Role", "AttachedWith", "Policy"].edge_index.size(1) == 2
        # Entrypoint edges should be partially hidden
        assert res.total_hidden_edges > 0


class TestConditionDEphemeralSTSMasking:
    """Tests for Condition D: Runtime ephemeral STS session delegation masking."""

    def test_ephemeral_sts_role_chaining_masking(self) -> None:
        """Verify dynamic Role -> Role chaining is hidden while static attachments remain."""
        data = HeteroData()
        data["Role"].num_nodes = 4
        data["Policy"].num_nodes = 2

        # Role -> Role dynamic chaining
        data["Role", "AssumesRole", "Role"].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long
        )
        # Static attachment
        data["Role", "AttachedWith", "Policy"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )

        operator = EphemeralSTSMaskingOperator(mask_role_chaining=True)
        res = operator.apply(data, seed=42)

        # Ephemeral role chaining must be excised
        assert res.observed_data["Role", "AssumesRole", "Role"].edge_index.size(1) == 0
        assert res.total_hidden_edges == 2
        # Static attachments must be untouched
        assert res.observed_data["Role", "AttachedWith", "Policy"].edge_index.size(1) == 2


class TestMaskingWithNegativeSupervisionAndSymbolicGraph:
    """Tests for supervision tensor packaging and symbolic IAMGraph masking."""

    def test_masking_supervision_packet_generation(
        self, enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any]
    ) -> None:
        """Verify that passing negative_sampler populates supervision_by_edge_type."""
        _, hetero, _ = enterprise_graph_and_hetero
        sampler = StratifiedNegativeSampler()
        operator = AdversarialBridgeMaskingOperator()

        res = operator.apply(hetero, seed=42, negative_sampler=sampler, negative_ratio=1.0)

        assert len(res.supervision_by_edge_type) > 0
        for triple, sup in res.supervision_by_edge_type.items():
            assert sup.num_pos == int(res.hidden_edges[triple].size(1))
            assert sup.num_neg > 0
            assert sup.edge_label_index.size(1) == sup.num_pos + sup.num_neg
            assert (sup.edge_label[: sup.num_pos] == 1.0).all()
            assert (sup.edge_label[sup.num_pos :] == 0.0).all()

    def test_symbolic_graph_masking(
        self, enterprise_graph_and_hetero: tuple[IAMGraph, HeteroData, Any]
    ) -> None:
        """Verify apply_to_graph on IAMGraph."""
        graph, _, _ = enterprise_graph_and_hetero
        operator = AdversarialBridgeMaskingOperator()

        masked_graph, hidden_edges = operator.apply_to_graph(graph, seed=42)

        assert isinstance(masked_graph, IAMGraph)
        assert masked_graph.num_nodes == graph.num_nodes
        assert masked_graph.num_edges + len(hidden_edges) == graph.num_edges
        for edge in hidden_edges:
            assert edge.is_bridge is True
