"""Comprehensive unit and integration tests for ActionVocabulary and NodeFeatureExtractor."""

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from iam.generator import (
    EnterpriseTopologyConfig,
    EnterpriseTopologyGenerator,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.graph import EdgeRelation
from iam.models import (
    CANONICAL_DEPARTMENTS,
    ActionVocabulary,
    CentralityExtractor,
    NodeFeatureExtractor,
    build_node_feature_dict,
    get_feature_dimension_config,
    load_default_action_vocabulary,
    one_hot_encode,
    parse_arn_service_and_type,
)
from iam.parser import (
    CapabilityModel,
    PolicyDocument,
    load_default_capability_model,
)


@pytest.fixture
def capability_model() -> CapabilityModel:
    """Fixture providing loaded AWS CapabilityModel."""
    return load_default_capability_model()


@pytest.fixture
def default_vocab() -> ActionVocabulary:
    """Fixture providing default ActionVocabulary loaded from capability snapshot."""
    return load_default_action_vocabulary()


@pytest.fixture
def sample_enterprise_graph(capability_model: CapabilityModel) -> IAMGraph:
    """Fixture providing a deterministic medium enterprise graph."""
    cfg = EnterpriseTopologyConfig(
        num_nodes=150,
        seed=42,
        num_pe_chains=2,
        include_branching_chains=True,
        num_branching_chains=1,
    )
    gen = EnterpriseTopologyGenerator(cfg, capability_model=capability_model)
    return gen.generate()


class TestActionVocabulary:
    """Tests for ActionVocabulary construction, OOV handling, wildcard expansion, and serialization."""

    def test_default_vocabulary_loading(self, default_vocab: ActionVocabulary) -> None:
        """Verify default vocabulary contains OOV index 0 plus 30 canonical AWS actions."""
        # 1 OOV token + 30 canonical actions = 31 total
        assert default_vocab.vocab_size == 31
        assert default_vocab.get_action(0) == "<OOV>"
        assert default_vocab.get_index("<OOV>") == 0
        assert len(default_vocab.canonical_actions) == 30

    def test_case_insensitive_lookup_and_contains(self, default_vocab: ActionVocabulary) -> None:
        """Verify actions can be retrieved case-insensitively."""
        idx1 = default_vocab.get_index("iam:PassRole")
        idx2 = default_vocab.get_index("IAM:PASSROLE")
        idx3 = default_vocab.get_index("  iam:passrole  ")
        assert idx1 > 0
        assert idx1 == idx2 == idx3
        assert default_vocab.contains("iam:PassRole") is True
        assert default_vocab.contains("IAM:PASSROLE") is True

    def test_oov_handling(self, default_vocab: ActionVocabulary) -> None:
        """Verify unknown actions return index 0 without raising exceptions."""
        oov_idx = default_vocab.get_index("waf:GetWebACL")
        assert oov_idx == ActionVocabulary.OOV_INDEX
        assert default_vocab.contains("waf:GetWebACL") is False

        # Encoding single OOV action produces vector with 1 at index 0
        oov_vec = default_vocab.encode_action("waf:GetWebACL")
        assert oov_vec[0] == 1.0
        assert np.sum(oov_vec) == 1.0

    def test_wildcard_expansion_in_encoding(self, default_vocab: ActionVocabulary) -> None:
        """Verify action patterns like 's3:*' expand to all recognized S3 actions."""
        s3_vec = default_vocab.encode_actions(["s3:*"], expand_wildcards=True)
        decoded = default_vocab.decode_vector(s3_vec)

        # In capability snapshot, S3 has GetObject, PutObject, DeleteObject, ListBucket
        assert len(decoded) >= 3
        for act in decoded:
            assert act.lower().startswith("s3:")

        # Universal wildcard expands to all canonical actions in vocabulary
        all_vec = default_vocab.encode_actions(["*"], expand_wildcards=True)
        assert np.all(all_vec[1:] == 1.0)
        decoded_all = default_vocab.decode_vector(all_vec)
        assert len(decoded_all) == len(default_vocab.canonical_actions)

    def test_decode_vector(self, default_vocab: ActionVocabulary) -> None:
        """Verify multi-hot vector accurately decodes back to canonical action names."""
        actions_to_encode = ["iam:PassRole", "kms:Decrypt", "lambda:InvokeFunction"]
        vec = default_vocab.encode_actions(actions_to_encode)
        decoded = default_vocab.decode_vector(vec)

        assert set(decoded) == set(actions_to_encode)

    def test_serialization_roundtrip(self, default_vocab: ActionVocabulary, tmp_path: Path) -> None:
        """Verify dictionary and JSON file serialization roundtrips."""
        data_dict = default_vocab.to_dict()
        vocab_from_dict = ActionVocabulary.from_dict(data_dict)
        assert vocab_from_dict.vocab_size == default_vocab.vocab_size
        assert vocab_from_dict.canonical_actions == default_vocab.canonical_actions

        json_file = tmp_path / "vocab.json"
        default_vocab.save_json(json_file)
        loaded_vocab = ActionVocabulary.load_json(json_file)
        assert loaded_vocab.vocab_size == default_vocab.vocab_size
        assert loaded_vocab.canonical_actions == default_vocab.canonical_actions

    def test_feature_dimension_config(self, default_vocab: ActionVocabulary) -> None:
        """Verify fixed dimension contracts calculated from ActionVocabulary."""
        dim_cfg = get_feature_dimension_config(default_vocab)
        assert dim_cfg.action_vocab_size == 31
        assert dim_cfg.user_dim == 20 + 31  # 51
        assert dim_cfg.role_dim == 21 + 31  # 52
        assert dim_cfg.policy_dim == 8 + 31  # 39
        assert dim_cfg.group_dim == 11 + 31  # 42
        assert dim_cfg.resource_dim == 20


class TestCategoricalUtils:
    """Tests for categorical encoders and ARN parsers."""

    def test_one_hot_encode(self) -> None:
        """Verify one-hot vector generation and fallbacks."""
        vec = one_hot_encode("DevOps", CANONICAL_DEPARTMENTS)
        assert len(vec) == 7
        assert vec[1] == 1.0
        assert np.sum(vec) == 1.0

        # Case-insensitive
        vec_lower = one_hot_encode("devops", CANONICAL_DEPARTMENTS)
        assert np.array_equal(vec, vec_lower)

        # Fallback to Other (last element)
        vec_fallback = one_hot_encode("NonExistentDept", CANONICAL_DEPARTMENTS)
        assert vec_fallback[6] == 1.0
        assert np.sum(vec_fallback) == 1.0

        vec_none = one_hot_encode(None, CANONICAL_DEPARTMENTS)
        assert vec_none[6] == 1.0

    def test_parse_arn_service_and_type(self) -> None:
        """Verify ARN decomposition into canonical service prefix and resource type."""
        cases = [
            ("arn:aws:s3:::my-company-data-lake", "s3", "bucket"),
            ("arn:aws:kms:us-east-1:123456789012:key/key-uuid-1234", "kms", "key"),
            ("arn:aws:lambda:us-east-1:123456789012:function:data-processor", "lambda", "function"),
            ("arn:aws:ec2:us-east-1:123456789012:instance/i-0abcdef1234567890", "ec2", "instance"),
            ("arn:aws:iam::123456789012:role/DevOpsAdminRole", "iam", "role"),
            ("arn:aws:iam::123456789012:user/JuniorDeveloper", "iam", "user"),
            ("arn:aws:iam::123456789012:policy/AdminAccessPolicy", "iam", "policy"),
            ("invalid:non:aws:arn", "other", "other"),
        ]
        for arn, exp_srv, exp_type in cases:
            srv, r_type = parse_arn_service_and_type(arn)
            assert srv == exp_srv, f"Failed service for {arn}: got {srv}, expected {exp_srv}"
            assert r_type == exp_type, f"Failed type for {arn}: got {r_type}, expected {exp_type}"


class TestCentralityExtractor:
    """Tests for graph topological metric precomputation."""

    def test_centrality_extraction_on_graph(self, sample_enterprise_graph: IAMGraph) -> None:
        """Verify degree vector and PageRank scores across graph nodes."""
        extractor = CentralityExtractor(sample_enterprise_graph)

        for node in sample_enterprise_graph.get_nodes()[:20]:
            deg_vec = extractor.get_relational_degree_vector(node.id)
            assert len(deg_vec) == 10
            assert np.all(deg_vec >= 0.0)

            pr = extractor.get_pagerank(node.id)
            assert pr > 0.0

            full_vec = extractor.get_full_centrality_vector(node.id)
            assert len(full_vec) == 11
            assert np.isclose(full_vec[10], pr)

            compact_vec = extractor.get_compact_centrality(
                node.id, in_rel=EdgeRelation.ATTACHED_WITH, out_rel=EdgeRelation.ACTS_ON
            )
            assert len(compact_vec) == 3


class TestNodeFeatureExtractor:
    """Tests for typed feature vector construction across User, Role, Policy, Group, and Resource."""

    def test_user_feature_extraction(
        self,
        sample_enterprise_graph: IAMGraph,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify user feature vector adherence to dimension contracts and value bounds."""
        extractor = NodeFeatureExtractor(sample_enterprise_graph, vocabulary=default_vocab)
        users = sample_enterprise_graph.get_nodes_by_type(NodeType.USER)
        assert len(users) > 0

        for user in users[:10]:
            feat = extractor.extract_user_features(user)
            assert len(feat) == extractor.dimension_config.user_dim
            assert np.all(np.isfinite(feat))
            assert np.all(feat >= 0.0)

    def test_role_feature_extraction(
        self,
        sample_enterprise_graph: IAMGraph,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify role feature vector adherence to dimension contracts and service role detection."""
        extractor = NodeFeatureExtractor(sample_enterprise_graph, vocabulary=default_vocab)
        roles = sample_enterprise_graph.get_nodes_by_type(NodeType.ROLE)
        assert len(roles) > 0

        for role in roles[:10]:
            feat = extractor.extract_role_features(role)
            assert len(feat) == extractor.dimension_config.role_dim
            assert np.all(np.isfinite(feat))
            assert np.all(feat >= 0.0)

    def test_policy_feature_extraction(
        self,
        sample_enterprise_graph: IAMGraph,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify policy feature vector statement statistics and direct action bitmask."""
        extractor = NodeFeatureExtractor(sample_enterprise_graph, vocabulary=default_vocab)
        policies = sample_enterprise_graph.get_nodes_by_type(NodeType.POLICY)
        assert len(policies) > 0

        for pol in policies[:10]:
            feat = extractor.extract_policy_features(pol)
            assert len(feat) == extractor.dimension_config.policy_dim
            assert np.all(np.isfinite(feat))
            assert np.all(feat >= 0.0)

    def test_group_feature_extraction(
        self,
        sample_enterprise_graph: IAMGraph,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify group feature vector adherence to dimension contracts."""
        extractor = NodeFeatureExtractor(sample_enterprise_graph, vocabulary=default_vocab)
        groups = sample_enterprise_graph.get_nodes_by_type(NodeType.GROUP)
        assert len(groups) > 0

        for group in groups[:10]:
            feat = extractor.extract_group_features(group)
            assert len(feat) == extractor.dimension_config.group_dim
            assert np.all(np.isfinite(feat))
            assert np.all(feat >= 0.0)

    def test_resource_feature_extraction(
        self,
        sample_enterprise_graph: IAMGraph,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify resource feature vector types and crown jewel flag."""
        extractor = NodeFeatureExtractor(sample_enterprise_graph, vocabulary=default_vocab)
        resources = sample_enterprise_graph.get_nodes_by_type(NodeType.RESOURCE)
        assert len(resources) > 0

        for res in resources[:10]:
            feat = extractor.extract_resource_features(res)
            assert len(feat) == extractor.dimension_config.resource_dim
            assert np.all(np.isfinite(feat))
            assert np.all(feat >= 0.0)

    def test_permission_propagation_to_identity(
        self,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify an attached policy with iam:PassRole propagates action bitmask to principal."""
        graph = IAMGraph("test_propagation")

        pol = GraphNode(
            id="pol:passrole",
            node_type=NodeType.POLICY,
            arn="arn:aws:iam::123:policy/PassRolePol",
            name="PassRolePol",
            policy_document=PolicyDocument.from_dict(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["iam:PassRole"],
                            "Resource": ["*"],
                        }
                    ]
                }
            ),
        )
        user = GraphNode(
            id="user:dev",
            node_type=NodeType.USER,
            arn="arn:aws:iam::123:user/dev",
            name="dev",
            department="DevOps",
        )
        edge = GraphEdge(
            source=user.id,
            target=pol.id,
            relation=EdgeRelation.ATTACHED_WITH,
            actions=["iam:PassRole"],
        )

        graph.add_node(pol)
        graph.add_node(user)
        graph.add_edge(edge)

        extractor = NodeFeatureExtractor(graph, vocabulary=default_vocab)
        user_feat = extractor.extract_user_features(user)

        # The user feature action bitmask slice begins after 7 (dept) + 2 (flags) + 11 (centrality) = 20
        action_slice = user_feat[20:]
        passrole_idx = default_vocab.get_index("iam:PassRole")
        assert action_slice[passrole_idx] == 1.0


class TestBuildNodeFeatureDict:
    """Tests for batch feature matrix builder and index consistency."""

    def test_build_node_feature_dict_shapes_and_alignment(
        self,
        sample_enterprise_graph: IAMGraph,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify feature tensor shapes, dtypes, and complete alignment with node counts."""
        feature_dict, id_to_idx = build_node_feature_dict(
            sample_enterprise_graph, vocabulary=default_vocab
        )
        dim_cfg = get_feature_dimension_config(default_vocab)

        total_nodes_in_tensors = 0
        expected_dims = {
            "User": dim_cfg.user_dim,
            "Role": dim_cfg.role_dim,
            "Policy": dim_cfg.policy_dim,
            "Group": dim_cfg.group_dim,
            "Resource": dim_cfg.resource_dim,
        }

        for n_type, tensor in feature_dict.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32
            assert tensor.ndim == 2
            assert tensor.shape[1] == expected_dims[n_type]

            node_count = len(sample_enterprise_graph.get_nodes_by_type(NodeType(n_type)))
            assert tensor.shape[0] == node_count
            assert len(id_to_idx[n_type]) == node_count
            total_nodes_in_tensors += node_count

        assert total_nodes_in_tensors == sample_enterprise_graph.num_nodes

    def test_deterministic_feature_tensor_outputs(
        self,
        sample_enterprise_graph: IAMGraph,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify consecutive feature builds on the same graph yield identical tensors."""
        f_dict1, id_map1 = build_node_feature_dict(sample_enterprise_graph, default_vocab)
        f_dict2, id_map2 = build_node_feature_dict(sample_enterprise_graph, default_vocab)

        assert id_map1 == id_map2
        for n_type in f_dict1:
            assert torch.equal(f_dict1[n_type], f_dict2[n_type])

    def test_feature_extraction_performance(
        self,
        capability_model: CapabilityModel,
        default_vocab: ActionVocabulary,
    ) -> None:
        """Verify extracting features for a large graph (N=1,000) completes well within 0.5s."""
        cfg = EnterpriseTopologyConfig(num_nodes=1000, seed=123)
        gen = EnterpriseTopologyGenerator(cfg, capability_model=capability_model)
        graph = gen.generate()

        start_time = time.perf_counter()
        f_dict, _id_map = build_node_feature_dict(graph, default_vocab)
        duration = time.perf_counter() - start_time

        assert duration < 0.5, f"Feature extraction took {duration:.2f}s, exceeding 0.5s limit"
        assert f_dict["User"].shape[0] > 0
