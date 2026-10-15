import pytest

from iam.dashboard.ego_graph import (
    EgoNetworkExtractor,
    EgoSubgraph,
    PyVisGraphSynthesizer,
    render_ego_canvas_html,
)
from iam.generator.graph import EdgeRelation, GraphEdge, GraphNode, IAMGraph, NodeType
from iam.optimizer.sampler import CandidateEdge


@pytest.fixture
def sample_graph() -> IAMGraph:
    """Create a multi-hop test authorization graph."""
    graph = IAMGraph("test_ego_graph")

    # Nodes
    u1 = GraphNode(
        id="user:alice", name="Alice", node_type=NodeType.USER, arn="arn:aws:iam::123:user/alice"
    )
    r1 = GraphNode(
        id="role:runner", name="Runner", node_type=NodeType.ROLE, arn="arn:aws:iam::123:role/runner"
    )
    r2 = GraphNode(
        id="role:admin", name="Admin", node_type=NodeType.ROLE, arn="arn:aws:iam::123:role/admin"
    )
    p1 = GraphNode(
        id="policy:pass",
        name="PassPolicy",
        node_type=NodeType.POLICY,
        arn="arn:aws:iam::123:policy/pass",
    )
    res1 = GraphNode(
        id="resource:crown-s3",
        name="CrownS3",
        node_type=NodeType.RESOURCE,
        arn="arn:aws:s3:::crown-jewel-bucket",
    )

    for n in (u1, r1, r2, p1, res1):
        graph.add_node(n)

    # Edges
    # alice -> runner (AssumesRole)
    # runner -> admin (PassesTo)
    # admin -> crown-s3 (ActsOn)
    graph.add_edge(
        GraphEdge(source="user:alice", target="role:runner", relation=EdgeRelation.ASSUMES_ROLE)
    )
    graph.add_edge(
        GraphEdge(source="role:runner", target="role:admin", relation=EdgeRelation.PASSES_TO)
    )
    graph.add_edge(
        GraphEdge(source="role:admin", target="resource:crown-s3", relation=EdgeRelation.ACTS_ON)
    )

    return graph


class TestEgoNetworkExtractor:
    """Tests for ego-subgraph extraction and bounding."""

    def test_two_hop_traversal_boundary(self, sample_graph: IAMGraph) -> None:
        extractor = EgoNetworkExtractor(max_nodes=30)
        subgraph = extractor.extract(sample_graph, source_id="user:alice", max_depth=2)

        assert isinstance(subgraph, EgoSubgraph)
        assert subgraph.source_id == "user:alice"
        assert "user:alice" in subgraph.nodes
        assert "role:runner" in subgraph.nodes  # Hop 1
        assert "role:admin" in subgraph.nodes  # Hop 2
        # Hop 3 (resource:crown-s3) should NOT be reachable when max_depth=2
        assert "resource:crown-s3" not in subgraph.nodes

    def test_one_hop_traversal_boundary(self, sample_graph: IAMGraph) -> None:
        extractor = EgoNetworkExtractor(max_nodes=30)
        subgraph = extractor.extract(sample_graph, source_id="user:alice", max_depth=1)

        assert "user:alice" in subgraph.nodes
        assert "role:runner" in subgraph.nodes
        assert "role:admin" not in subgraph.nodes

    def test_max_nodes_capping_safeguard(self, sample_graph: IAMGraph) -> None:
        # Strict ceiling of 2 nodes
        extractor = EgoNetworkExtractor(max_nodes=2)
        subgraph = extractor.extract(sample_graph, source_id="user:alice", max_depth=2)

        assert subgraph.num_nodes <= 2

    def test_latent_candidate_edge_injection(self, sample_graph: IAMGraph) -> None:
        extractor = EgoNetworkExtractor(max_nodes=30)
        candidates = [
            CandidateEdge(
                source="user:alice",
                target="role:admin",
                relation=EdgeRelation.PASSES_TO,
                probability=0.85,
                is_bridge=True,
            )
        ]
        subgraph = extractor.extract(
            sample_graph,
            source_id="user:alice",
            max_depth=2,
            candidate_edges=candidates,
        )

        latent_edges = [e for e in subgraph.edges if e.get("is_latent")]
        assert len(latent_edges) == 1
        assert latent_edges[0]["from"] == "user:alice"
        assert latent_edges[0]["to"] == "role:admin"
        assert latent_edges[0]["probability"] == 0.85
        assert latent_edges[0]["dashes"] is not False

    def test_simulation_severed_edge_marking(self, sample_graph: IAMGraph) -> None:
        extractor = EgoNetworkExtractor(max_nodes=30)
        severed = {("user:alice", "role:runner", "AssumesRole")}

        subgraph = extractor.extract(
            sample_graph,
            source_id="user:alice",
            max_depth=2,
            severed_edges=severed,
        )

        severed_edges = [e for e in subgraph.edges if e.get("is_severed")]
        assert len(severed_edges) == 1
        assert "SEVERED" in severed_edges[0]["label"]

    def test_nonexistent_source_node(self, sample_graph: IAMGraph) -> None:
        extractor = EgoNetworkExtractor(max_nodes=30)
        subgraph = extractor.extract(sample_graph, source_id="user:ghost", max_depth=2)
        assert subgraph.num_nodes == 0


class TestPyVisGraphSynthesizer:
    """Tests for HTML synthesis and physics stabilization options."""

    def test_html_synthesis_structure(self, sample_graph: IAMGraph) -> None:
        extractor = EgoNetworkExtractor(max_nodes=30)
        subgraph = extractor.extract(sample_graph, source_id="user:alice", max_depth=2)

        synthesizer = PyVisGraphSynthesizer(height="500px")
        html_out = synthesizer.synthesize(subgraph)

        assert isinstance(html_out, str)
        assert "forceAtlas2Based" in html_out
        assert "stabilization" in html_out
        assert "user:alice" in html_out
        assert "#mynetwork" in html_out

    def test_render_ego_canvas_helper(self, sample_graph: IAMGraph) -> None:
        html_out = render_ego_canvas_html(
            graph=sample_graph,
            source_id="user:alice",
            height="400px",
        )
        assert isinstance(html_out, str)
        assert len(html_out) > 500
