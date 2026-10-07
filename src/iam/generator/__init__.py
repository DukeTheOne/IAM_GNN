"""Parameterized enterprise cloud generator and canonical privilege escalation motifs."""

from iam.generator.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    IAMGraph,
    NodeType,
)
from iam.generator.motifs import (
    AssumeRoleChainMotif,
    AttachPolicyMotif,
    CreateAccessKeyMotif,
    GroundTruthVerification,
    MotifInstance,
    MotifRegistry,
    PassRoleEC2Motif,
    PassRoleLambdaMotif,
    PEMotif,
    SetDefaultPolicyVersionMotif,
    default_motif_registry,
)

__all__ = [
    "AssumeRoleChainMotif",
    "AttachPolicyMotif",
    "CreateAccessKeyMotif",
    "EdgeRelation",
    "GraphEdge",
    "GraphNode",
    "GroundTruthVerification",
    "IAMGraph",
    "MotifInstance",
    "MotifRegistry",
    "NodeType",
    "PEMotif",
    "PassRoleEC2Motif",
    "PassRoleLambdaMotif",
    "SetDefaultPolicyVersionMotif",
    "default_motif_registry",
]
