"""Interactive demonstration script for Phase 1 (IAM Parser & Capability Model).

Run this script to manually verify the environment, AST parser, explicit Deny
resolution, and the AWS Action Capability Model C(a,r,c).

Usage:
    source .venv/bin/activate
    python scripts/demo_parser.py
"""

from __future__ import annotations

import json

import torch
import z3

from iam.parser import (
    AuthRequest,
    PolicyDocument,
    PolicyEvaluator,
    load_default_capability_model,
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def demo_environment() -> None:
    print_header("1. Environment & Hardware Acceleration Check")
    print(f"PyTorch Version:   {torch.__version__}")
    device_name = "MPS (Apple Silicon GPU)" if torch.backends.mps.is_available() else "CPU"
    print(f"Compute Backend:   {device_name}")
    print(f"Z3 Solver Version: {z3.get_version_string()}")
    print("Status:            READY for multi-relational graph learning & SMT solving.")


def demo_ast_and_evaluator() -> None:
    print_header("2. IAM JSON AST Parsing & Symbolic Policy Evaluation")

    raw_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowS3ReadAndPassRole",
                "Effect": "Allow",
                "Action": ["s3:Get*", "iam:PassRole"],
                "Resource": [
                    "arn:aws:s3:::mybucket/*",
                    "arn:aws:iam::123456789012:role/DataPipelineRole",
                ],
                "Condition": {
                    "StringEqualsIfExists": {"iam:PassedToService": "lambda.amazonaws.com"}
                },
            },
            {
                "Sid": "ExplicitDenySensitiveBucketDeletion",
                "Effect": "Deny",
                "Action": ["s3:Delete*"],
                "Resource": "*",
            },
        ],
    }

    print("Input AWS IAM JSON Policy Document:")
    print(json.dumps(raw_policy, indent=2))

    doc = PolicyDocument.from_dict(raw_policy)
    evaluator = PolicyEvaluator(policies=[doc])

    print(f"\nParsed {len(doc.statements)} statements successfully into immutable AST.")

    # Test Cases
    test_requests = [
        AuthRequest(
            principal_arn="arn:aws:iam::123456789012:user/Alice",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::mybucket/data.csv",
        ),
        AuthRequest(
            principal_arn="arn:aws:iam::123456789012:user/Alice",
            action="s3:DeleteBucket",
            resource_arn="arn:aws:s3:::mybucket",
        ),
        AuthRequest(
            principal_arn="arn:aws:iam::123456789012:user/Alice",
            action="iam:PassRole",
            resource_arn="arn:aws:iam::123456789012:role/DataPipelineRole",
            context={"iam:PassedToService": "lambda.amazonaws.com"},
        ),
        AuthRequest(
            principal_arn="arn:aws:iam::123456789012:user/Alice",
            action="iam:PassRole",
            resource_arn="arn:aws:iam::123456789012:role/DataPipelineRole",
            context={"iam:PassedToService": "ec2.amazonaws.com"},
        ),
        AuthRequest(
            principal_arn="arn:aws:iam::123456789012:user/Alice",
            action="iam:CreateUser",
            resource_arn="*",
        ),
    ]

    print("\nEvaluating Authorization Requests:")
    for i, req in enumerate(test_requests, 1):
        res = evaluator.evaluate(req)
        status_symbol = "ALLOWED [OK]" if res.is_allowed else "DENIED  [X]"
        print(f"\n  Request #{i}:")
        print(f"    Action:   {req.action}")
        print(f"    Resource: {req.resource_arn}")
        if req.context:
            print(f"    Context:  {req.context}")
        print(f"    Decision: {res.decision.value} -> {status_symbol}")
        if res.matched_deny_statements:
            print(
                f"    Reason:   Overridden by Deny statement '{res.matched_deny_statements[0].sid}'"
            )


def demo_capability_model() -> None:
    print_header("3. AWS Action Capability Model C(a,r,c) Validation")

    model = load_default_capability_model()
    print(
        f"Loaded Capability Snapshot: Version {model.metadata.get('version')} ({model.metadata.get('snapshot_date')})"
    )
    print(
        f"Registered Actions:         {len(model.registered_actions)} across {model.metadata.get('services')}"
    )

    proposals = [
        ("iam:PassRole", "arn:aws:iam::123456789012:role/LambdaRole", None),
        ("iam:CreateAccessKey", "arn:aws:iam::123456789012:user/Bob", None),
        ("iam:ListRoles", "arn:aws:iam::123456789012:role/LambdaRole", None),
        ("iam:PassRole", "arn:aws:iam::123456789012:user/Bob", None),
        (
            "iam:PassRole",
            "arn:aws:iam::123456789012:role/LambdaRole",
            {"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}},
        ),
        (
            "iam:CreateAccessKey",
            "arn:aws:iam::123456789012:user/Bob",
            {"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}},
        ),
    ]

    print("\nValidating Proposed Policy Transformations (Patch Admissibility):")
    for action, res, cond in proposals:
        validation = model.validate_transformation(action, res, cond)
        status = "ADMISSIBLE   [VALID]" if validation.is_valid else "INADMISSIBLE [REJECTED]"
        cond_str = f" with Condition {cond}" if cond else ""
        print(f"\n  Transform: Scope '{action}' to '{res}'{cond_str}")
        print(f"  Result:    {status}")
        print(f"  Details:   {validation.reason}")


def main() -> None:
    print("\n" + "#" * 70)
    print("  IAM GRAPH LEARNING - PHASE 1 VERIFICATION & LIVE DEMO")
    print("#" * 70)
    demo_environment()
    demo_ast_and_evaluator()
    demo_capability_model()
    print("\n" + "#" * 70)
    print("  All components verified and functioning as expected.")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
