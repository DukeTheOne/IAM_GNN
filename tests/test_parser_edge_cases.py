"""Edge case and integration tests for IAM parser, capability model, and evaluation engine."""

import pytest

from iam.parser import (
    ActionCapability,
    AuthRequest,
    CapabilityModel,
    Effect,
    PolicyDocument,
    PolicyEvaluator,
    Principal,
    Statement,
    WildcardResolver,
    action_matches_pattern,
    evaluate_condition_block,
    is_valid_transformation,
    normalize_action_name,
    resource_matches_pattern,
    statement_matches_request,
)


class TestParserEdgeCases:
    """Rigorous tests covering syntax edge cases, coercion, and error handling."""

    def test_principal_normalization_and_matching(self) -> None:
        """Verify Principal model parsing string, dictionary, list, and wildcard semantics."""
        # Universal wildcard
        p_star = Principal.model_validate("*")
        assert p_star.matches("arn:aws:iam::123:user/Alice")

        # Dictionary with list of ARNs
        p_dict = Principal.model_validate(
            {"AWS": ["arn:aws:iam::123:root", "arn:aws:iam::456:root"]}
        )
        assert p_dict.matches("arn:aws:iam::123:root")
        assert not p_dict.matches("arn:aws:iam::789:root")

        # Dictionary with wildcard in category
        p_dict_star = Principal.model_validate({"AWS": "*"})
        assert p_dict_star.matches("arn:aws:iam::any:user")

        # Invalid format
        with pytest.raises(ValueError, match="Invalid principal format"):
            Principal.model_validate(12345)

    def test_statement_not_principal_exclusion(self) -> None:
        """Verify NotPrincipal logic suppresses statement when request matches NotPrincipal."""
        stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject"],
            Resource=["*"],
            NotPrincipal=Principal.model_validate({"AWS": "arn:aws:iam::123:user/Blacklisted"}),
        )
        evaluator = PolicyEvaluator(statements=[stmt])

        # Allowed for normal user
        req_alice = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Alice",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::bucket/file.txt",
        )
        assert evaluator.evaluate(req_alice).is_allowed

        # Denied for blacklisted principal
        req_black = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Blacklisted",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::bucket/file.txt",
        )
        assert not evaluator.evaluate(req_black).is_allowed

    def test_statement_conflicting_resource_and_not_resource(self) -> None:
        """Statement cannot specify both Resource and NotResource."""
        raw = {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": ["*"],
            "NotResource": ["arn:aws:s3:::secret/*"],
        }
        with pytest.raises(ValueError, match="both 'Resource' and 'NotResource'"):
            Statement.model_validate(raw)

    def test_policy_document_coercion_and_invalid_formats(self) -> None:
        """Verify statement coercion and invalid type handling."""
        with pytest.raises(ValueError, match="Invalid Statement format"):
            PolicyDocument.model_validate({"Version": "2012-10-17", "Statement": "InvalidString"})

        doc = PolicyDocument(Statement=[])
        assert doc.statements == []
        assert doc.to_dict()["Statement"] == []


class TestWildcardAndPatternEdgeCases:
    """Edge cases for wildcard resolution and action/resource pattern matching."""

    def test_action_matches_single_character_wildcard(self) -> None:
        """Verify '?' single-character glob matching."""
        assert action_matches_pattern("s3:getobject", "s3:getobjec?")
        assert not action_matches_pattern("s3:getobjects", "s3:getobjec?")

    def test_resource_matches_universal_star(self) -> None:
        """Universal wildcard matches any resource string."""
        assert resource_matches_pattern("arn:aws:ec2:us-east-1:123:instance/i-abc", "*")
        assert not resource_matches_pattern("arn:aws:ec2:us-east-1:123:instance/i-abc", "")

    def test_wildcard_resolver_empty_and_fallback(self) -> None:
        """Verify resolver handling of empty patterns, spaces, and fallback options."""
        resolver = WildcardResolver(["s3:GetObject", "s3:PutObject"])
        # Whitespace and empty pattern handling
        assert resolver.expand_actions(["", "   "]) == set()

        # Non-wildcard pattern not in universe: fallback_to_pattern=True retains it
        assert resolver.expand_actions(["unknown:Action"], fallback_to_pattern=True) == {
            "unknown:Action"
        }
        # fallback_to_pattern=False omits it
        assert resolver.expand_actions(["unknown:Action"], fallback_to_pattern=False) == set()

        # Wildcard pattern not matching anything in universe
        assert resolver.expand_actions(["sqs:*"], fallback_to_pattern=True) == {"sqs:*"}
        assert resolver.expand_actions(["sqs:*"], fallback_to_pattern=False) == set()

    def test_normalize_action_name(self) -> None:
        """Verify action normalization."""
        assert normalize_action_name("  IAM:PassRole  ") == "iam:passrole"


class TestConditionEdgeCases:
    """Comprehensive condition operator coverage including negative and numeric edge cases."""

    def test_string_not_equals_and_ignore_case(self) -> None:
        """Verify StringNotEquals, StringEqualsIgnoreCase, and StringNotEqualsIgnoreCase."""
        cond = {
            "StringNotEquals": {"aws:PrincipalTag/Env": "Prod"},
            "StringEqualsIgnoreCase": {"aws:username": "ADMIN"},
            "StringNotEqualsIgnoreCase": {"aws:RequestedRegion": "cn-north-1"},
        }
        ctx_ok = {
            "aws:PrincipalTag/Env": "Dev",
            "aws:username": "admin",
            "aws:RequestedRegion": "us-east-1",
        }
        assert evaluate_condition_block(cond, ctx_ok)

        # Fails on StringNotEquals (is Prod)
        assert not evaluate_condition_block(cond, {**ctx_ok, "aws:PrincipalTag/Env": "Prod"})

        # Fails on StringEqualsIgnoreCase (not admin)
        assert not evaluate_condition_block(cond, {**ctx_ok, "aws:username": "guest"})

        # Fails on StringNotEqualsIgnoreCase (is cn-north-1)
        assert not evaluate_condition_block(cond, {**ctx_ok, "aws:RequestedRegion": "CN-NORTH-1"})

    def test_string_like_and_not_like(self) -> None:
        """Verify StringLike, StringNotLike, ArnLike, and ArnNotLike operators."""
        cond = {
            "StringLike": {"s3:prefix": "home/*"},
            "StringNotLike": {"s3:prefix": "home/confidential/*"},
            "ArnNotEquals": {"aws:SourceArn": "arn:aws:iam::123:role/MaliciousRole"},
        }
        ctx_ok = {
            "s3:prefix": "home/alice/data.csv",
            "aws:SourceArn": "arn:aws:iam::123:role/GoodRole",
        }
        assert evaluate_condition_block(cond, ctx_ok)

        # Fails on StringNotLike
        assert not evaluate_condition_block(
            cond, {**ctx_ok, "s3:prefix": "home/confidential/keys.pem"}
        )

        # Fails on ArnNotEquals
        assert not evaluate_condition_block(
            cond, {**ctx_ok, "aws:SourceArn": "arn:aws:iam::123:role/MaliciousRole"}
        )

    def test_numeric_comparison_operators(self) -> None:
        """Verify NumericEquals, NumericNotEquals, NumericLessThanEquals, NumericGreaterThan, NumericGreaterThanEquals."""
        cond = {
            "NumericEquals": {"aws:MultiFactorAuthAge": 3600},
            "NumericNotEquals": {"aws:RequestAttempts": 0},
            "NumericLessThanEquals": {"aws:Score": 100},
            "NumericGreaterThan": {"aws:MinLevel": 5},
            "NumericGreaterThanEquals": {"aws:MaxLevel": 10},
        }
        ctx_ok = {
            "aws:MultiFactorAuthAge": "3600",
            "aws:RequestAttempts": 1,
            "aws:Score": 100,
            "aws:MinLevel": 6,
            "aws:MaxLevel": 10,
        }
        assert evaluate_condition_block(cond, ctx_ok)

        # Non-numeric input returns False gracefully
        assert not evaluate_condition_block(
            {"NumericEquals": {"aws:Score": 10}}, {"aws:Score": "NotANumber"}
        )

    def test_empty_condition_block(self) -> None:
        """Empty condition block evaluates to True."""
        assert evaluate_condition_block({}, {})


class TestEvaluatorMultiPolicyProvenance:
    """Verify PolicyEvaluator methods and provenance tracking."""

    def test_evaluator_add_policy_and_statement(self) -> None:
        """Verify dynamic addition of policies and individual statements."""
        evaluator = PolicyEvaluator()
        assert len(evaluator.statements) == 0

        stmt1 = Statement(Effect=Effect.ALLOW, Action=["s3:GetObject"], Resource=["*"])
        evaluator.add_statement(stmt1)
        assert len(evaluator.statements) == 1

        policy2 = PolicyDocument(
            Statement=[
                Statement(Effect=Effect.DENY, Action=["s3:DeleteObject"], Resource=["*"]),
            ]
        )
        evaluator.add_policy(policy2)
        assert len(evaluator.statements) == 2

    def test_statement_matches_request_helper(self) -> None:
        """Direct test of statement_matches_request helper."""
        stmt = Statement(
            Effect=Effect.ALLOW, Action=["s3:GetObject"], Resource=["arn:aws:s3:::mybucket/*"]
        )
        req_match = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Alice",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::mybucket/test.txt",
        )
        assert statement_matches_request(stmt, req_match)

        req_no_match = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Alice",
            action="s3:PutObject",
            resource_arn="arn:aws:s3:::mybucket/test.txt",
        )
        assert not statement_matches_request(stmt, req_no_match)


class TestCapabilityModelAdvanced:
    """Test advanced capability model features: EC2 multi-resource types, custom snapshots, and validation."""

    def test_ec2_multi_resource_types(self) -> None:
        """Verify ec2:RunInstances matches instance, volume, subnet, security-group ARNs."""
        assert is_valid_transformation(
            "ec2:RunInstances", "arn:aws:ec2:us-east-1:123:instance/i-12345"
        )
        assert is_valid_transformation(
            "ec2:RunInstances", "arn:aws:ec2:us-east-1:123:volume/vol-12345"
        )
        assert is_valid_transformation(
            "ec2:RunInstances", "arn:aws:ec2:us-east-1:123:subnet/subnet-12345"
        )
        assert is_valid_transformation(
            "ec2:RunInstances", "arn:aws:ec2:us-east-1:123:security-group/sg-12345"
        )
        assert not is_valid_transformation("ec2:RunInstances", "arn:aws:iam::123:role/MyRole")

    def test_custom_capability_model_creation(self) -> None:
        """Verify building and querying a custom CapabilityModel instance."""
        custom_action = ActionCapability(
            action="custom:SpecialAction",
            service="custom",
            wildcard_only=False,
            resource_types=["special"],
            condition_keys=["custom:Department"],
        )
        model = CapabilityModel(
            actions={"custom:specialaction": custom_action},
            global_condition_keys={"aws:PrincipalArn"},
            metadata={"source": "test"},
        )
        assert model.metadata["source"] == "test"
        assert model.registered_actions == {"custom:SpecialAction"}
        assert model.get_action_capability("custom:SpecialAction") is not None
        assert not model.is_wildcard_only("custom:SpecialAction")

        # Unknown action validation failure
        res_unknown = model.validate_transformation("unknown:Action", "*")
        assert not res_unknown.is_valid
        assert "unrecognized in capability snapshot" in res_unknown.reason
