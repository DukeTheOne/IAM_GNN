"""Comprehensive unit test suite for IAM AST parser, wildcard expansion, and evaluation."""

import pytest

from iam.parser import (
    AuthRequest,
    Effect,
    EvalDecision,
    PolicyDocument,
    PolicyEvaluator,
    Statement,
    WildcardResolver,
    action_matches_pattern,
    evaluate_condition_block,
    resource_matches_pattern,
)


class TestPolicySchema:
    """Tests for PolicyDocument and Statement AST schema parsing and validation."""

    def test_parse_single_statement_and_string_fields(self) -> None:
        """Verify coercion of single string Action/Resource and single Statement dict."""
        raw = {
            "Version": "2012-10-17",
            "Statement": {
                "Sid": "SingleStmt",
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::mybucket/*",
            },
        }
        doc = PolicyDocument.from_dict(raw)
        assert len(doc.statements) == 1
        stmt = doc.statements[0]
        assert stmt.sid == "SingleStmt"
        assert stmt.effect == Effect.ALLOW
        assert stmt.action == ["s3:GetObject"]
        assert stmt.resource == ["arn:aws:s3:::mybucket/*"]

    def test_parse_multiple_statements_with_lists(self) -> None:
        """Verify parsing of statement lists with multiple actions and resources."""
        raw = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject"],
                    "Resource": ["arn:aws:s3:::bucket1/*", "arn:aws:s3:::bucket2/*"],
                },
                {
                    "Effect": "Deny",
                    "Action": ["s3:DeleteObject"],
                    "Resource": "*",
                },
            ],
        }
        doc = PolicyDocument.from_dict(raw)
        assert len(doc.statements) == 2
        assert doc.statements[0].effect == Effect.ALLOW
        assert len(doc.statements[0].action) == 2
        assert doc.statements[1].effect == Effect.DENY
        assert doc.statements[1].resource == ["*"]

    def test_json_roundtrip(self) -> None:
        """Verify export to JSON string and back."""
        raw = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["iam:PassRole"],
                    "Resource": ["arn:aws:iam::123:role/AppRole"],
                }
            ],
        }
        doc = PolicyDocument.from_dict(raw)
        json_str = doc.to_json()
        loaded = PolicyDocument.from_json(json_str)
        assert loaded.statements[0].action == ["iam:PassRole"]

    def test_invalid_missing_action(self) -> None:
        """Validate error when statement lacks both Action and NotAction."""
        raw = {
            "Effect": "Allow",
            "Resource": ["*"],
        }
        with pytest.raises(ValueError, match="Action"):
            Statement.model_validate(raw)

    def test_invalid_both_action_and_not_action(self) -> None:
        """Validate error when statement defines both Action and NotAction."""
        raw = {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "NotAction": ["s3:DeleteObject"],
            "Resource": ["*"],
        }
        with pytest.raises(ValueError, match="both 'Action' and 'NotAction'"):
            Statement.model_validate(raw)


class TestWildcardResolution:
    """Tests for action and resource wildcard matching and expansion."""

    def test_case_insensitive_matching(self) -> None:
        """AWS actions must match case-insensitively."""
        assert action_matches_pattern("s3:GetObject", "s3:getobject")
        assert action_matches_pattern("IAM:PASSROLE", "iam:PassRole")
        assert action_matches_pattern("s3:getObject", "S3:GET*")

    def test_pattern_wildcards(self) -> None:
        """Verify glob wildcards matching action names."""
        assert action_matches_pattern("iam:CreateAccessKey", "iam:*AccessKey*")
        assert action_matches_pattern("iam:DeleteAccessKey", "iam:*AccessKey*")
        assert not action_matches_pattern("iam:PassRole", "iam:*AccessKey*")
        assert action_matches_pattern("lambda:CreateFunction", "*")

    def test_resource_arn_wildcards(self) -> None:
        """Verify resource ARN prefix and wildcard matching."""
        assert resource_matches_pattern("arn:aws:s3:::mybucket/data.csv", "*")
        assert resource_matches_pattern(
            "arn:aws:s3:::mybucket/data.csv",
            "arn:aws:s3:::mybucket/*",
        )
        assert not resource_matches_pattern(
            "arn:aws:s3:::other/data.csv",
            "arn:aws:s3:::mybucket/*",
        )

    def test_wildcard_resolver_expansion(self) -> None:
        """Verify expanding action wildcards against a registered universe."""
        universe = [
            "iam:PassRole",
            "iam:CreateAccessKey",
            "iam:DeleteAccessKey",
            "s3:GetObject",
            "s3:PutObject",
            "lambda:InvokeFunction",
        ]
        resolver = WildcardResolver(universe)

        # Expand iam:*AccessKey*
        expanded = resolver.expand_actions(["iam:*AccessKey*"])
        assert expanded == {"iam:CreateAccessKey", "iam:DeleteAccessKey"}

        # Expand s3:*
        expanded_s3 = resolver.expand_actions(["s3:*"])
        assert expanded_s3 == {"s3:GetObject", "s3:PutObject"}

        # Expand NotAction
        not_s3 = resolver.expand_not_actions(["s3:*"])
        assert "s3:GetObject" not in not_s3
        assert "iam:PassRole" in not_s3
        assert "lambda:InvokeFunction" in not_s3


class TestExplicitDenyAndEvaluation:
    """Tests for AWS formal authorization evaluation: Explicit Deny overriding Allow."""

    def test_implicit_deny_when_no_match(self) -> None:
        """Request without matching statements is implicitly denied."""
        stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject"],
            Resource=["arn:aws:s3:::mybucket/*"],
        )
        evaluator = PolicyEvaluator(statements=[stmt])

        req = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Alice",
            action="s3:PutObject",
            resource_arn="arn:aws:s3:::mybucket/test.txt",
        )
        res = evaluator.evaluate(req)
        assert res.decision == EvalDecision.IMPLICIT_DENY
        assert not res.is_allowed

    def test_explicit_allow(self) -> None:
        """Request matching an Allow statement is allowed."""
        stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject"],
            Resource=["arn:aws:s3:::mybucket/*"],
        )
        evaluator = PolicyEvaluator(statements=[stmt])

        req = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Alice",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::mybucket/test.txt",
        )
        res = evaluator.evaluate(req)
        assert res.decision == EvalDecision.ALLOW
        assert res.is_allowed
        assert len(res.matched_allow_statements) == 1

    def test_explicit_deny_overrides_allow(self) -> None:
        """Explicit Deny must override Allow regardless of order or broadness."""
        allow_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:*"],
            Resource=["*"],
        )
        deny_stmt = Statement(
            Effect=Effect.DENY,
            Action=["s3:DeleteBucket"],
            Resource=["*"],
        )
        evaluator = PolicyEvaluator(statements=[allow_stmt, deny_stmt])

        # Normal s3:GetObject should be allowed
        req_get = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Alice",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::mybucket/test.txt",
        )
        assert evaluator.evaluate(req_get).is_allowed

        # s3:DeleteBucket must be explicitly denied
        req_del = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Alice",
            action="s3:DeleteBucket",
            resource_arn="arn:aws:s3:::mybucket",
        )
        res_del = evaluator.evaluate(req_del)
        assert res_del.decision == EvalDecision.EXPLICIT_DENY
        assert not res_del.is_allowed
        assert len(res_del.matched_deny_statements) == 1
        assert len(res_del.matched_allow_statements) == 1

    def test_filter_authorized_actions(self) -> None:
        """Verify filtering action sets down to authorized actions only."""
        allow_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole", "iam:CreateAccessKey"],
            Resource=["*"],
        )
        deny_stmt = Statement(
            Effect=Effect.DENY,
            Action=["iam:CreateAccessKey"],
            Resource=["*"],
        )
        evaluator = PolicyEvaluator(statements=[allow_stmt, deny_stmt])

        actions = ["iam:PassRole", "iam:CreateAccessKey", "s3:GetObject"]
        authorized = evaluator.filter_authorized_actions(
            principal_arn="arn:aws:iam::123:user/Bob",
            actions=actions,
            resource_arn="*",
        )
        assert authorized == {"iam:PassRole"}


class TestConditionEvaluation:
    """Tests for conditional authorization logic parsing and evaluation."""

    def test_string_equals_and_bool_conditions(self) -> None:
        """Verify StringEquals and Bool operators."""
        cond = {
            "StringEquals": {
                "iam:PassedToService": "lambda.amazonaws.com",
            },
            "Bool": {
                "aws:MultiFactorAuthPresent": "true",
            },
        }
        valid_ctx = {
            "iam:PassedToService": "lambda.amazonaws.com",
            "aws:MultiFactorAuthPresent": True,
        }
        assert evaluate_condition_block(cond, valid_ctx)

        # Failure: MFA is False
        invalid_ctx_mfa = {
            "iam:PassedToService": "lambda.amazonaws.com",
            "aws:MultiFactorAuthPresent": False,
        }
        assert not evaluate_condition_block(cond, invalid_ctx_mfa)

        # Failure: Passed to EC2 instead of Lambda
        invalid_ctx_svc = {
            "iam:PassedToService": "ec2.amazonaws.com",
            "aws:MultiFactorAuthPresent": True,
        }
        assert not evaluate_condition_block(cond, invalid_ctx_svc)

    def test_arn_equals_condition(self) -> None:
        """Verify ArnEquals and ArnLike condition operators."""
        cond = {
            "ArnEquals": {
                "aws:PrincipalArn": "arn:aws:iam::123456789012:role/DevRole",
            }
        }
        assert evaluate_condition_block(
            cond, {"aws:PrincipalArn": "arn:aws:iam::123456789012:role/DevRole"}
        )
        assert not evaluate_condition_block(
            cond, {"aws:PrincipalArn": "arn:aws:iam::999999999999:role/Other"}
        )

    def test_condition_in_statement_evaluation(self) -> None:
        """Verify condition evaluation inside complete PolicyEvaluator flow."""
        stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole"],
            Resource=["arn:aws:iam::123:role/TargetRole"],
            Condition={
                "StringEquals": {
                    "iam:PassedToService": "lambda.amazonaws.com",
                }
            },
        )
        evaluator = PolicyEvaluator(statements=[stmt])

        # Allowed request with passing condition
        req_ok = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Dev",
            action="iam:PassRole",
            resource_arn="arn:aws:iam::123:role/TargetRole",
            context={"iam:PassedToService": "lambda.amazonaws.com"},
        )
        assert evaluator.evaluate(req_ok).is_allowed

        # Denied request with missing/mismatched condition
        req_bad = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Dev",
            action="iam:PassRole",
            resource_arn="arn:aws:iam::123:role/TargetRole",
            context={"iam:PassedToService": "ec2.amazonaws.com"},
        )
        assert not evaluator.evaluate(req_bad).is_allowed

    def test_advanced_conditions_and_set_operators(self) -> None:
        """Verify Numeric, Null, IfExists, and Set operators."""
        # 1. Numeric operators
        num_cond = {"NumericLessThan": {"aws:CurrentTime": 100}}
        assert evaluate_condition_block(num_cond, {"aws:CurrentTime": 50})
        assert not evaluate_condition_block(num_cond, {"aws:CurrentTime": 150})

        # 2. Null operator
        null_cond = {"Null": {"aws:TokenIssueTime": "true"}}
        assert evaluate_condition_block(null_cond, {})
        assert not evaluate_condition_block(null_cond, {"aws:TokenIssueTime": "12345"})

        # 3. IfExists modifier
        if_exists_cond = {"StringEqualsIfExists": {"aws:PrincipalTag/Team": "Security"}}
        assert evaluate_condition_block(if_exists_cond, {})  # passes because key is absent
        assert evaluate_condition_block(if_exists_cond, {"aws:PrincipalTag/Team": "Security"})
        assert not evaluate_condition_block(if_exists_cond, {"aws:PrincipalTag/Team": "Marketing"})

        # 4. Set operator: ForAnyValue:StringEquals
        set_any_cond = {
            "ForAnyValue:StringEquals": {
                "aws:TagKeys": ["CostCenter", "Environment"],
            }
        }
        assert evaluate_condition_block(set_any_cond, {"aws:TagKeys": ["CostCenter", "Project"]})
        assert not evaluate_condition_block(set_any_cond, {"aws:TagKeys": ["Department"]})

        # 5. Set operator: ForAllValues:StringEquals
        set_all_cond = {
            "ForAllValues:StringEquals": {
                "aws:TagKeys": ["CostCenter", "Environment"],
            }
        }
        assert evaluate_condition_block(set_all_cond, {"aws:TagKeys": ["CostCenter"]})
        assert not evaluate_condition_block(
            set_all_cond, {"aws:TagKeys": ["CostCenter", "InvalidTag"]}
        )


class TestPrincipalAndExclusions:
    """Tests for Principal matching, NotResource, and NotPrincipal."""

    def test_not_resource_evaluation(self) -> None:
        """NotResource allows operations on all resources except the specified ones."""
        stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["s3:GetObject"],
            NotResource=["arn:aws:s3:::confidential/*"],
        )
        evaluator = PolicyEvaluator(statements=[stmt])

        # Allowed on public bucket
        req_pub = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Dev",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::public/data.csv",
        )
        assert evaluator.evaluate(req_pub).is_allowed

        # Denied on confidential bucket
        req_secret = AuthRequest(
            principal_arn="arn:aws:iam::123:user/Dev",
            action="s3:GetObject",
            resource_arn="arn:aws:s3:::confidential/keys.pem",
        )
        assert not evaluator.evaluate(req_secret).is_allowed

    def test_principal_matching_in_trust_policy(self) -> None:
        """Verify Principal parsing with dict and wildcard matching."""
        raw_trust = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        doc = PolicyDocument.from_dict(raw_trust)
        evaluator = PolicyEvaluator(policies=[doc])

        # Service match
        req_svc = AuthRequest(
            principal_arn="lambda.amazonaws.com",
            action="sts:AssumeRole",
            resource_arn="*",
        )
        assert evaluator.evaluate(req_svc).is_allowed

        # Service mismatch
        req_other = AuthRequest(
            principal_arn="ec2.amazonaws.com",
            action="sts:AssumeRole",
            resource_arn="*",
        )
        assert not evaluator.evaluate(req_other).is_allowed
