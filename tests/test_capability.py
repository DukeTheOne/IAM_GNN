"""Unit tests for AWS Action-Resource-Condition Capability Model C(a, r, c)."""

from iam.parser import (
    Effect,
    Statement,
    is_valid_transformation,
    load_default_capability_model,
)


class TestCapabilityModel:
    """Test suite verifying capability checking against the AWS capability snapshot."""

    def test_load_default_capability_model(self) -> None:
        """Verify default capability model snapshot loads correctly with metadata and services."""
        model = load_default_capability_model()
        meta = model.metadata
        assert meta["version"] == "1.0.0"
        assert meta["snapshot_date"] == "2026-10-07"
        assert "iam" in meta["services"]
        assert len(model.registered_actions) >= 20

    def test_valid_resource_scoping(self) -> None:
        """Verify valid narrowing of Resource: '*' to specific permitted resource ARNs."""
        # iam:PassRole targets roles
        assert is_valid_transformation(
            action="iam:PassRole",
            target_resource="arn:aws:iam::123456789012:role/DevRole",
        )

        # iam:CreateAccessKey targets users
        assert is_valid_transformation(
            action="iam:CreateAccessKey",
            target_resource="arn:aws:iam::123456789012:user/Alice",
        )

        # s3:GetObject targets objects
        assert is_valid_transformation(
            action="s3:GetObject",
            target_resource="arn:aws:s3:::mybucket/data.csv",
        )

        # lambda:InvokeFunction targets functions
        assert is_valid_transformation(
            action="lambda:InvokeFunction",
            target_resource="arn:aws:lambda:us-east-1:123:function:data-processor",
        )

    def test_wildcard_resource_is_always_accepted(self) -> None:
        """Resource: '*' should be acceptable for any recognized action."""
        assert is_valid_transformation("iam:PassRole", "*")
        assert is_valid_transformation("iam:ListRoles", "*")
        assert is_valid_transformation("s3:GetObject", "*")

    def test_wildcard_only_actions_reject_arn_scoping(self) -> None:
        """Wildcard-only actions in AWS must be rejected if scoped to a specific resource ARN."""
        # iam:ListRoles does not support resource-level permissions
        res_list = is_valid_transformation(
            action="iam:ListRoles",
            target_resource="arn:aws:iam::123456789012:role/DevRole",
        )
        assert not res_list

        # iam:GetAccountAuthorizationDetails does not support resource scoping
        res_auth = is_valid_transformation(
            action="iam:GetAccountAuthorizationDetails",
            target_resource="arn:aws:iam::123456789012:role/DevRole",
        )
        assert not res_auth

        # sts:GetCallerIdentity is wildcard-only
        res_caller = is_valid_transformation(
            action="sts:GetCallerIdentity",
            target_resource="arn:aws:iam::123456789012:user/Alice",
        )
        assert not res_caller

    def test_resource_type_mismatch_rejection(self) -> None:
        """Rejects scoping an action to a mismatched resource ARN type."""
        model = load_default_capability_model()

        # iam:PassRole expects role ARN, not user ARN
        val_pass_user = model.validate_transformation(
            action="iam:PassRole",
            target_resource="arn:aws:iam::123456789012:user/Alice",
        )
        assert not val_pass_user.is_valid
        assert "does not match supported resource types" in val_pass_user.reason

        # iam:CreateAccessKey expects user ARN, not role ARN
        val_key_role = model.validate_transformation(
            action="iam:CreateAccessKey",
            target_resource="arn:aws:iam::123456789012:role/DevRole",
        )
        assert not val_key_role.is_valid

    def test_valid_condition_key_injection(self) -> None:
        """Accepts valid condition keys supported by the action or global keys."""
        # Action-specific condition key for iam:PassRole
        assert is_valid_transformation(
            action="iam:PassRole",
            target_resource="arn:aws:iam::123:role/DevRole",
            condition={"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}},
        )

        # Global condition key: aws:PrincipalArn
        assert is_valid_transformation(
            action="iam:PassRole",
            target_resource="arn:aws:iam::123:role/DevRole",
            condition={"ArnEquals": {"aws:PrincipalArn": "arn:aws:iam::123:role/SecRole"}},
        )

        # Global condition key with prefix wildcard: aws:PrincipalTag/*
        assert is_valid_transformation(
            action="s3:GetObject",
            target_resource="arn:aws:s3:::mybucket/data.csv",
            condition={"StringEquals": {"aws:PrincipalTag/Team": "Security"}},
        )

    def test_invalid_condition_key_rejection(self) -> None:
        """Rejects condition keys not supported by the action and not present in global keys."""
        model = load_default_capability_model()

        # iam:PassedToService is invalid for iam:CreateAccessKey
        val_bad_key = model.validate_transformation(
            action="iam:CreateAccessKey",
            target_resource="arn:aws:iam::123:user/Alice",
            condition={"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}},
        )
        assert not val_bad_key.is_valid
        assert "not applicable to action" in val_bad_key.reason

        # Unrecognized arbitrary condition key
        val_bogus = model.validate_transformation(
            action="iam:PassRole",
            target_resource="arn:aws:iam::123:role/DevRole",
            condition={"StringEquals": {"unsupported:BogusKey": "val"}},
        )
        assert not val_bogus.is_valid

    def test_statement_validation_helper(self) -> None:
        """Verify bulk statement validation against the capability model."""
        model = load_default_capability_model()

        valid_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:PassRole"],
            Resource=["arn:aws:iam::123:role/DevRole"],
            Condition={"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}},
        )
        assert model.validate_statement(valid_stmt) == []

        invalid_stmt = Statement(
            Effect=Effect.ALLOW,
            Action=["iam:ListRoles"],
            Resource=["arn:aws:iam::123:role/DevRole"],  # Invalid: ListRoles is wildcard-only
        )
        errors = model.validate_statement(invalid_stmt)
        assert len(errors) == 1
        assert "wildcard-only" in errors[0]
