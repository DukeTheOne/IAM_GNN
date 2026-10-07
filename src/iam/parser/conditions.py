"""Parsing and evaluation logic for AWS IAM condition blocks."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from typing import Any


def _coerce_to_list(value: Any) -> list[Any]:
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _eval_string_equals(val: Any, target: Any) -> bool:
    return str(val) == str(target)


def _eval_string_equals_ignore_case(val: Any, target: Any) -> bool:
    return str(val).lower() == str(target).lower()


def _eval_string_like(val: Any, target: Any) -> bool:
    pattern = fnmatch.translate(str(target))
    return bool(re.match(pattern, str(val)))


def _eval_bool(val: Any, target: Any) -> bool:
    target_bool = str(target).lower() in ("true", "1")
    val_bool = val if isinstance(val, bool) else str(val).lower() in ("true", "1")
    return val_bool == target_bool


def _eval_numeric(val: Any, target: Any, op: str) -> bool:
    try:
        v = float(val)
        t = float(target)
    except (ValueError, TypeError):
        return False

    if op == "NumericEquals":
        return v == t
    if op == "NumericNotEquals":
        return v != t
    if op == "NumericLessThan":
        return v < t
    if op == "NumericLessThanEquals":
        return v <= t
    if op == "NumericGreaterThan":
        return v > t
    if op == "NumericGreaterThanEquals":
        return v >= t
    return False


def _eval_arn_equals(val: Any, target: Any) -> bool:
    return str(val).strip() == str(target).strip()


def _eval_arn_like(val: Any, target: Any) -> bool:
    pattern = fnmatch.translate(str(target).strip())
    return bool(re.match(pattern, str(val).strip()))


def evaluate_single_operator(
    operator: str,
    context_val: Any,
    policy_vals: list[Any],
) -> bool:
    """Evaluate a single condition operator against context value and policy values."""
    op = operator

    # Handle IfExists: condition key is evaluated only if it exists
    if op.endswith("IfExists") and context_val is None:
        return True
    if op.endswith("IfExists"):
        op = op[:-8]

    # Handle Null operator
    if op == "Null":
        for target in policy_vals:
            expect_null = str(target).lower() in ("true", "1")
            actual_null = context_val is None
            if expect_null == actual_null:
                return True
        return False

    if context_val is None:
        return False

    # Check against any of the policy target values (standard AWS OR semantics for values list)
    for target in policy_vals:
        if op == "StringEquals" and _eval_string_equals(context_val, target):
            return True
        if op == "StringNotEquals" and not _eval_string_equals(context_val, target):
            return True
        if op == "StringEqualsIgnoreCase" and _eval_string_equals_ignore_case(context_val, target):
            return True
        if op == "StringNotEqualsIgnoreCase" and not _eval_string_equals_ignore_case(
            context_val, target
        ):
            return True
        if op in ("StringLike", "ArnLike") and _eval_string_like(context_val, target):
            return True
        if op in ("StringNotLike", "ArnNotLike") and not _eval_string_like(context_val, target):
            return True
        if op == "Bool" and _eval_bool(context_val, target):
            return True
        if op.startswith("Numeric") and _eval_numeric(context_val, target, op):
            return True
        if op == "ArnEquals" and _eval_arn_equals(context_val, target):
            return True
        if op == "ArnNotEquals" and not _eval_arn_equals(context_val, target):
            return True

    return False


def evaluate_condition_block(
    condition_block: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
) -> bool:
    """Evaluate a complete AWS Condition block against an authorization request context.

    All operators in the condition block must evaluate to True (logical AND).
    Within each operator, all keys must evaluate to True (logical AND).
    Within each key, if multiple target values are specified, matching ANY satisfies (logical OR).
    """
    if not condition_block:
        return True

    for operator_str, key_dict in condition_block.items():
        # Handle Set modifiers
        is_for_any = operator_str.startswith("ForAnyValue:")
        is_for_all = operator_str.startswith("ForAllValues:")

        core_operator = operator_str
        if is_for_any:
            core_operator = operator_str[len("ForAnyValue:") :]
        elif is_for_all:
            core_operator = operator_str[len("ForAllValues:") :]

        for key_name, target_values_raw in key_dict.items():
            context_val = context.get(key_name)
            target_values = _coerce_to_list(target_values_raw)

            if is_for_any:
                context_vals = _coerce_to_list(context_val) if context_val is not None else []
                # True if at least one context value satisfies the condition
                if not any(
                    evaluate_single_operator(core_operator, c_val, target_values)
                    for c_val in context_vals
                ):
                    return False

            elif is_for_all:
                context_vals = _coerce_to_list(context_val) if context_val is not None else []
                # True if every context value satisfies the condition
                if not all(
                    evaluate_single_operator(core_operator, c_val, target_values)
                    for c_val in context_vals
                ):
                    return False

            else:
                if not evaluate_single_operator(core_operator, context_val, target_values):
                    return False

    return True
