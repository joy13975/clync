"""CloudTrail tamper-alert Lambda for clync-share.

Receives EventBridge events for watched KMS, Lambda, EventBridge, CWL, and
Bedrock-logging mutations. Ignores expected kms:Decrypt calls from the owner's
Lambda execution role or owner SSO principals, and emits one structured ERROR
JSON log line (`@event:clync_tamper_alert`) for every unexpected tamper event
so an optional log subscription can route it to an alerting system.
"""
from __future__ import annotations

import json
import os
from typing import Any

OWNER = os.environ.get("CLYNC_OWNER", "unknown")
ENV_LABEL = os.environ.get("CLYNC_ENV_LABEL", "dev")
ALLOWED_KMS_ROLE_ARN = os.environ.get("CLYNC_LAMBDA_ROLE_ARN", "")
ALLOWED_OWNER_USER_IDS = {
    x.strip() for x in os.environ.get("CLYNC_OWNER_USER_IDS", "").split(",") if x.strip()
}
WATCHED_KMS_KEY_ID = os.environ.get("CLYNC_KMS_KEY_ID", "")
WATCHED_LAMBDA_NAME = os.environ.get("CLYNC_WATCHED_LAMBDA_NAME", "")
WATCHED_RULE_NAME = os.environ.get("CLYNC_WATCHED_RULE_NAME", "")
WATCHED_LOG_GROUP = os.environ.get("CLYNC_WATCHED_LOG_GROUP", "")


def _is_relevant_target(event_name: str, detail: dict[str, Any]) -> tuple[bool, str]:
    req = detail.get("requestParameters") or {}
    resources = detail.get("resources") or []
    res_arns = [str(r.get("ARN", "")) for r in resources]
    if event_name.startswith("Decrypt") or event_name in ("PutKeyPolicy", "ScheduleKeyDeletion", "DisableKey"):
        key_id = str(req.get("keyId", ""))
        matched = bool(
            WATCHED_KMS_KEY_ID
            and (WATCHED_KMS_KEY_ID in key_id or any(WATCHED_KMS_KEY_ID in a for a in res_arns))
        )
        return matched, key_id or (res_arns[0] if res_arns else "kms-key")
    if event_name in ("UpdateFunctionCode20150331v2", "UpdateFunctionConfiguration20150331v2", "PutFunctionCodeSigningConfig", "UpdateFunctionCode", "UpdateFunctionConfiguration"):
        fn = str(req.get("functionName", ""))
        return bool(WATCHED_LAMBDA_NAME and WATCHED_LAMBDA_NAME in fn), fn or "lambda"
    if event_name in ("DisableRule", "DeleteRule", "RemoveTargets", "PutRule"):
        rule = str(req.get("name", "") or req.get("rule", ""))
        return bool(WATCHED_RULE_NAME and WATCHED_RULE_NAME == rule), rule or "eventbridge-rule"
    if event_name == "DeleteSubscriptionFilter":
        lg = str(req.get("logGroupName", ""))
        return bool(WATCHED_LOG_GROUP and WATCHED_LOG_GROUP == lg), lg or "log-group"
    if event_name == "PutModelInvocationLoggingConfiguration":
        return True, f"bedrock-logging:{detail.get('awsRegion', '')}"
    return False, event_name


def _is_expected_kms_decrypt(event_name: str, detail: dict[str, Any]) -> bool:
    if event_name != "Decrypt":
        return False
    ident = detail.get("userIdentity") or {}
    principal_id = str(ident.get("principalId", ""))
    arn = str(ident.get("arn", ""))
    session_issuer_arn = str(
        ((ident.get("sessionContext") or {}).get("sessionIssuer") or {}).get("arn", "")
    )
    if principal_id in ALLOWED_OWNER_USER_IDS:
        return True
    if ALLOWED_KMS_ROLE_ARN and (
        session_issuer_arn == ALLOWED_KMS_ROLE_ARN or arn.startswith(ALLOWED_KMS_ROLE_ARN)
    ):
        return True
    return False


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    detail = (event or {}).get("detail") or {}
    event_name = str(detail.get("eventName", "unknown"))
    relevant, target = _is_relevant_target(event_name, detail)
    if not relevant:
        return {"alerted": False, "reason": "unrelated_target", "event_name": event_name}
    if _is_expected_kms_decrypt(event_name, detail):
        return {"alerted": False, "reason": "expected_decrypt", "event_name": event_name}

    ident = detail.get("userIdentity") or {}
    record = {
        "level": "ERROR",
        "event": "clync_tamper_alert",
        "env": ENV_LABEL,
        "owner": OWNER,
        "event_name": event_name,
        "actor_arn": ident.get("arn") or ident.get("principalId") or "unknown",
        "target": target,
        "source_ip": detail.get("sourceIPAddress") or "unknown",
        "event_time": detail.get("eventTime") or (event or {}).get("time") or "unknown",
        "aws_region": detail.get("awsRegion") or "unknown",
    }
    print(json.dumps(record, sort_keys=True))
    return {"alerted": True, "record": record}
