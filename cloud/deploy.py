"""Deploy and push tool for serverless per-owner clync-share on AWS (ADR 0006).

Zero hardcoded account IDs, profiles, subnets, or hostnames: pass environment
settings via CLI flags; per-owner deployment metadata is saved locally under
`$CLYNC_DATA_HOME/cloud_<owner>.json`.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "share-client"))

from clync_cipher import (  # noqa: E402
    ClyncCipher,
    dense_literal,
    parse_dense_literal,
)
import clync_share_client as share_client  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_HOME = Path(os.environ.get("CLYNC_DATA_HOME", Path.home() / ".local/share/clync"))
LAMBDA_MEMORY_MB = 10240
LAMBDA_TIMEOUT_SECS = 120
BATCH_CHUNKS_MAX = 2500
LOG_FORWARD_FILTER_PATTERN = (
    '?ERROR ?CRITICAL ?FATAL ?"\\"error\\"" ?"\\"critical\\"" ?"\\"fatal\\"" ?"Traceback (most recent call last)"'
)


def _state_path(owner: str) -> Path:
    return DATA_HOME / f"cloud_{owner}.json"


def _load_state(owner: str) -> dict[str, Any]:
    path = _state_path(owner)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_state(owner: str, state: dict[str, Any]) -> None:
    path = _state_path(owner)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def _tags_dict(system_name: str, owner: str) -> dict[str, str]:
    return {"System": system_name, "Component": "clync-share", "Owner": owner}


def _tags_kv(system_name: str, owner: str) -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in _tags_dict(system_name, owner).items()]


def _ensure_iam_role(
    iam,
    role_name: str,
    assume_policy: dict[str, Any],
    inline_policies: dict[str, dict[str, Any]],
    managed_arns: list[str],
    tags: list[dict[str, str]],
) -> tuple[str, str]:
    try:
        role = iam.get_role(RoleName=role_name)["Role"]
        iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(assume_policy))
        iam.tag_role(RoleName=role_name, Tags=tags)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_policy),
            Tags=tags,
        )["Role"]
        time.sleep(8)
    for arn in managed_arns:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=arn)
    for pol_name, doc in inline_policies.items():
        iam.put_role_policy(RoleName=role_name, PolicyName=pol_name, PolicyDocument=json.dumps(doc))
    return role["Arn"], role["RoleId"]


def _build_kms_policy(
    account_id: str,
    owner_user_ids: list[str],
    lambda_role_arn: str,
    lambda_role_id: str,
    owner_role_arns: list[str],
) -> dict[str, Any]:
    """Per-owner KMS key policy with NO account-root statement.

    Admin/policy edits require the owner's own SSO `aws:userid`; decryption is
    restricted to the owner's SSO `aws:userid` and the owner's Lambda role.
    """
    owner_principals = sorted(set(owner_role_arns))
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "OwnerAdminOnly",
                "Effect": "Allow",
                "Principal": {"AWS": owner_principals},
                "Action": "kms:*",
                "Resource": "*",
                "Condition": {"StringEquals": {"aws:userid": owner_user_ids}},
            },
            {
                "Sid": "OwnerLambdaDecryptOnly",
                "Effect": "Allow",
                "Principal": {"AWS": lambda_role_arn},
                "Action": ["kms:Decrypt", "kms:DescribeKey"],
                "Resource": "*",
                "Condition": {"StringLike": {"aws:userid": f"{lambda_role_id}:*"}},
            },
        ],
    }


def _ensure_kms_key(
    kms,
    owner: str,
    account_id: str,
    owner_user_ids: list[str],
    owner_role_arns: list[str],
    lambda_role_arn: str,
    lambda_role_id: str,
    system_name: str,
) -> tuple[str, str]:
    alias_name = f"alias/clync-share-{owner}"
    policy = _build_kms_policy(account_id, owner_user_ids, lambda_role_arn, lambda_role_id, owner_role_arns)
    key_id = None
    try:
        aliases = kms.list_aliases()["Aliases"]
        for a in aliases:
            if a.get("AliasName") == alias_name and a.get("TargetKeyId"):
                key_id = a["TargetKeyId"]
                break
    except ClientError:
        key_id = None

    if key_id:
        meta = kms.describe_key(KeyId=key_id)["KeyMetadata"]
        key_arn = meta["Arn"]
        kms.put_key_policy(KeyId=key_id, PolicyName="default", Policy=json.dumps(policy))
        kms.tag_resource(
            KeyId=key_id,
            Tags=[{"TagKey": k, "TagValue": v} for k, v in _tags_dict(system_name, owner).items()],
        )
        return key_id, key_arn

    created = kms.create_key(
        Description=f"clync-share per-owner encryption key for {owner}",
        Policy=json.dumps(policy),
        Tags=[{"TagKey": k, "TagValue": v} for k, v in _tags_dict(system_name, owner).items()],
    )["KeyMetadata"]
    key_id = created["KeyId"]
    kms.create_alias(AliasName=alias_name, TargetKeyId=key_id)
    return key_id, created["Arn"]


def _ensure_staging_bucket(s3, bucket: str, region: str, system_name: str, owner: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-clync-staging-1d",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "Expiration": {"Days": 1},
                }
            ]
        },
    )
    s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": _tags_kv(system_name, owner)})


def _ensure_security_group(ec2, vpc_id: str, name: str, rds_sg_id: str, system_name: str, owner: str) -> str:
    found = ec2.describe_security_groups(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": [name]}]
    )["SecurityGroups"]
    if found:
        sg_id = found[0]["GroupId"]
    else:
        sg_id = ec2.create_security_group(
            GroupName=name,
            Description=f"clync-share Lambda egress SG for {owner}",
            VpcId=vpc_id,
            TagSpecifications=[{"ResourceType": "security-group", "Tags": _tags_kv(system_name, owner)}],
        )["GroupId"]
    ec2.create_tags(Resources=[sg_id], Tags=_tags_kv(system_name, owner))
    if rds_sg_id:
        try:
            ec2.authorize_security_group_ingress(
                GroupId=rds_sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 5432,
                        "ToPort": 5432,
                        "UserIdGroupPairs": [{"GroupId": sg_id, "Description": f"Postgres 5432 from {name}"}],
                    }
                ],
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                raise
    return sg_id


def _ensure_ecr_image(
    ecr,
    region: str,
    account_id: str,
    repo_name: str,
    model_onnx: Path,
    tokenizer_json: Path,
    system_name: str,
    owner: str,
    *,
    skip_build: bool = False,
) -> str:
    try:
        repo = ecr.describe_repositories(repositoryNames=[repo_name])["repositories"][0]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise
        repo = ecr.create_repository(
            repositoryName=repo_name,
            imageScanningConfiguration={"scanOnPush": False},
            tags=_tags_kv(system_name, owner),
        )["repository"]
    ecr.tag_resource(resourceArn=repo["repositoryArn"], tags=_tags_kv(system_name, owner))
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"
    if skip_build:
        return image_uri

    if not model_onnx.exists() or not tokenizer_json.exists():
        raise FileNotFoundError(
            f"missing ONNX artifacts ({model_onnx}, {tokenizer_json}); run experiments/lambda_embed_bench/prepare_and_parity.py first"
        )

    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, passwd = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    registry = auth["proxyEndpoint"]
    subprocess.run(["docker", "login", "--username", user, "--password-stdin", registry], input=passwd, text=True, check=True)

    build_root = DATA_HOME / "bench" / "clync_share_docker_build"
    if build_root.exists():
        shutil.rmtree(build_root)
    (build_root / "model").mkdir(parents=True)
    shutil.copy2(HERE / "Dockerfile", build_root / "Dockerfile")
    shutil.copy2(HERE / "handler.py", build_root / "handler.py")
    shutil.copy2(ROOT / "clync_cipher.py", build_root / "clync_cipher.py")
    shutil.copy2(model_onnx, build_root / "model" / "bge_m3_int8_dense_sparse.onnx")
    shutil.copy2(tokenizer_json, build_root / "model" / "tokenizer.json")
    try:
        subprocess.run(
            ["docker", "build", "--platform", "linux/arm64", "--provenance=false", "-t", image_uri, str(build_root)],
            check=True,
        )
        subprocess.run(["docker", "push", image_uri], check=True)
    finally:
        shutil.rmtree(build_root, ignore_errors=True)
    return image_uri


def _wait_lambda_ready(lam, fn_name: str) -> dict[str, Any]:
    for _ in range(90):
        cfg = lam.get_function_configuration(FunctionName=fn_name)
        state = cfg.get("State", "Active")
        upd = cfg.get("LastUpdateStatus", "Successful")
        if state == "Active" and upd == "Successful":
            return cfg
        if state == "Failed" or upd == "Failed":
            raise RuntimeError(f"Lambda {fn_name} failed: {cfg.get('StateReason') or cfg.get('LastUpdateStatusReason')}")
        time.sleep(2)
    raise TimeoutError(f"Lambda {fn_name} did not become Active")


def _ensure_log_group(logs, name: str, system_name: str, owner: str, log_forwarder_arn: str | None) -> None:
    try:
        logs.create_log_group(logGroupName=name, tags=_tags_dict(system_name, owner))
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
        logs.tag_resource(
            resourceArn=logs.describe_log_groups(logGroupNamePrefix=name)["logGroups"][0]["arn"].rstrip(":*"),
            tags=_tags_dict(system_name, owner),
        )
    logs.put_retention_policy(logGroupName=name, retentionInDays=90)
    if log_forwarder_arn:
        logs.put_subscription_filter(
            logGroupName=name,
            filterName="clync-share-forward",
            filterPattern=LOG_FORWARD_FILTER_PATTERN,
            destinationArn=log_forwarder_arn,
        )


def _deploy_tamper_alert(
    session: boto3.Session,
    *,
    owner: str,
    account_id: str,
    region: str,
    system_name: str,
    env_label: str,
    kms_key_id: str,
    main_lambda_name: str,
    main_lambda_role_arn: str,
    owner_user_ids: list[str],
    log_forwarder_arn: str | None,
) -> dict[str, str]:
    iam = session.client("iam")
    lam = session.client("lambda")
    events = session.client("events")
    logs = session.client("logs")

    alert_fn_name = f"clync-tamper-alert-{owner}"
    alert_role_name = f"clync-tamper-alert-{owner}-role"
    rule_name = f"clync-tamper-watch-{owner}"
    log_group = f"/aws/lambda/{alert_fn_name}"

    _ensure_log_group(logs, log_group, system_name, owner, log_forwarder_arn)
    assume = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
    role_arn, _ = _ensure_iam_role(
        iam,
        alert_role_name,
        assume,
        {},
        ["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"],
        _tags_kv(system_name, owner),
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(HERE / "tamper_alert.py", arcname="tamper_alert.py")
    zip_bytes = buf.getvalue()

    env_vars = {
        "CLYNC_OWNER": owner,
        "CLYNC_ENV_LABEL": env_label,
        "CLYNC_KMS_KEY_ID": kms_key_id,
        "CLYNC_LAMBDA_ROLE_ARN": main_lambda_role_arn,
        "CLYNC_OWNER_USER_IDS": ",".join(owner_user_ids),
        "CLYNC_WATCHED_LAMBDA_NAME": main_lambda_name,
        "CLYNC_WATCHED_RULE_NAME": rule_name,
        "CLYNC_WATCHED_LOG_GROUP": log_group,
    }
    try:
        lam.get_function(FunctionName=alert_fn_name)
        lam.update_function_code(FunctionName=alert_fn_name, ZipFile=zip_bytes)
        _wait_lambda_ready(lam, alert_fn_name)
        lam.update_function_configuration(
            FunctionName=alert_fn_name,
            Runtime="python3.12",
            Handler="tamper_alert.lambda_handler",
            Role=role_arn,
            Timeout=15,
            MemorySize=256,
            Environment={"Variables": env_vars},
        )
        cfg = _wait_lambda_ready(lam, alert_fn_name)
        lam.tag_resource(Resource=cfg["FunctionArn"], Tags=_tags_dict(system_name, owner))
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        cfg = lam.create_function(
            FunctionName=alert_fn_name,
            Runtime="python3.12",
            Role=role_arn,
            Handler="tamper_alert.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Architectures=["arm64"],
            Timeout=15,
            MemorySize=256,
            Environment={"Variables": env_vars},
            Tags=_tags_dict(system_name, owner),
        )
        cfg = _wait_lambda_ready(lam, alert_fn_name)

    pattern = {
        "detail-type": ["AWS API Call via CloudTrail"],
        "detail": {
            "eventName": [
                "Decrypt",
                "PutKeyPolicy",
                "ScheduleKeyDeletion",
                "DisableKey",
                "UpdateFunctionCode20150331v2",
                "UpdateFunctionConfiguration20150331v2",
                "PutFunctionCodeSigningConfig",
                "DisableRule",
                "DeleteRule",
                "RemoveTargets",
                "PutRule",
                "DeleteSubscriptionFilter",
                "PutModelInvocationLoggingConfiguration",
            ]
        },
    }
    rule_arn = events.put_rule(
        Name=rule_name,
        EventPattern=json.dumps(pattern),
        State="ENABLED",
        Description=f"CloudTrail tamper watch for clync-share owner {owner}",
        Tags=_tags_kv(system_name, owner),
    )["RuleArn"]
    try:
        lam.add_permission(
            FunctionName=alert_fn_name,
            StatementId="AllowEventBridgeTamperWatch",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceConflictException":
            raise
    events.put_targets(Rule=rule_name, Targets=[{"Id": "tamper-alert-lambda", "Arn": cfg["FunctionArn"]}])
    return {"alert_lambda_arn": cfg["FunctionArn"], "alert_rule_arn": rule_arn, "alert_log_group": log_group}


def cmd_deploy(args: argparse.Namespace) -> int:
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    account_id = ident["Account"]
    iam = session.client("iam")
    kms = session.client("kms")
    s3 = session.client("s3")
    ec2 = session.client("ec2")
    ecr = session.client("ecr")
    lam = session.client("lambda")
    logs = session.client("logs")

    owner = args.owner
    system_name = args.system_name
    fn_name = f"clync-share-{owner}"
    role_name = f"clync-share-{owner}-role"
    repo_name = f"clync-share-{owner}"
    sg_name = f"clync-share-{owner}-sg"
    bucket = args.staging_bucket or f"clync-share-staging-{account_id}-{args.region}"
    db_schema = f"clync_{owner}"
    db_user = args.db_user or f"clync_{owner}"

    owner_user_ids = [x.strip() for x in args.owner_user_ids.split(",") if x.strip()]
    owner_role_arns = [x.strip() for x in args.owner_role_arns.split(",") if x.strip()]
    allowed_role_prefixes = [x.strip() for x in args.allowed_role_prefixes.split(",") if x.strip()]

    print(f"[1/6] Staging bucket {bucket} (tag={system_name})...")
    _ensure_staging_bucket(s3, bucket, args.region, system_name, owner)

    print(f"[2/6] IAM role {role_name} & SG {sg_name}...")
    assume = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
    inline = {
        "clync-share-runtime": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["rds-db:connect"],
                    "Resource": f"arn:aws:rds-db:{args.region}:{account_id}:dbuser:{args.db_resource_id}/{db_user}",
                },
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:DeleteObject"],
                    "Resource": f"arn:aws:s3:::{bucket}/{owner}/*",
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:GetModelInvocationLoggingConfiguration",
                    ],
                    "Resource": "*",
                },
            ],
        }
    }
    role_arn, role_id = _ensure_iam_role(
        iam,
        role_name,
        assume,
        inline,
        ["arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"],
        _tags_kv(system_name, owner),
    )
    sg_id = _ensure_security_group(ec2, args.vpc_id, sg_name, args.rds_sg_id, system_name, owner)

    print(f"[3/6] Per-owner KMS CMK alias/clync-share-{owner} (no account-root statement)...")
    key_id, key_arn = _ensure_kms_key(
        kms, owner, account_id, owner_user_ids, owner_role_arns, role_arn, role_id, system_name
    )
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="clync-share-kms-decrypt",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["kms:Decrypt", "kms:DescribeKey"], "Resource": key_arn}],
            }
        ),
    )

    state = _load_state(owner)
    data_key_ct_b64 = state.get("data_key_ct_b64")
    if not data_key_ct_b64 or state.get("kms_key_id") != key_id:
        dk = kms.generate_data_key(KeyId=key_id, KeySpec="AES_256", EncryptionContext={"clync_owner": owner})
        data_key_ct_b64 = base64.b64encode(dk["CiphertextBlob"]).decode("ascii")

    print(f"[4/6] ECR container image {repo_name}:latest...")
    image_uri = _ensure_ecr_image(
        ecr,
        args.region,
        account_id,
        repo_name,
        Path(args.model_onnx).expanduser(),
        Path(args.tokenizer_json).expanduser(),
        system_name,
        owner,
        skip_build=args.skip_image_build,
    )

    print(f"[5/6] Container Lambda {fn_name} ({LAMBDA_MEMORY_MB} MB, arm64)...")
    main_log_group = f"/aws/lambda/{fn_name}"
    _ensure_log_group(logs, main_log_group, system_name, owner, args.log_forwarder_arn)
    subnets = [s.strip() for s in args.subnets.split(",") if s.strip()]
    env_vars = {
        "CLYNC_OWNER": owner,
        "CLYNC_DB_SCHEMA": db_schema,
        "CLYNC_DB_HOST": args.db_host,
        "CLYNC_DB_PORT": str(args.db_port),
        "CLYNC_DB_NAME": args.db_name,
        "CLYNC_DB_USER": db_user,
        "CLYNC_ALLOWED_ACCOUNT_ID": account_id,
        "CLYNC_OWNER_USER_IDS": ",".join(owner_user_ids),
        "CLYNC_ALLOWED_ROLE_PREFIXES": ",".join(allowed_role_prefixes),
        "CLYNC_ENABLE_SHARE_MCP": "true" if not args.disable_share else "false",
        "CLYNC_ENABLE_OWNER_MCP": "true" if not args.disable_owner else "false",
        "CLYNC_DATA_KEY_CT_B64": data_key_ct_b64,
        "CLYNC_KMS_KEY_ID": key_arn,
        "CLYNC_STAGING_BUCKET": bucket,
        "CLYNC_BEDROCK_REGION": args.bedrock_region,
        "CLYNC_BEDROCK_MODEL_ID": args.bedrock_model_id,
    }
    try:
        lam.get_function(FunctionName=fn_name)
        lam.update_function_code(FunctionName=fn_name, ImageUri=image_uri)
        _wait_lambda_ready(lam, fn_name)
        lam.update_function_configuration(
            FunctionName=fn_name,
            Role=role_arn,
            Timeout=LAMBDA_TIMEOUT_SECS,
            MemorySize=LAMBDA_MEMORY_MB,
            VpcConfig={"SubnetIds": subnets, "SecurityGroupIds": [sg_id]},
            Environment={"Variables": env_vars},
        )
        cfg = _wait_lambda_ready(lam, fn_name)
        lam.tag_resource(Resource=cfg["FunctionArn"], Tags=_tags_dict(system_name, owner))
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        cfg = lam.create_function(
            FunctionName=fn_name,
            PackageType="Image",
            Code={"ImageUri": image_uri},
            Role=role_arn,
            Architectures=["arm64"],
            Timeout=LAMBDA_TIMEOUT_SECS,
            MemorySize=LAMBDA_MEMORY_MB,
            VpcConfig={"SubnetIds": subnets, "SecurityGroupIds": [sg_id]},
            Environment={"Variables": env_vars},
            Tags=_tags_dict(system_name, owner),
        )
        cfg = _wait_lambda_ready(lam, fn_name)

    print("[6/6] EventBridge tamper-alert Lambda & rule...")
    alert_info = _deploy_tamper_alert(
        session,
        owner=owner,
        account_id=account_id,
        region=args.region,
        system_name=system_name,
        env_label=args.env_label,
        kms_key_id=key_id,
        main_lambda_name=fn_name,
        main_lambda_role_arn=role_arn,
        owner_user_ids=owner_user_ids,
        log_forwarder_arn=args.log_forwarder_arn,
    )

    state.update(
        {
            "owner": owner,
            "account_id": account_id,
            "region": args.region,
            "system_name": system_name,
            "lambda_arn": cfg["FunctionArn"],
            "lambda_name": fn_name,
            "lambda_role_arn": role_arn,
            "sg_id": sg_id,
            "ecr_uri": image_uri,
            "kms_key_id": key_id,
            "kms_key_arn": key_arn,
            "data_key_ct_b64": data_key_ct_b64,
            "staging_bucket": bucket,
            "db_host": args.db_host,
            "db_name": args.db_name,
            "db_schema": db_schema,
            "db_user": db_user,
            "bedrock_region": args.bedrock_region,
            "bedrock_model_id": args.bedrock_model_id,
            **alert_info,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_state(owner, state)
    print(json.dumps({"lambda_arn": cfg["FunctionArn"], "kms_key_arn": key_arn, **alert_info}, indent=2))
    return 0


def _filter_changed_units(
    units: list[dict[str, Any]], pushed_units: dict[str, str], *, full: bool
) -> list[dict[str, Any]]:
    if full:
        return list(units)
    changed = []
    for u in units:
        stamp = u["updated_at"].isoformat() if u.get("updated_at") else ""
        if pushed_units.get(u["unit_id"]) != stamp:
            changed.append(u)
    return changed


def cmd_push(args: argparse.Namespace) -> int:
    import clync

    state = _load_state(args.owner)
    if not state:
        raise SystemExit(f"no cloud deployment state for {args.owner!r} at {_state_path(args.owner)}")

    with clync.connect_pg() as con:
        unit_sql = "SELECT unit_id, kind, source, title, summary, model, project_uuid, project_name, repo, worktree, git_branch, updated_at FROM units ORDER BY updated_at DESC NULLS LAST"
        if args.limit_units:
            unit_sql += f" LIMIT {int(args.limit_units)}"
        units = con.execute(unit_sql).fetchall()
        if args.include_dream:
            dream_units = con.execute(
                "SELECT unit_id, kind, source, title, summary, model, project_uuid, project_name, repo, worktree, git_branch, updated_at "
                "FROM units WHERE source = 'dream'"
            ).fetchall()
            seen_u = {u["unit_id"] for u in units}
            for du in dream_units:
                if du["unit_id"] not in seen_u:
                    units.append(du)
                    seen_u.add(du["unit_id"])

        pushed_units: dict[str, str] = dict(state.get("pushed_units") or {})
        units = _filter_changed_units(units, pushed_units, full=bool(args.full))
        if not units:
            print("Up to date (0 changed units; pass --full to force re-push).", flush=True)
            return 0
        unit_ids = [u["unit_id"] for u in units]

        msgs = con.execute(
            "SELECT unit_id, idx, sender, text, created_at FROM messages WHERE unit_id = ANY(%s) ORDER BY unit_id, idx",
            (unit_ids,),
        ).fetchall()
        chunks = con.execute(
            "SELECT chunk_id, unit_id, source, unit_name, sender, lang, msg_idx, chunk_idx, updated_at, repo, project_name, text, dense::text AS dense, sparse::text AS sparse "
            "FROM chunks WHERE unit_id = ANY(%s)",
            (unit_ids,),
        ).fetchall()

    session = boto3.Session(profile_name=args.profile, region_name=state["region"])
    kms = session.client("kms")
    s3 = session.client("s3")
    dk_plain = kms.decrypt(
        KeyId=state["kms_key_arn"],
        CiphertextBlob=base64.b64decode(state["data_key_ct_b64"]),
        EncryptionContext={"clync_owner": args.owner},
    )["Plaintext"]
    cipher = ClyncCipher(dk_plain)

    print(f"Encrypting {len(units)} units, {len(msgs)} messages, {len(chunks)} chunks...")
    t0 = time.perf_counter()
    records: list[str] = []
    for u in units:
        meta_json = json.dumps(
            {
                "kind": u["kind"],
                "source": u["source"],
                "title": u["title"],
                "summary": u["summary"],
                "model": u["model"],
                "project_name": u["project_name"],
                "repo": u["repo"],
                "worktree": u["worktree"],
                "git_branch": u["git_branch"],
            }
        )
        records.append(
            json.dumps(
                {
                    "t": "unit",
                    "unit_id": u["unit_id"],
                    "source_hmac": cipher.hmac_facet("source", u["source"]),
                    "kind_hmac": cipher.hmac_facet("kind", u["kind"]),
                    "repo_hmac": cipher.hmac_facet("repo", u["repo"]),
                    "project_hmac": cipher.hmac_facet("project", u["project_name"]),
                    "updated_at": u["updated_at"].isoformat() if u["updated_at"] else None,
                    "meta_ct": cipher.encrypt_text(meta_json, aad=f"units:{u['unit_id']}"),
                }
            )
        )
    for m in msgs:
        if m["idx"] is None:
            continue
        payload = json.dumps({"sender": m["sender"], "text": m["text"] or ""})
        records.append(
            json.dumps(
                {
                    "t": "msg",
                    "unit_id": m["unit_id"],
                    "msg_idx": int(m["idx"]),
                    "sender_hmac": cipher.hmac_facet("sender", m["sender"]),
                    "created_at": m["created_at"].isoformat() if m["created_at"] else None,
                    "payload_ct": cipher.encrypt_text(payload, aad=f"messages:{m['unit_id']}:{m['idx']}"),
                }
            )
        )
    for c in chunks:
        payload = json.dumps(
            {
                "source": c["source"],
                "unit_name": c["unit_name"],
                "sender": c["sender"],
                "lang": c["lang"],
                "text": c["text"],
            }
        )
        rot_dense = cipher.rotate_dense(parse_dense_literal(c["dense"]))
        perm_sparse = cipher.permute_sparse_literal(c["sparse"])
        records.append(
            json.dumps(
                {
                    "t": "chunk",
                    "chunk_id": c["chunk_id"],
                    "unit_id": c["unit_id"],
                    "source_hmac": cipher.hmac_facet("source", c["source"]),
                    "repo_hmac": cipher.hmac_facet("repo", c["repo"]),
                    "project_hmac": cipher.hmac_facet("project", c["project_name"]),
                    "sender_hmac": cipher.hmac_facet("sender", c["sender"]),
                    "lang_hmac": cipher.hmac_facet("lang", c["lang"]),
                    "msg_idx": c["msg_idx"],
                    "chunk_idx": c["chunk_idx"],
                    "updated_at": c["updated_at"].isoformat() if c["updated_at"] else None,
                    "payload_ct": cipher.encrypt_text(payload, aad=f"chunks:{c['chunk_id']}"),
                    "dense": dense_literal(rot_dense),
                    "sparse": perm_sparse,
                }
            )
        )
    print(f"Encrypted {len(records)} records in {time.perf_counter() - t0:.1f}s")

    batch_size = max(100, int(args.batch_records))
    total_ingested = {"units": 0, "messages": 0, "chunks": 0}
    n_batches = (len(records) + batch_size - 1) // batch_size
    for idx in range(n_batches):
        batch = records[idx * batch_size : (idx + 1) * batch_size]
        raw_gz = gzip.compress(("\n".join(batch) + "\n").encode("utf-8"))
        digest = hashlib.sha256(raw_gz).hexdigest()[:12]
        key = f"{args.owner}/push/{int(time.time())}-{idx:04d}-{digest}.ndjson.gz"
        s3.put_object(Bucket=state["staging_bucket"], Key=key, Body=raw_gz)
        is_last = idx == n_batches - 1
        res = share_client.invoke_share_lambda(
            state["lambda_arn"],
            "ingest",
            {"s3_key": key, "ensure_hnsw": is_last},
            profile=args.profile,
            region=state["region"],
        )
        for k in total_ingested:
            total_ingested[k] += int(res.get("ingested", {}).get(k, 0))
        print(
            f"  batch {idx + 1}/{n_batches} ({len(raw_gz) / 1024:.1f} KB gz) -> "
            f"{res['ingested']} hnsw={res['hnsw_ensured']}",
            flush=True,
        )
    if args.full:
        pushed_units = {}
    for u in units:
        pushed_units[u["unit_id"]] = u["updated_at"].isoformat() if u.get("updated_at") else ""
    state["pushed_units"] = pushed_units
    state["last_push_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(args.owner, state)
    print("Push complete:", json.dumps(total_ingested), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy and push per-owner clync-share stack on AWS")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dep = sub.add_parser("deploy", help="create/update per-owner clync-share AWS stack")
    p_dep.add_argument("--owner", required=True, help="owner short slug (e.g. joy)")
    p_dep.add_argument("--region", required=True)
    p_dep.add_argument("--profile", help="AWS profile")
    p_dep.add_argument("--system-name", default="clync-share", help="cost-allocation tag value")
    p_dep.add_argument("--env-label", default="dev")
    p_dep.add_argument("--vpc-id", required=True)
    p_dep.add_argument("--subnets", required=True, help="comma-separated private subnet IDs")
    p_dep.add_argument("--rds-sg-id", default="", help="RDS security group ID to authorize 5432 from Lambda SG")
    p_dep.add_argument("--db-host", required=True)
    p_dep.add_argument("--db-port", type=int, default=5432)
    p_dep.add_argument("--db-name", default="clync")
    p_dep.add_argument("--db-user", default="")
    p_dep.add_argument("--db-resource-id", required=True, help="Aurora DbClusterResourceId for rds-db:connect IAM")
    p_dep.add_argument("--owner-user-ids", required=True, help="comma-separated owner aws:userid values")
    p_dep.add_argument("--owner-role-arns", required=True, help="comma-separated owner IAM role ARNs for KMS policy")
    p_dep.add_argument("--allowed-role-prefixes", required=True, help="comma-separated caller STS ARN prefixes for read-only ask")
    p_dep.add_argument("--staging-bucket", default="")
    p_dep.add_argument("--log-forwarder-arn", default="")
    p_dep.add_argument("--bedrock-region", required=True, help="region whose Bedrock invocation logging is off")
    p_dep.add_argument("--bedrock-model-id", required=True)
    p_dep.add_argument("--model-onnx", default=str(DATA_HOME / "bench/onnx/bge_m3_int8_dense_sparse.onnx"))
    p_dep.add_argument("--tokenizer-json", default="")
    p_dep.add_argument("--skip-image-build", action="store_true")
    p_dep.add_argument("--disable-share", action="store_true")
    p_dep.add_argument("--disable-owner", action="store_true")

    p_push = sub.add_parser("push", help="encrypt local clync store and push deltas to cloud Lambda")
    p_push.add_argument("--owner", required=True)
    p_push.add_argument("--profile", help="AWS profile (must match owner KMS key policy)")
    p_push.add_argument("--limit-units", type=int, default=0, help="optional cap on most recent units to push")
    p_push.add_argument("--include-dream", action="store_true", help="always include dream insight units")
    p_push.add_argument("--full", action="store_true", help="ignore local push watermark and re-push selected units")
    p_push.add_argument("--batch-records", type=int, default=1500)

    args = parser.parse_args(argv)
    if args.cmd == "deploy":
        if not args.tokenizer_json:
            tok = next(
                (
                    p
                    for pat in (
                        ".cache/huggingface/hub/models--gpahal--bge-m3-onnx-int8/snapshots/*/tokenizer.json",
                        ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*/tokenizer.json",
                    )
                    for p in Path.home().glob(pat)
                ),
                None,
            )
            if not tok:
                raise SystemExit("pass --tokenizer-json pointing to BGE-M3 tokenizer.json")
            args.tokenizer_json = str(tok)
        return cmd_deploy(args)
    if args.cmd == "push":
        return cmd_push(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
