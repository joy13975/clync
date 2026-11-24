"""Throwaway AWS Lambda CPU embedding benchmark (S3-to-RAM zip vs ECR container image).

Provisions temporary resources tagged with any `--tag` pairs, runs all three arms
(5 forced cold starts + 20 warm queries per arm/tier), prints + saves measurements,
and unconditionally tears down every created resource in a `finally` block.

Zero hardcoded AWS profiles, account IDs, or private queries: pass everything via
CLI flags / ambient AWS credentials.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import subprocess
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
REPORT_RE = re.compile(
    r"Duration:\s*(?P<dur>[\d.]+)\s*ms\s+"
    r"Billed Duration:\s*(?P<billed>[\d.]+)\s*ms\s+"
    r"Memory Size:\s*(?P<mem>\d+)\s*MB\s+"
    r"Max Memory Used:\s*(?P<max_mem>\d+)\s*MB"
    r"(?:\s+Init Duration:\s*(?P<init>[\d.]+)\s*ms)?"
)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, round(q * (len(s) - 1)))], 1)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "p50": round(statistics.median(values), 1),
        "p95": _pct(values, 0.95),
        "max": round(max(values), 1),
    }


def _parse_report(log_b64: str) -> dict:
    text = base64.b64decode(log_b64).decode("utf-8", errors="replace")
    m = REPORT_RE.search(text)
    if not m:
        return {"raw_tail": text[-300:]}
    return {
        "duration_ms": float(m.group("dur")),
        "billed_ms": float(m.group("billed")),
        "memory_mb": int(m.group("mem")),
        "max_memory_mb": int(m.group("max_mem")),
        "init_duration_ms": float(m.group("init")) if m.group("init") else None,
    }


def build_zip_bundle(build_dir: Path) -> Path:
    """Install arm64 Linux wheels for onnxruntime/tokenizers/numpy inside Lambda's
    arm64 Python 3.12 image and zip them with lambda_handler.py."""
    pkg_dir = build_dir / "zip_pkg"
    zip_path = build_dir / "lambda_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(["rm", "-rf", str(pkg_dir)], check=True)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "docker", "run", "--rm", "--platform", "linux/arm64",
            "-v", f"{pkg_dir}:/out",
            "--entrypoint", "pip",
            "public.ecr.aws/lambda/python:3.12-arm64",
            "install", "--no-cache-dir", "--target", "/out",
            "onnxruntime==1.32.0", "tokenizers==0.21.0", "numpy==2.2.3",
        ],
        check=True,
    )
    # Strip unnecessary metadata/tests/caches to stay comfortably below 250 MB unzipped
    for pat in ("__pycache__", "*.dist-info", "tests"):
        subprocess.run(["find", str(pkg_dir), "-name", pat, "-exec", "rm", "-rf", "{}", "+"], check=False)
    subprocess.run(["cp", str(HERE / "lambda_handler.py"), str(pkg_dir / "lambda_handler.py")], check=True)

    unzipped_bytes = sum(p.stat().st_size for p in pkg_dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(pkg_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(pkg_dir))
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    unzipped_mb = unzipped_bytes / (1024 * 1024)
    print(f"Built zip bundle: {zip_mb:.1f} MB zipped, {unzipped_mb:.1f} MB unzipped", flush=True)
    if unzipped_mb < 10:
        raise RuntimeError(f"Unzipped package unexpectedly tiny ({unzipped_mb:.1f} MB) — check Docker mount")
    if unzipped_mb >= 248:
        raise RuntimeError(f"Unzipped package {unzipped_mb:.1f} MB approaches 250 MB Lambda limit")
    return zip_path


def build_and_push_image(
    build_dir: Path, stripped_model: Path, tokenizer: Path, ecr_uri: str, region: str
) -> str:
    ctx = build_dir / "docker_ctx"
    subprocess.run(["rm", "-rf", str(ctx)], check=True)
    (ctx / "model").mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", str(HERE / "Dockerfile"), str(ctx / "Dockerfile")], check=True)
    subprocess.run(["cp", str(HERE / "lambda_handler.py"), str(ctx / "lambda_handler.py")], check=True)
    subprocess.run(["cp", str(stripped_model), str(ctx / "model/bge_m3_int8_dense_sparse.onnx")], check=True)
    subprocess.run(["cp", str(tokenizer), str(ctx / "model/tokenizer.json")], check=True)

    ecr = boto3.client("ecr", region_name=region)
    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, pwd = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    registry = auth["proxyEndpoint"]
    subprocess.run(["docker", "login", "--username", user, "--password-stdin", registry], input=pwd.encode(), check=True)

    image_tag = f"{ecr_uri}:bench"
    # --provenance=false avoids OCI attestations that Lambda rejects
    subprocess.run(
        [
            "docker", "buildx", "build", "--platform", "linux/arm64",
            "--provenance=false", "--sbom=false", "--load",
            "-t", image_tag, str(ctx),
        ],
        check=True,
    )
    subprocess.run(["docker", "push", image_tag], check=True)
    return image_tag


def wait_lambda_active(lam, fn_name: str) -> None:
    for _ in range(120):
        cfg = lam.get_function_configuration(FunctionName=fn_name)
        state = cfg.get("State")
        upd = cfg.get("LastUpdateStatus")
        if state == "Active" and upd in (None, "Successful"):
            return
        if state == "Failed" or upd == "Failed":
            raise RuntimeError(f"Lambda {fn_name} failed: {cfg.get('StateReason')} / {cfg.get('LastUpdateStatusReason')}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {fn_name} to become Active")


def measure_fn(lam, fn_name: str, env_base: dict, mem_mb: int, queries: list[str], cold_runs: int = 5, warm_runs: int = 20) -> dict:
    cold_records = []
    warm_records = []
    for i in range(cold_runs):
        # Update an env var nonce to force a fresh Lambda execution environment
        env = dict(env_base, COLD_NONCE=f"{mem_mb}-{i}-{int(time.time()*1000)}")
        lam.update_function_configuration(FunctionName=fn_name, MemorySize=mem_mb, Environment={"Variables": env})
        wait_lambda_active(lam, fn_name)

        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        resp = lam.invoke(FunctionName=fn_name, LogType="Tail", Payload=json.dumps({"query": q}).encode())
        wall_ms = (time.perf_counter() - t0) * 1000
        payload = json.loads(resp["Payload"].read())
        rep = _parse_report(resp.get("LogResult", ""))
        if resp.get("FunctionError"):
            cold_records.append({"error": payload, "report": rep, "wall_ms": round(wall_ms, 1)})
            print(f"  COLD {fn_name} @{mem_mb}MB #{i+1}: ERROR {payload} rep={rep}", flush=True)
            if i == 0 and payload.get("errorType") == "Runtime.ExitError" and (rep.get("max_memory_mb") or 0) < mem_mb * 0.5:
                print("  Aborting remaining cold runs for this tier (non-OOM runtime crash)", flush=True)
                break
            continue
        init_info = payload.get("init") or {}
        init_dur = rep.get("init_duration_ms") or init_info.get("init_total_ms") or 0.0
        cold_total = round(init_dur + rep.get("duration_ms", payload["handler_ms"]), 1)
        rec = {
            "is_cold": payload["is_cold"],
            "init_duration_ms": rep.get("init_duration_ms"),
            "init_download_ms": init_info.get("init_download_ms"),
            "init_ort_load_ms": init_info.get("init_ort_load_ms"),
            "handler_ms": payload["handler_ms"],
            "embed_ms": payload["embed_ms"],
            "cold_total_ms": cold_total,
            "wall_ms": round(wall_ms, 1),
            "max_memory_mb": rep.get("max_memory_mb"),
            "rss_after_dl_mb": init_info.get("rss_after_dl_mb"),
            "rss_after_ort_mb": init_info.get("rss_after_ort_mb"),
            "dense_head": payload.get("dense_head"),
            "sparse_nnz": payload.get("sparse_nnz"),
        }
        cold_records.append(rec)
        print(
            f"  COLD {fn_name} @{mem_mb}MB #{i+1}: init={rec['init_duration_ms']}ms "
            f"(dl={rec['init_download_ms']}ms ort={rec['init_ort_load_ms']}ms) "
            f"embed={rec['embed_ms']}ms total={rec['cold_total_ms']}ms max_mem={rec['max_memory_mb']}MB",
            flush=True,
        )

    # Warm runs on the last warm container (only if last cold run succeeded)
    if cold_records and "error" not in cold_records[-1]:
        for j in range(warm_runs):
            q = queries[(cold_runs + j) % len(queries)]
            t0 = time.perf_counter()
            resp = lam.invoke(FunctionName=fn_name, LogType="Tail", Payload=json.dumps({"query": q}).encode())
            wall_ms = (time.perf_counter() - t0) * 1000
            payload = json.loads(resp["Payload"].read())
            rep = _parse_report(resp.get("LogResult", ""))
            if resp.get("FunctionError"):
                warm_records.append({"error": payload, "report": rep})
                continue
            warm_records.append({
                "is_cold": payload["is_cold"],
                "embed_ms": payload["embed_ms"],
                "handler_ms": payload["handler_ms"],
                "duration_ms": rep.get("duration_ms"),
                "wall_ms": round(wall_ms, 1),
                "max_memory_mb": rep.get("max_memory_mb"),
            })

    ok_cold = [r for r in cold_records if "error" not in r]
    ok_warm = [r for r in warm_records if "error" not in r and not r.get("is_cold")]
    max_mem = max((r["max_memory_mb"] for r in ok_cold + ok_warm if r.get("max_memory_mb")), default=None)
    cold_summary = _summarize([r["cold_total_ms"] for r in ok_cold])
    warm_embed_summary = _summarize([r["embed_ms"] for r in ok_warm])
    mem_pct = round(100.0 * max_mem / mem_mb, 1) if max_mem else None
    passed = (
        len(ok_cold) == cold_runs
        and cold_summary.get("p95", 999999) <= 10000
        and warm_embed_summary.get("p95", 999999) <= 100
        and (mem_pct is not None and mem_pct <= 70.0)
    )
    return {
        "fn_name": fn_name,
        "memory_mb": mem_mb,
        "cold_ok": len(ok_cold),
        "cold_errors": len(cold_records) - len(ok_cold),
        "init_download_ms": _summarize([r["init_download_ms"] for r in ok_cold if r["init_download_ms"] is not None]),
        "init_ort_load_ms": _summarize([r["init_ort_load_ms"] for r in ok_cold if r["init_ort_load_ms"] is not None]),
        "init_duration_ms": _summarize([r["init_duration_ms"] for r in ok_cold if r["init_duration_ms"] is not None]),
        "cold_total_ms": cold_summary,
        "warm_embed_ms": warm_embed_summary,
        "warm_wall_ms": _summarize([r["wall_ms"] for r in ok_warm]),
        "max_memory_mb": max_mem,
        "max_memory_pct": mem_pct,
        "pass_bar": passed,
        "cold_records": cold_records,
        "warm_records": warm_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--vpc-id", required=True)
    ap.add_argument("--subnets", required=True, help="Comma-separated subnet IDs")
    ap.add_argument("--tag", dest="tags", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--model-full", required=True)
    ap.add_argument("--model-stripped", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cold-runs", type=int, default=5)
    ap.add_argument("--warm-runs", type=int, default=20)
    args = ap.parse_args()

    region = args.region
    subnets = [s.strip() for s in args.subnets.split(",") if s.strip()]
    tag_kv = {**dict(t.split("=", 1) for t in args.tags), "ManagedBy": "clync-bench"}
    tag_list = [{"Key": k, "Value": v} for k, v in tag_kv.items()]

    sts = boto3.client("sts", region_name=region)
    account = sts.get_caller_identity()["Account"]
    run_id = time.strftime("%Y%m%d%H%M%S")
    prefix = f"clync-embed-bench-{run_id}"
    bucket = f"{prefix}-{account}"
    role_name = f"{prefix}-role"
    sg_name = f"{prefix}-sg"
    ecr_repo = prefix
    fn_full = f"{prefix}-full-s3"
    fn_strip = f"{prefix}-strip-s3"
    fn_img = f"{prefix}-strip-img"

    s3 = boto3.client("s3", region_name=region)
    iam = boto3.client("iam", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)
    lam = boto3.client("lambda", region_name=region)
    ecr = boto3.client("ecr", region_name=region)
    logs = boto3.client("logs", region_name=region)

    queries_raw = json.loads(Path(args.queries_file).read_text())
    queries = queries_raw[:25]
    # Keep Docker bind-mount build dir under $HOME (Colima mounts /Users, not /tmp)
    build_dir = Path.home() / ".local/share/clync/bench/tmp_build" / prefix
    build_dir.mkdir(parents=True, exist_ok=True)

    created = {"bucket": None, "role": None, "sg": None, "ecr": None, "fns": []}
    results: dict = {"run_id": run_id, "region": region, "arms": {}}

    try:
        # 1. Build zip bundle locally before creating AWS resources
        zip_path = build_zip_bundle(build_dir)

        # 2. Create S3 bucket + upload models + zip bundle
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
        created["bucket"] = bucket
        s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": tag_list})
        print(f"Uploading models + zip to s3://{bucket}...", flush=True)
        s3.upload_file(args.model_full, bucket, "models/full.onnx")
        s3.upload_file(args.model_stripped, bucket, "models/stripped.onnx")
        s3.upload_file(args.tokenizer, bucket, "models/tokenizer.json")
        s3.upload_file(str(zip_path), bucket, "code/bundle.zip")

        # 3. Create IAM execution role
        trust = {
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}],
        }
        role_arn = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust), Tags=tag_list)["Role"]["Arn"]
        created["role"] = role_name
        iam.attach_role_policy(RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole")
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="s3-read-models",
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject", "s3:HeadObject", "s3:ListBucket"], "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]}],
            }),
        )

        # 4. Create throwaway egress-only SG in the VPC
        sg_id = ec2.create_security_group(
            GroupName=sg_name,
            Description="Temporary clync embedding benchmark SG",
            VpcId=args.vpc_id,
            TagSpecifications=[{"ResourceType": "security-group", "Tags": tag_list}],
        )["GroupId"]
        created["sg"] = sg_id

        # Wait for IAM role propagation
        time.sleep(10)

        vpc_cfg = {"SubnetIds": subnets, "SecurityGroupIds": [sg_id]}

        # 5. Arm 1: Full int8 S3-to-RAM zip across 3008 / 5120 / 10240 MB
        env_full = {
            "LOAD_MODE": "s3_ram",
            "MODEL_BUCKET": bucket,
            "MODEL_KEY": "models/full.onnx",
            "TOKENIZER_KEY": "models/tokenizer.json",
            "EAGER_INIT": "1",
        }
        lam.create_function(
            FunctionName=fn_full,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_handler.handler",
            Code={"S3Bucket": bucket, "S3Key": "code/bundle.zip"},
            Timeout=120,
            MemorySize=3008,
            Architectures=["arm64"],
            VpcConfig=vpc_cfg,
            Environment={"Variables": env_full},
            Tags=tag_kv,
        )
        created["fns"].append(fn_full)
        wait_lambda_active(lam, fn_full)

        arm1 = {}
        for mem in (3008, 5120, 10240):
            print(f"=== Arm 1 (Full int8 S3-to-RAM) @ {mem} MB ===", flush=True)
            arm1[str(mem)] = measure_fn(lam, fn_full, env_full, mem, queries, args.cold_runs, args.warm_runs)
        results["arms"]["arm1_full_s3_ram"] = arm1

        # Pick best tier from Arm 1 (lowest memory that passed, else fastest cold that succeeded, else 5120)
        passing = [m for m in (3008, 5120, 10240) if arm1[str(m)]["pass_bar"]]
        if passing:
            best_m = passing[0]
        else:
            ok_tiers = [m for m in (3008, 5120, 10240) if arm1[str(m)]["cold_ok"] > 0]
            best_m = min(ok_tiers, key=lambda m: arm1[str(m)]["cold_total_ms"]["p95"]) if ok_tiers else 5120
        lower_map = {10240: 5120, 5120: 3008, 3008: 2048}
        arm2_tiers = sorted({best_m, lower_map[best_m]})

        # 6. Arm 2: ColBERT-stripped int8 S3-to-RAM zip
        env_strip = {
            "LOAD_MODE": "s3_ram",
            "MODEL_BUCKET": bucket,
            "MODEL_KEY": "models/stripped.onnx",
            "TOKENIZER_KEY": "models/tokenizer.json",
            "EAGER_INIT": "1",
        }
        lam.create_function(
            FunctionName=fn_strip,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_handler.handler",
            Code={"S3Bucket": bucket, "S3Key": "code/bundle.zip"},
            Timeout=120,
            MemorySize=arm2_tiers[0],
            Architectures=["arm64"],
            VpcConfig=vpc_cfg,
            Environment={"Variables": env_strip},
            Tags=tag_kv,
        )
        created["fns"].append(fn_strip)
        wait_lambda_active(lam, fn_strip)

        arm2 = {}
        for mem in arm2_tiers:
            print(f"=== Arm 2 (Stripped int8 S3-to-RAM) @ {mem} MB ===", flush=True)
            arm2[str(mem)] = measure_fn(lam, fn_strip, env_strip, mem, queries, args.cold_runs, args.warm_runs)
        results["arms"]["arm2_stripped_s3_ram"] = arm2

        # Pick best tier from Arm 2 for Arm 3 container image control
        passing2 = [m for m in arm2_tiers if arm2[str(m)]["pass_bar"]]
        best_m2 = passing2[0] if passing2 else best_m

        # 7. Arm 3: Container image control in throwaway ECR repo
        ecr_uri = ecr.create_repository(repositoryName=ecr_repo, tags=tag_list)["repository"]["repositoryUri"]
        created["ecr"] = ecr_repo
        image_uri = build_and_push_image(build_dir, Path(args.model_stripped), Path(args.tokenizer), ecr_uri, region)

        env_img = {
            "LOAD_MODE": "local_disk",
            "MODEL_PATH": "/var/task/model/bge_m3_int8_dense_sparse.onnx",
            "TOKENIZER_PATH": "/var/task/model/tokenizer.json",
            "EAGER_INIT": "1",
        }
        lam.create_function(
            FunctionName=fn_img,
            PackageType="Image",
            Role=role_arn,
            Code={"ImageUri": image_uri},
            Timeout=120,
            MemorySize=best_m2,
            Architectures=["arm64"],
            VpcConfig=vpc_cfg,
            Environment={"Variables": env_img},
            Tags=tag_kv,
        )
        created["fns"].append(fn_img)
        wait_lambda_active(lam, fn_img)

        print(f"=== Arm 3 (Stripped int8 Container Image) @ {best_m2} MB ===", flush=True)
        results["arms"]["arm3_stripped_container"] = {
            str(best_m2): measure_fn(lam, fn_img, env_img, best_m2, queries, args.cold_runs, args.warm_runs)
        }
        # Also test 10240 MB container image if best_m2 != 10240 so we see upper bound on container init
        if best_m2 != 10240:
            print("=== Arm 3 (Stripped int8 Container Image) @ 10240 MB ===", flush=True)
            results["arms"]["arm3_stripped_container"]["10240"] = measure_fn(
                lam, fn_img, env_img, 10240, queries, args.cold_runs, args.warm_runs
            )

        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=1))
        print(f"Saved results to {args.out_json}", flush=True)

    finally:
        print("=== TEARDOWN ===", flush=True)
        for fn in created["fns"]:
            try:
                lam.delete_function(FunctionName=fn)
                print(f"Deleted Lambda {fn}", flush=True)
            except ClientError as e:
                print(f"Teardown Lambda {fn} warn: {e}", flush=True)
            try:
                logs.delete_log_group(logGroupName=f"/aws/lambda/{fn}")
                print(f"Deleted log group /aws/lambda/{fn}", flush=True)
            except ClientError:
                pass

        if created["ecr"]:
            try:
                ecr.delete_repository(repositoryName=created["ecr"], force=True)
                print(f"Deleted ECR repo {created['ecr']}", flush=True)
            except ClientError as e:
                print(f"Teardown ECR warn: {e}", flush=True)

        if created["bucket"]:
            try:
                pag = s3.get_paginator("list_objects_v2")
                for page in pag.paginate(Bucket=created["bucket"]):
                    objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                    if objs:
                        s3.delete_objects(Bucket=created["bucket"], Delete={"Objects": objs})
                s3.delete_bucket(Bucket=created["bucket"])
                print(f"Deleted S3 bucket {created['bucket']}", flush=True)
            except ClientError as e:
                print(f"Teardown S3 warn: {e}", flush=True)

        if created["role"]:
            try:
                iam.delete_role_policy(RoleName=created["role"], PolicyName="s3-read-models")
            except ClientError:
                pass
            try:
                iam.detach_role_policy(RoleName=created["role"], PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole")
            except ClientError:
                pass
            try:
                iam.delete_role(RoleName=created["role"])
                print(f"Deleted IAM role {created['role']}", flush=True)
            except ClientError as e:
                print(f"Teardown IAM role warn: {e}", flush=True)

        if created["sg"]:
            # Wait for Hyperplane ENIs from deleted VPC Lambdas to release before deleting SG
            for attempt in range(40):
                try:
                    ec2.delete_security_group(GroupId=created["sg"])
                    print(f"Deleted SG {created['sg']}", flush=True)
                    break
                except ClientError as e:
                    if e.response["Error"]["Code"] == "DependencyViolation" and attempt < 39:
                        time.sleep(10)
                    else:
                        print(f"Teardown SG warn: {e}", flush=True)
                        break

        subprocess.run(["rm", "-rf", str(build_dir)], check=False)


if __name__ == "__main__":
    main()
