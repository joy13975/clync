# ADR 0006 — Serverless per-owner `clync-share` over direct Lambda invoke

- **Status:** Accepted
- **Date:** 2026-11-23
- **Scope:** Opt-in AWS serverless deployment for remote owner MCP access and read-only, live-redacted teammate Q&A (`share-client/`, `clync_cipher.py`, `cloud/`).

## Context

Users want an opt-in serverless AWS mode for `clync` with near-zero idle cost and two optional interfaces:
1. **Owner full MCP:** Remote read/search/citation-resolution over the owner's own archive.
2. **Read-only "ask this owner" MCP (`clync-share`):** Lets teammates ask work questions against the owner's archive without exposing verbatim transcripts, profanity/scolding, private non-work topics, or unmasked secrets, while providing opaque citation handles (`clync-<owner>-<rand>`) that only the owner can decipher back to raw transcript turns.

Shared dev databases are accessible to team admins, so storing plaintext transcripts or unpermuted sparse vectors (which reconstruct lexical tokens) in RDS is unacceptable.

## Decision

### 1. Transport & IAM authentication: direct `lambda:InvokeFunction` + SigV4 STS proof

Instead of ALB, API Gateway, Cognito, or bastion IP allowlists, clients invoke each owner's Lambda directly via `lambda:InvokeFunction`. Because Lambda invoke does not forward caller IAM identity inside the event payload, every call includes a fresh (<90 s) **SigV4-signed `sts:GetCallerIdentity` request**. The Lambda replays that signed request to AWS STS (~95 ms) and checks:
- Caller account matches `CLYNC_ALLOWED_ACCOUNT_ID`.
- **Owner mode (`ENABLE_OWNER_MCP`):** Caller `aws:userid` is in `CLYNC_OWNER_USER_IDS`.
- **Read-only ask mode (`ENABLE_SHARE_MCP`):** Caller ARN matches `CLYNC_ALLOWED_ROLE_PREFIXES` (e.g., team PowerUser / Admin SSO roles) or is the owner.

### 2. Compute & embedder: 10,240 MB arm64 ECR Container Image Lambda

Lambda CPU embedding benchmarks showed S3-to-RAM zip Lambdas hit a ~6 s S3 download floor and +500 MB RAM overhead across memory tiers. A **10,240 MB arm64 ECR container image** baking ColBERT-stripped `gpahal/bge-m3-onnx-int8` (`542.55 MB`) with `onnxruntime==1.32.0` (1.21.0 crashes on arm64 Firecracker `cpuinfo`) achieves **3.8 s p50 / 5.4 s p95 cold**, **38.8 ms p50 / 43.3 ms p95 warm**, **17% peak RAM**, and **~$0.075/mo** idle ECR storage.

### 3. Storage & lite cipher (`clync_cipher.py`)

Each owner has an isolated schema (`clync_<owner>`, IAM-authenticated `rds_iam` role) inside a separate database `clync` on the shared dev Aurora PostgreSQL 17 cluster (keeping it out of nightly app-DB dumps). Before data leaves the owner's laptop:
- **Key management:** One KMS CMK per owner (`$1.00/mo`) with **no account-root policy statement**; `kms:*` requires the owner's SSO `aws:userid`, and `kms:Decrypt` is granted only to the owner's SSO `aws:userid` and the owner's Lambda execution role.
- **Text & metadata:** AES-256-GCM with AAD bound to `<table>:<id>` (`chunks`, `messages`, `units`, `citations`).
- **Dense vectors:** Multiplied by a per-owner deterministic 1024×1024 orthogonal matrix $Q$ ($Q^\top Q = I$), preserving cosine similarity and `halfvec(1024)` HNSW rankings while hiding raw coordinates.
- **Sparse vectors:** Per-owner bijective permutation over the 250,002 XLM-R token IDs, preserving inner products and `sparsevec` HNSW rankings while preventing lexical token reconstruction.
- **Facets:** Domain-separated HMAC-SHA256 digests (`source`, `repo`, `project`, `sender`, `lang`).
- **Bulk push:** Encrypted gzipped NDJSON batches upload to a per-owner prefix in a 1-day-lifecycle S3 staging bucket and are ingested and deleted by the owner's Lambda.

### 4. Query-time redaction & opaque citation handles

Verbatim text is decrypted only in Lambda RAM during a query (never pre-paraphrased at ingest, preserving full recall resolution). For `ask`:
1. The Lambda verifies Bedrock model invocation logging is **OFF** in the configured Bedrock region (`--bedrock-region`, chosen because its invocation logging is off) before every call and fails loud if enabled.
2. It calls the configured Bedrock model (`--bedrock-model-id`) to synthesize a non-verbatim, redacted answer citing temporary slots `[S1]`, `[S2]`.
3. Each cited slot is replaced with a fresh random opaque handle `clync-<owner>-<12-hex>`, whose mapping to `(unit_id, chunk_id, msg_idx, caller_arn, question)` is stored AES-GCM encrypted in `citations`. Only the owner can call `resolve_citation` to rehydrate the underlying raw turns.

### 5. Tamper-evident alerting

An EventBridge rule watches CloudTrail for unexpected `kms:Decrypt` (outside the owner's Lambda role and owner `aws:userid`), `kms:PutKeyPolicy` / `ScheduleKeyDeletion` / `DisableKey`, `lambda:UpdateFunctionCode` / `UpdateFunctionConfiguration`, `events:DisableRule` / `DeleteRule` / `RemoveTargets`, `logs:DeleteSubscriptionFilter`, and `bedrock:PutModelInvocationLoggingConfiguration`. A lightweight alert Lambda emits a structured `ERROR` JSON line (`@event:clync_tamper_alert`), which `--log-forwarder-arn` can subscribe to an existing CloudWatch Logs forwarder for alerting.

### 6. Zero-install client onboarding (`clync connect` / `share-client/`)

- Existing `clync` users run `clync connect <lambda-arn>` (or `/clync-connect <lambda-arn>`), which auto-selects the matching AWS profile for the ARN's account from `~/.aws/config`, executes a live IAM handshake, saves `$CLYNC_DATA_HOME/shares.json`, and exposes the connected endpoints inside `mcp_server.py`.
- Teammates without `clync` installed run the lightweight `share-client/` package (`mcp` + `boto3` only, zero PyTorch/Postgres):
  ```sh
  uvx --from "git+https://github.com/joy13975/clync#subdirectory=share-client" \
    clync-share connect <lambda-arn>
  ```

## Consequences

- Steady-state idle cost is **~$1.09/mo per owner** ($1.00 KMS CMK + ~$0.075 ECR image + ~$0.01 S3 bucket).
- Protection is **tamper-evident**, not hardware-enclave tamper-proof: any admin or PowerUser who modifies the Lambda or key policy to read ciphertext triggers a loud CloudTrail tamper alert.
- Every taggable resource carries `Component`, `Owner`, and any `--tag KEY=VALUE` pairs (for example a cost-allocation tag); log groups keep 90 days.
