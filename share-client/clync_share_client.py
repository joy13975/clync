"""Lightweight IAM-authenticated client & stdio MCP relay for clync-share Lambdas.

Zero PyTorch or Postgres dependencies — usable standalone via `uvx` by teammates
without `clync` installed, and imported directly by `clync` for unified local +
cloud MCP access.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

LAMBDA_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:lambda:(?P<region>[a-z0-9-]+):(?P<account>\d{12}):function:(?P<name>[A-Za-z0-9-_]+)(?::(?P<qualifier>[A-Za-z0-9-_$]+))?$"
)
ROLE_ARN_RE = re.compile(r"^arn:aws[a-z-]*:iam::(?P<account>\d{12}):role/(?P<role>.+)$")
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
SHARES_FILENAME = "shares.json"
MCP_SERVER_NAME = "clync-share"
PUBLIC_REPO_URL = "git+https://github.com/joy13975/clync#subdirectory=share-client"


def data_home() -> Path:
    return Path(os.environ.get("CLYNC_DATA_HOME", Path.home() / ".local/share/clync"))


def shares_path() -> Path:
    return data_home() / SHARES_FILENAME


def parse_lambda_arn(arn: str) -> dict[str, str]:
    match = LAMBDA_ARN_RE.match(arn.strip())
    if not match:
        raise ValueError(
            f"invalid Lambda ARN {arn!r}: expected arn:aws:lambda:<region>:<account>:function:<name>"
        )
    return {
        "arn": arn.strip(),
        "region": match.group("region"),
        "account": match.group("account"),
        "function_name": match.group("name"),
        "qualifier": match.group("qualifier") or "",
    }


def default_alias_for(function_name: str, owner: str | None = None) -> str:
    raw = (owner or function_name).lower()
    if raw.startswith("clync-share-"):
        raw = raw[len("clync-share-") :]
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-_")
    return cleaned[:32] or "share"


def validate_alias(alias: str) -> str:
    if not ALIAS_RE.match(alias):
        raise ValueError(f"invalid alias {alias!r}: must match {ALIAS_RE.pattern}")
    return alias


def load_shares() -> dict[str, Any]:
    path = shares_path()
    if not path.exists():
        return {"default_owner": None, "endpoints": {}}
    data = json.loads(path.read_text())
    data.setdefault("default_owner", None)
    data.setdefault("endpoints", {})
    return data


def save_shares(data: dict[str, Any]) -> None:
    path = shares_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def build_sts_proof(region: str, profile: str | None = None) -> dict[str, Any]:
    """Build a short-lived SigV4-signed sts:GetCallerIdentity request for Lambda verification."""
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    session = boto3.Session(profile_name=profile, region_name=region)
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError(
            f"no AWS credentials found (profile={profile!r}, region={region!r})"
        )
    frozen = creds.get_frozen_credentials()
    host = f"sts.{region}.amazonaws.com"
    url = f"https://{host}/?Action=GetCallerIdentity&Version=2011-06-15"
    req = AWSRequest(method="GET", url=url, headers={"Host": host})
    SigV4Auth(frozen, "sts", region).add_auth(req)
    return {"url": url, "headers": dict(req.headers.items()), "region": region}


def invoke_share_lambda(
    lambda_arn: str,
    action: str,
    params: dict[str, Any] | None = None,
    *,
    profile: str | None = None,
    region: str | None = None,
    read_timeout: int = 120,
) -> dict[str, Any]:
    """Invoke a clync-share Lambda with a fresh SigV4 STS caller proof."""
    import boto3
    from botocore.config import Config

    parsed = parse_lambda_arn(lambda_arn)
    use_region = region or parsed["region"]
    session = boto3.Session(profile_name=profile, region_name=use_region)
    client = session.client(
        "lambda",
        config=Config(read_timeout=max(60, int(read_timeout)), retries={"max_attempts": 1}),
    )
    payload = {
        "action": action,
        "params": params or {},
        "auth": build_sts_proof(use_region, profile=profile),
    }
    resp = client.invoke(
        FunctionName=parsed["arn"],
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    raw = resp["Payload"].read().decode("utf-8")
    if resp.get("FunctionError"):
        raise RuntimeError(f"clync-share Lambda error ({resp['FunctionError']}): {raw}")
    data = json.loads(raw)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"clync-share rejected {action!r}: {data['error']}")
    return data


def _profile_priority(name: str, role_hint: str) -> tuple[int, str]:
    """Rank matching AWS profiles for clync-share: PowerUser first, Admin second, ReadOnly last."""
    text = f"{name} {role_hint}".lower()
    if "readonly" in text or re.search(r"(^|[-_/])ro($|[-_/])", name.lower()):
        return (3, name)
    if "poweruser" in text or re.search(r"(^|[-_/])pu($|[-_/])", name.lower()):
        return (0, name)
    if "admin" in text:
        return (1, name)
    return (2, name)


def list_profiles_for_account(account_id: str, config_path: Path | None = None) -> list[str]:
    """Return AWS CLI profiles configured for `account_id`, ordered by invoke suitability."""
    cfg_file = config_path or Path(os.environ.get("AWS_CONFIG_FILE", Path.home() / ".aws/config"))
    if not cfg_file.exists():
        return []
    parser = configparser.RawConfigParser()
    parser.read(cfg_file)
    matches: list[tuple[tuple[int, str], str]] = []
    for section in parser.sections():
        if section == "default":
            prof_name = "default"
        elif section.startswith("profile "):
            prof_name = section[len("profile ") :].strip()
        else:
            continue
        if not prof_name:
            continue
        sso_acct = parser.get(section, "sso_account_id", fallback="").strip()
        role_arn = parser.get(section, "role_arn", fallback="").strip()
        role_match = ROLE_ARN_RE.match(role_arn)
        acct = sso_acct or (role_match.group("account") if role_match else "")
        if acct != account_id:
            continue
        role_hint = parser.get(section, "sso_role_name", fallback="") or (
            role_match.group("role") if role_match else ""
        )
        matches.append((_profile_priority(prof_name, role_hint), prof_name))
    matches.sort(key=lambda item: item[0])
    return [name for _, name in matches]


def resolve_profile_for_account(account_id: str, config_path: Path | None = None) -> str | None:
    profiles = list_profiles_for_account(account_id, config_path=config_path)
    return profiles[0] if profiles else None


def connect_endpoint(
    lambda_arn: str,
    *,
    alias: str | None = None,
    profile: str | None = None,
    set_default: bool = False,
) -> dict[str, Any]:
    """Verify IAM access via handshake and persist endpoint to shares.json.

    When `profile` is omitted and `AWS_PROFILE` is unset, automatically picks a
    matching AWS profile for the Lambda ARN's account from `~/.aws/config`.
    """
    parsed = parse_lambda_arn(lambda_arn)
    candidates: list[str | None]
    if profile:
        candidates = [profile]
    elif os.environ.get("AWS_PROFILE"):
        candidates = [os.environ["AWS_PROFILE"]]
    else:
        matched = list_profiles_for_account(parsed["account"])
        candidates = list(matched) if matched else [None]

    chosen_profile: str | None = candidates[0]
    info: dict[str, Any] | None = None
    last_exc: Exception | None = None
    for idx, cand in enumerate(candidates):
        try:
            info = invoke_share_lambda(parsed["arn"], "handshake", {}, profile=cand, region=parsed["region"])
            chosen_profile = cand
            break
        except Exception as exc:
            last_exc = exc
            # Only try another matching profile on authorization refusal; fail loud on expired SSO or network errors.
            msg = str(exc)
            is_authz_denial = (
                "AccessDeniedException" in msg
                or "not authorized" in msg
                or "rejected 'handshake'" in msg
            )
            if not is_authz_denial or idx == len(candidates) - 1:
                hint = f" (using auto-detected AWS profile {cand!r})" if (not profile and cand) else ""
                raise RuntimeError(f"clync-share connect failed{hint}: {exc}") from exc
    assert info is not None, last_exc

    resolved_alias = validate_alias(alias or default_alias_for(parsed["function_name"], info.get("owner")))
    entry = {
        "alias": resolved_alias,
        "arn": parsed["arn"],
        "region": parsed["region"],
        "profile": chosen_profile,
        "owner": info.get("owner", resolved_alias),
        "modes": info.get("modes", []),
        "tools": info.get("tools", []),
        "caller_arn": info.get("caller_arn"),
    }
    cfg = load_shares()
    cfg["endpoints"][resolved_alias] = entry
    if set_default:
        if "owner" not in entry["modes"]:
            raise RuntimeError(
                f"cannot set {resolved_alias!r} as default owner store: caller only has modes {entry['modes']}"
            )
        cfg["default_owner"] = resolved_alias
    save_shares(cfg)
    return entry


def disconnect_endpoint(alias: str) -> bool:
    cfg = load_shares()
    removed = cfg["endpoints"].pop(alias, None) is not None
    if cfg.get("default_owner") == alias:
        cfg["default_owner"] = None
    if removed:
        save_shares(cfg)
    return removed


def _register_claude_mcp() -> bool:
    claude = shutil.which("claude")
    if not claude:
        return False
    cmd = [
        claude,
        "mcp",
        "add",
        "--scope",
        "user",
        MCP_SERVER_NAME,
        "--",
        "uvx",
        "--from",
        PUBLIC_REPO_URL,
        "clync-share",
        "serve",
    ]
    subprocess.run([claude, "mcp", "remove", "--scope", "user", MCP_SERVER_NAME],
                   capture_output=True, text=True, check=False)
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"`claude mcp add` failed: {res.stderr.strip()}")
    return True


def _format_ask_response(owner: str, data: dict[str, Any]) -> str:
    lines = [data.get("answer", "").strip()]
    citations = data.get("citations") or []
    if citations:
        lines.append("\nSources:")
        for c in citations:
            facts = [c.get("provenance"), c.get("source_type"), c.get("date")]
            lines.append(f"- [{c['handle']}] " + ", ".join(f for f in facts if f))
    return "\n".join(lines).strip()


def register_share_tools(
    mcp_app,
    *,
    endpoints: dict[str, dict[str, Any]] | None = None,
    include_owner_tools: bool = True,
) -> list[str]:
    """Attach MCP tools for every configured clync-share endpoint onto `mcp_app`."""
    cfg = load_shares()
    eps = endpoints if endpoints is not None else cfg.get("endpoints", {})
    registered: list[str] = []

    def _add_ask_tool(alias: str, ep: dict[str, Any]) -> None:
        owner = ep.get("owner", alias)
        tool_name = f"ask_{alias.replace('-', '_')}"

        def _ask(question: str, limit: int = 8) -> str:
            res = invoke_share_lambda(
                ep["arn"],
                "ask",
                {"question": question, "limit": limit},
                profile=ep.get("profile"),
                region=ep.get("region"),
            )
            return _format_ask_response(owner, res)

        _ask.__name__ = tool_name
        _ask.__doc__ = (
            f"Ask what {owner} would say or decide, inferred from {owner}'s past decisions in "
            f"{owner}'s shared work archive. Returns {owner}'s call, the reasons, the past decisions "
            f"it rests on, and a one-line basis; each source names its provenance. Relay that answer "
            f"as {owner}'s call in its own structure. Do not add caveats about missing records, "
            f"inference, or source handles."
        )
        mcp_app.add_tool(_ask)
        registered.append(tool_name)

    owner_eps: dict[str, dict[str, Any]] = {}
    for alias, ep in sorted(eps.items()):
        modes = ep.get("modes") or ["ask"]
        if "ask" in modes:
            _add_ask_tool(alias, ep)
        if include_owner_tools and "owner" in modes:
            owner_eps[alias] = ep

    if include_owner_tools and owner_eps:
        default_alias = cfg.get("default_owner") or next(iter(owner_eps))

        def _pick_owner(alias: str | None) -> dict[str, Any]:
            chosen = alias or default_alias
            if chosen not in owner_eps:
                raise ValueError(f"unknown owner endpoint {chosen!r}; configured: {sorted(owner_eps)}")
            return owner_eps[chosen]

        def resolve_citation(handle: str, alias: str | None = None) -> str:
            """Resolve an opaque `clync-<owner>-<rand>` citation handle back to raw transcript turns (owner only)."""
            target = None
            if alias:
                target = _pick_owner(alias)
            else:
                for ep in owner_eps.values():
                    if handle.startswith(f"clync-{ep.get('owner', ep['alias'])}-"):
                        target = ep
                        break
                target = target or _pick_owner(None)
            res = invoke_share_lambda(
                target["arn"],
                "resolve_citation",
                {"handle": handle},
                profile=target.get("profile"),
                region=target.get("region"),
            )
            return res.get("text", json.dumps(res, indent=2))

        def cloud_search_history(
            query: str,
            limit: int = 10,
            alias: str | None = None,
        ) -> str:
            """Search the owner's encrypted cloud clync archive (insights + raw transcripts)."""
            ep = _pick_owner(alias)
            res = invoke_share_lambda(
                ep["arn"],
                "search_history",
                {"query": query, "limit": limit},
                profile=ep.get("profile"),
                region=ep.get("region"),
            )
            return res.get("text", "")

        def cloud_search_transcripts(
            query: str,
            source: str = "all",
            repo: str | None = None,
            sender: str | None = None,
            limit: int = 10,
            per_unit: int = 1,
            alias: str | None = None,
        ) -> str:
            """Search the owner's encrypted cloud raw transcripts with facets."""
            ep = _pick_owner(alias)
            res = invoke_share_lambda(
                ep["arn"],
                "search_transcripts",
                {
                    "query": query,
                    "source": source,
                    "repo": repo,
                    "sender": sender,
                    "limit": limit,
                    "per_unit": per_unit,
                },
                profile=ep.get("profile"),
                region=ep.get("region"),
            )
            return res.get("text", "")

        def cloud_get_conversation(
            uuid: str,
            from_idx: int | None = None,
            to_idx: int | None = None,
            sender: str | None = None,
            alias: str | None = None,
        ) -> str:
            """Read decrypted turns from a conversation in the owner's cloud archive."""
            ep = _pick_owner(alias)
            res = invoke_share_lambda(
                ep["arn"],
                "get_conversation",
                {"uuid": uuid, "from_idx": from_idx, "to_idx": to_idx, "sender": sender},
                profile=ep.get("profile"),
                region=ep.get("region"),
            )
            return res.get("text", "")

        for fn in (resolve_citation, cloud_search_history, cloud_search_transcripts, cloud_get_conversation):
            mcp_app.add_tool(fn)
            registered.append(fn.__name__)

    return registered


def run_mcp_server(inline_endpoints: list[str] | None = None, profile: str | None = None) -> None:
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(MCP_SERVER_NAME)
    if inline_endpoints:
        eps = {}
        for spec in inline_endpoints:
            if "=" in spec:
                alias, arn = spec.split("=", 1)
                parsed = parse_lambda_arn(arn)
                alias = validate_alias(alias)
            else:
                parsed = parse_lambda_arn(spec)
                alias = default_alias_for(parsed["function_name"])
            eps[alias] = {
                "alias": alias,
                "arn": parsed["arn"],
                "region": parsed["region"],
                "profile": profile,
                "owner": alias,
                "modes": ["ask"],
            }
        register_share_tools(app, endpoints=eps, include_owner_tools=False)
    else:
        register_share_tools(app)
    app.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clync-share",
        description="IAM-authenticated MCP relay and CLI for clync-share Lambda endpoints",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conn = sub.add_parser("connect", help="verify IAM access and register a clync-share Lambda ARN")
    p_conn.add_argument("lambda_arn", help="Lambda function ARN (arn:aws:lambda:<region>:<account>:function:<name>)")
    p_conn.add_argument("--as", dest="alias", help="short alias for this endpoint (default: derived from owner/function)")
    p_conn.add_argument("--profile", help="AWS profile to use when invoking this endpoint")
    p_conn.add_argument("--default", dest="set_default", action="store_true",
                        help="set as default owner cloud store (requires owner mode)")
    p_conn.add_argument("--no-mcp", action="store_true", help="save config only; do not run `claude mcp add`")

    p_disc = sub.add_parser("disconnect", help="remove a configured clync-share endpoint")
    p_disc.add_argument("alias")

    sub.add_parser("list", help="list configured clync-share endpoints")

    p_ask = sub.add_parser("ask", help="ask a connected clync-share endpoint from the terminal")
    p_ask.add_argument("alias")
    p_ask.add_argument("question")
    p_ask.add_argument("--limit", type=int, default=8)

    p_serve = sub.add_parser("serve", help="run stdio MCP server relaying configured clync-share endpoints")
    p_serve.add_argument("--endpoint", action="append", default=[],
                         help="inline endpoint (<arn> or <alias>=<arn>); overrides shares.json")
    p_serve.add_argument("--profile", help="AWS profile for inline --endpoint values")

    args = parser.parse_args(argv)

    if args.cmd == "connect":
        entry = connect_endpoint(
            args.lambda_arn,
            alias=args.alias,
            profile=args.profile,
            set_default=args.set_default,
        )
        print(f"Connected {entry['alias']} ({entry['owner']}) -> {entry['arn']}")
        print(f"  modes  : {', '.join(entry['modes'])}")
        print(f"  profile: {entry['profile'] or '(default AWS credentials)'}")
        print(f"  caller : {entry['caller_arn']}")
        print(f"  config : {shares_path()}")
        if not args.no_mcp:
            if _register_claude_mcp():
                print(f"  MCP   : registered `{MCP_SERVER_NAME}` in Claude Code (reload session to use)")
            else:
                print(
                    f"  MCP   : `claude` not on PATH; register manually:\n"
                    f"          uvx --from '{PUBLIC_REPO_URL}' clync-share serve"
                )
        return 0

    if args.cmd == "disconnect":
        if disconnect_endpoint(args.alias):
            print(f"Disconnected {args.alias}")
            return 0
        print(f"No endpoint named {args.alias!r}", file=sys.stderr)
        return 1

    if args.cmd == "list":
        cfg = load_shares()
        eps = cfg.get("endpoints") or {}
        if not eps:
            print(f"No clync-share endpoints configured ({shares_path()}).")
            return 0
        for alias, ep in sorted(eps.items()):
            mark = " [default]" if cfg.get("default_owner") == alias else ""
            prof = f" profile={ep['profile']}" if ep.get("profile") else ""
            print(f"- {alias}{mark}: {ep['arn']} modes={','.join(ep.get('modes', []))}{prof}")
        return 0

    if args.cmd == "ask":
        cfg = load_shares()
        ep = cfg.get("endpoints", {}).get(args.alias)
        if not ep:
            print(f"Unknown endpoint {args.alias!r}; run `clync-share list`.", file=sys.stderr)
            return 1
        res = invoke_share_lambda(
            ep["arn"],
            "ask",
            {"question": args.question, "limit": args.limit},
            profile=ep.get("profile"),
            region=ep.get("region"),
        )
        print(_format_ask_response(ep.get("owner", args.alias), res))
        return 0

    if args.cmd == "serve":
        run_mcp_server(args.endpoint or None, profile=args.profile)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
