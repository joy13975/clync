---
name: clync-connect
description: "Connect a clync-share Lambda ARN so this assistant can ask a teammate's shared archive (`ask_<alias>`). Automatically discovers the matching AWS profile for the ARN's account from `~/.aws/config`, runs the live IAM handshake, and saves `$CLYNC_DATA_HOME/shares.json`. Invoke for `/clync-connect <lambda-arn>`."
allowed-tools: Bash
---

Connect a `clync-share` Lambda ARN passed as the argument (e.g. `arn:aws:lambda:<region>:<account>:function:clync-share-<owner>`).

1. Run `clync connect <lambda-arn>` directly — do **not** ask the user for an AWS profile.
   - `clync connect` parses the 12-digit account ID from the Lambda ARN, automatically picks the matching AWS profile from `~/.aws/config` (preferring PowerUser, then Admin, then other roles), runs the live IAM handshake, and persists the resolved profile into `$CLYNC_DATA_HOME/shares.json`.
   - If `clync` is not on PATH, run `uv run --project <repo> python <repo>/clync.py connect <lambda-arn>` (where `<repo>` is the resolved parent directory of this skill).
   - If `clync` is not installed at all, run `uvx --from "git+https://github.com/joy13975/clync#subdirectory=share-client" clync-share connect <lambda-arn>`.
2. If the handshake fails because the auto-detected AWS SSO session is expired, tell the user the exact profile name and `aws sso login --profile <profile>` command to refresh it, or retry once refreshed.
3. Pass `--profile <p>` only if the user explicitly asked for a specific profile or if `~/.aws/config` has no profile matching that AWS account.
4. Report the connected alias, resolved AWS profile, granted modes (`ask` or `ask, owner`), and note:
   - `clync ask <alias> "<question>"` works immediately in the current session.
   - Reload/restart the MCP client to expose the native `ask_<alias>` MCP tool.
