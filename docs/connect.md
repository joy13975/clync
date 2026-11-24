# Connect an Agent to `clync-share`

Agent-executable guide to install or update `clync` on this host and connect a
`clync-share` Lambda ARN (`arn:aws:lambda:<region>:<account>:function:clync-share-<owner>`).

Do **not** ask the user for an AWS profile. Both `clync connect` and `clync-share connect`
parse the 12-digit AWS account ID from the Lambda ARN and auto-select the matching profile
from `~/.aws/config` (preferring PowerUser, then Admin, then other roles).

---

## Step 1 — Detect whether `clync` is already installed

Check if an existing `clync` checkout is linked on this host:

```sh
readlink ~/.local/bin/clync ~/.claude/skills/clync 2>/dev/null || command -v clync || true
```

- **Installed** (a local `clync` repo path was found) → try **Path A** first.
- **Not installed** (no output) → follow **Path B** (lightweight client-only install:
  `boto3` + `mcp` only, zero PyTorch/Postgres/Chrome setup). If the user explicitly asked
  for full local conversation archiving too, see [README.md](../README.md) after connecting.

---

## Path A — `clync` is already installed (update + connect)

Let `<REPO>` be the resolved `clync` repository directory (parent of `clync` or `skill/`).

1. **Check working tree and remote:**
   ```sh
   git -C "<REPO>" status --short
   git -C "<REPO>" remote -v
   ```
   - Ensure `https://github.com/joy13975/clync.git` (or `git@github.com:joy13975/clync.git`) is fetched.
   - If the working tree has uncommitted changes, **do not clobber them** — switch immediately to **Path B** so the share endpoint connects right now, and mention the untouched local changes in your final report.

2. **Pull latest `main` and sync dependencies:**
   ```sh
   git -C "<REPO>" fetch origin main
   git -C "<REPO>" pull --ff-only origin main
   uv sync --quiet --project "<REPO>"
   ```
   - If `pull --ff-only` fails (diverged/rewritten upstream history), **do not force-reset without asking** — switch immediately to **Path B** so the share endpoint connects right now, and mention in your final report that `<REPO>` diverged from `origin/main` and can be realigned later with `git -C "<REPO>" checkout -B main origin/main && uv sync --project "<REPO>"`.

3. **Connect the Lambda ARN** (auto-links `~/.claude/skills/clync-connect` when `~/.claude/skills/clync` points to `<REPO>/skill`):
   ```sh
   uv run --project "<REPO>" python "<REPO>/clync.py" connect "<LAMBDA_ARN>"
   ```

4. **Remove any redundant standalone `clync-share` MCP server** (since updated `clync` MCP now relays `shares.json` directly):
   ```sh
   claude mcp remove --scope user clync-share 2>/dev/null || true
   ```

5. **Verify:**
   ```sh
   uv run --project "<REPO>" python "<REPO>/clync.py" shares
   ```

---

## Path B — Lightweight client-only connect (no `clync` install, or local checkout dirty/diverged)

1. **Ensure `uvx` is installed:**
   ```sh
   command -v uvx || brew install uv
   ```

2. **Install/refresh `clync-share` and connect the Lambda ARN:**
   ```sh
   uvx --refresh --from "git+https://github.com/joy13975/clync#subdirectory=share-client" \
     clync-share connect "<LAMBDA_ARN>"
   ```
   This single command:
   - auto-detects the matching AWS profile from `~/.aws/config` for the ARN's account ID;
   - runs a live SigV4 STS IAM handshake against `<LAMBDA_ARN>`;
   - saves `~/.local/share/clync/shares.json` (`0600`);
   - registers the `clync-share` stdio MCP server in Claude Code (`--scope user`).

3. **If the host uses Codex CLI or OpenCode instead of Claude Code**, register the MCP server command:
   ```text
   uvx --from "git+https://github.com/joy13975/clync#subdirectory=share-client" clync-share serve
   ```

4. **Verify:**
   ```sh
   uvx --from "git+https://github.com/joy13975/clync#subdirectory=share-client" clync-share list
   ```
   If `claude` is installed, verify the MCP server connects cleanly (`claude mcp list` shows `clync-share` connected).

---

## Handling AWS authentication errors

- **Expired AWS SSO session:** the error names the auto-detected profile (e.g. `using auto-detected AWS profile '<profile>'`). Have the user refresh SSO (`aws sso login --profile <profile>`) and re-run the `connect` command.
- **No matching profile in `~/.aws/config`:** tell the user which 12-digit AWS account ID is in `<LAMBDA_ARN>` and ask which AWS profile to pass with `--profile <name>`.

---

## What to report when done

State:
1. Connected alias, resolved AWS profile, granted modes (`ask` or `ask, owner`), and whether Path A (full `clync` updated) or Path B (standalone `clync-share` fallback, plus why Path A was skipped if applicable) ran.
2. Immediate CLI usage works now without restarting:
   - Path A: `clync ask <alias> "<question>"`
   - Path B: `uvx --from "git+https://github.com/joy13975/clync#subdirectory=share-client" clync-share ask <alias> "<question>"`
3. Restart / open a new assistant session to expose the native `ask_<alias>` MCP tool.
