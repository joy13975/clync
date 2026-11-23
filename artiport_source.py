from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import clync


def config_path() -> Path:
    return Path(os.environ.get("CLYNC_ARTIPORT_CONFIG", clync.DATA_HOME / "artiport.json"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "Artiport API redirect is not allowed", headers, fp)


def _get(config: dict, path: str) -> dict:
    request = urllib.request.Request(
        config["url"].rstrip("/") + path,
        headers={"Authorization": "Bearer " + config["token"]},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=30) as response:
        return json.load(response)


def _delete(con, unit_id: str) -> None:
    con.execute("DELETE FROM units WHERE unit_id=%s AND source='artiport'", (unit_id,))


def _upsert(con, prefix: str, artifact: dict) -> bool:
    unit_id = prefix + artifact["id"]
    if artifact["state"] == "deleted":
        _delete(con, unit_id)
        return True
    revision = next(r for r in artifact["revisions"] if r["number"] == artifact["latest"])
    metadata = {key: artifact[key] for key in (
        "id", "title", "description", "project", "owner", "url", "state", "latest",
        "parent_id", "parent_revision",
    )}
    metadata["content_hash"] = revision["hash"]
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    previous = con.execute("SELECT raw FROM units WHERE unit_id=%s", (unit_id,)).fetchone()
    if previous and previous["raw"]["signature"] == signature:
        return False
    created = datetime.fromtimestamp(artifact["created"], UTC)
    updated = datetime.fromtimestamp(artifact["updated"], UTC)
    project = artifact["project"] or None
    project_id = prefix + "project:" + project if project else None
    con.execute(
        "INSERT INTO units(unit_id,kind,source,title,summary,project_uuid,project_name,"
        "created_at,updated_at,entrypoint,raw,synced_at) "
        "VALUES (%s,'artifact','artiport',%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT(unit_id) DO UPDATE SET title=excluded.title,summary=excluded.summary,"
        "project_uuid=excluded.project_uuid,project_name=excluded.project_name,"
        "updated_at=excluded.updated_at,entrypoint=excluded.entrypoint,raw=excluded.raw,"
        "synced_at=excluded.synced_at",
        (unit_id, artifact["title"], artifact["description"], project_id, project,
         created, updated, artifact["url"], clync._json({**metadata, "signature": signature}),
         clync._now()),
    )
    text = (
        f"{artifact['title']}\n{revision['url']}\n"
        f"Owner: {artifact['owner']} | State: {artifact['state']}\n"
        f"Parent: {artifact['parent_id']}@{artifact['parent_revision']}\n"
        f"{artifact['description']}\n\n{revision['search_text']}"
    )
    clync._replace_unit_messages(con, unit_id, [(
        "revision:" + str(revision["number"]), 0, "assistant", "artifact", text, None,
        updated, clync._json({"revision": revision["number"], "url": revision["url"]}),
    )])
    return True


def ingest(full: bool = False) -> dict:
    config = json.loads(config_path().read_text())
    inventory = _get(config, "/api/inventory")
    prefix = "artiport:" + inventory["instance_id"] + ":"
    key = prefix + "cursor"
    con = clync.connect()
    updated = removed = 0
    try:
        cursor = int(clync.get_meta(con, key) or 0)
        if full or cursor > inventory["cursor"]:
            cursor = 0
        while True:
            page = _get(config, "/api/sync?" + urlencode({"after": cursor, "limit": 100}))
            if page["instance_id"] != inventory["instance_id"]:
                raise RuntimeError("Artiport instance changed during sync; cursor was not advanced")
            for event in page["items"]:
                updated += _upsert(con, prefix, event["artifact"])
            if page["has_more"] and page["cursor"] <= cursor:
                raise RuntimeError("Artiport feed did not advance its pagination cursor")
            cursor = page["cursor"]
            clync.set_meta(con, key, str(cursor))
            con.commit()
            if not page["has_more"]:
                break
        final = _get(config, "/api/inventory")
        if final["instance_id"] != inventory["instance_id"]:
            raise RuntimeError("Artiport instance changed during reconciliation; nothing purged")
        live = {prefix + artifact_id for artifact_id in final["ids"]}
        stored = con.execute("SELECT unit_id FROM units WHERE source='artiport'").fetchall()
        for row in stored:
            if row["unit_id"] not in live:
                _delete(con, row["unit_id"])
                removed += 1
        con.commit()
    finally:
        con.close()
    stats = {"updated": updated, "reconciled_deletions": removed, "cursor": cursor}
    print("[artiport] " + json.dumps(stats))
    return stats
