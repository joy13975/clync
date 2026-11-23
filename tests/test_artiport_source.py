import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psycopg
import pytest

import artiport_source
import clync


def test_artifact_search_incremental_sync_and_missing_deletion_event(
    pg_test_db, mock_embed, tmp_path, monkeypatch
):
    source = {
        "instance_id": "test-instance", "cursor": 2, "ids": ["one", "two"],
        "items": [],
    }
    for number, title in enumerate(("Heliotrope findings", "Orchid notes"), 1):
        source["items"].append({"sequence": number, "artifact": {
            "id": source["ids"][number - 1], "owner": "task-a", "title": title,
            "description": "Botanical report", "project": "Research", "latest": 1,
            "state": "active", "created": 1800000000, "updated": 1800000000,
            "parent_id": None, "parent_revision": None,
            "url": f"https://artiport.example/a/{number}",
            "revisions": [{"number": 1, "hash": title, "search_text": title,
                           "url": f"https://artiport.example/a/{number}/r/1"}],
        }})

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer test-reader":
                self.send_error(401)
                return
            parsed = urlparse(self.path)
            if parsed.path == "/api/inventory":
                body = {k: source[k] for k in ("instance_id", "cursor", "ids")}
            else:
                after = int(parse_qs(parsed.query)["after"][0])
                pending = [e for e in source["items"] if e["sequence"] > after]
                body = {"instance_id": source["instance_id"], "items": pending[:1],
                        "cursor": pending[0]["sequence"] if pending else after,
                        "has_more": len(pending) > 1}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "artiport.json"
    config.write_text(json.dumps({
        "url": f"http://127.0.0.1:{server.server_port}", "token": "test-reader"
    }))
    monkeypatch.setenv("CLYNC_ARTIPORT_CONFIG", str(config))
    try:
        assert artiport_source.ingest()["updated"] == 2
        pg_test_db.build_index()
        matches = pg_test_db.hybrid_search("Heliotrope", source="artiport", project="Research")
        first = matches[0]
        assert "Heliotrope" in first["text"]
        assert "https://artiport.example/a/1/r/1" in first["text"]
        assert not pg_test_db.hybrid_search(
            "Heliotrope", exclude_attempted="retention-check"
        )
        assert artiport_source.ingest()["updated"] == 0
        archived = copy.deepcopy(source["items"][0])
        archived["sequence"] = 3
        archived["artifact"]["state"] = "archived"
        source["items"].append(archived)
        source["cursor"] = 3
        artiport_source.ingest()
        pg_test_db.build_index()
        assert "Heliotrope" in pg_test_db.hybrid_search("Heliotrope", source="artiport")[0]["text"]
        source["ids"] = ["two"]
        assert artiport_source.ingest()["reconciled_deletions"] == 1
        with clync.connect() as con:
            assert con.execute("SELECT * FROM units WHERE unit_id=%s", (first["unit_id"],)).fetchone() is None
            assert con.execute("SELECT * FROM messages WHERE unit_id=%s", (first["unit_id"],)).fetchone() is None
        assert all(hit["unit_id"] != first["unit_id"]
                   for hit in pg_test_db.hybrid_search("Heliotrope", source="artiport"))
        assert "Orchid notes" in pg_test_db.hybrid_search("Orchid", source="artiport")[0]["text"]
        source.update(instance_id="replacement-instance", cursor=0, ids=[], items=[])
        assert artiport_source.ingest()["reconciled_deletions"] == 1
        assert not pg_test_db.hybrid_search("Orchid", source="artiport")
        with clync.connect() as con:
            assert not con.execute("SELECT * FROM units WHERE source='artiport'").fetchall()
        source.update(instance_id="race-instance", cursor=1, ids=["one"],
                      items=[source_item := copy.deepcopy(archived)])
        source_item["sequence"] = 1
        source_item["artifact"]["state"] = "active"
        artiport_source.ingest()
        with clync.connect() as con:
            for table in ("chunks", "indexed_units"):
                con.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_unit_id_fkey")
        pg_test_db.ensure_index_schema()
        embed = pg_test_db._embed

        def delete_while_embedding(texts, max_length):
            vectors = embed(texts, max_length)
            source["ids"] = []
            artiport_source.ingest()
            return vectors

        with monkeypatch.context() as race:
            race.setattr(pg_test_db, "_embed", delete_while_embedding)
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                pg_test_db.build_index()
        assert not pg_test_db.hybrid_search("Heliotrope", source="artiport")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_authenticated_redirect_does_not_forward_bearer_token():
    from urllib.error import HTTPError

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/inventory':
                self.send_response(302)
                self.send_header('Location', f'http://localhost:{self.server.server_port}/collect')
                self.end_headers()
            else:
                received.append(self.headers.get('Authorization'))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError, match='redirect'):
            artiport_source._get({'url': f'http://127.0.0.1:{server.server_port}',
                                  'token': 'test-token'}, '/api/inventory')
        assert not received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
