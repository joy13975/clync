"""Lite client-side cipher for cloud clync-share storage (ADR 0006).

Encrypts text and metadata with AES-256-GCM (AAD-bound to table/row/field),
preserves cosine similarity on dense vectors via a deterministic orthogonal
matrix (QR), preserves inner products on sparse vectors via a bijective token
permutation, and supports exact-match facet filtering via keyed HMAC-SHA256.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from typing import Iterable

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

DENSE_DIM = 1024
SPARSE_DIM = 250002
_FORMAT_VERSION = b"\x01"
_NONCE_BYTES = 12
_HMAC_HEX_LEN = 32


def _derive(master_key: bytes, info: bytes, length: int = 32) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=info,
    ).derive(master_key)


class ClyncCipher:
    """Per-owner cipher derived from a 32-byte master data key."""

    def __init__(
        self,
        master_key: bytes,
        *,
        dense_dim: int = DENSE_DIM,
        sparse_dim: int = SPARSE_DIM,
    ) -> None:
        if len(master_key) != 32:
            raise ValueError(f"master_key must be 32 bytes (got {len(master_key)})")
        self.dense_dim = dense_dim
        self.sparse_dim = sparse_dim
        self._aes = AESGCM(_derive(master_key, b"clync-share/aes-gcm/v1"))
        self._facet_key = _derive(master_key, b"clync-share/facet-hmac/v1")
        self._dense_seed = int.from_bytes(_derive(master_key, b"clync-share/dense-rot/v1"), "big")
        self._sparse_seed = int.from_bytes(_derive(master_key, b"clync-share/sparse-perm/v1"), "big")
        self._rotation: np.ndarray | None = None
        self._perm: np.ndarray | None = None

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Lazy 1024x1024 orthogonal matrix Q (Q^T Q = I) in float32."""
        if self._rotation is None:
            rng = np.random.default_rng(self._dense_seed)
            raw = rng.standard_normal((self.dense_dim, self.dense_dim)).astype(np.float32)
            q, r = np.linalg.qr(raw)
            diag = np.sign(np.diag(r))
            diag[diag == 0] = 1.0
            q *= diag
            self._rotation = np.ascontiguousarray(q, dtype=np.float32)
        return self._rotation

    @property
    def sparse_permutation(self) -> np.ndarray:
        """Lazy bijective permutation over 0..sparse_dim-1."""
        if self._perm is None:
            rng = np.random.default_rng(self._sparse_seed)
            self._perm = rng.permutation(self.sparse_dim).astype(np.int32)
        return self._perm

    def encrypt_text(self, plaintext: str | None, *, aad: str) -> str | None:
        if plaintext is None:
            return None
        nonce = os.urandom(_NONCE_BYTES)
        ct = self._aes.encrypt(nonce, plaintext.encode("utf-8"), aad.encode("utf-8"))
        return base64.b64encode(_FORMAT_VERSION + nonce + ct).decode("ascii")

    def decrypt_text(self, token: str | None, *, aad: str) -> str | None:
        if token is None:
            return None
        raw = base64.b64decode(token.encode("ascii"))
        if len(raw) < 1 + _NONCE_BYTES + 16 or raw[:1] != _FORMAT_VERSION:
            raise ValueError("invalid clync ciphertext envelope")
        nonce = raw[1 : 1 + _NONCE_BYTES]
        ct = raw[1 + _NONCE_BYTES :]
        return self._aes.decrypt(nonce, ct, aad.encode("utf-8")).decode("utf-8")

    def hmac_facet(self, field: str, value: str | None) -> str | None:
        if value is None:
            return None
        digest = hmac.new(
            self._facet_key,
            f"{field}:{value}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return digest[:_HMAC_HEX_LEN]

    def rotate_dense(self, vec: Iterable[float] | np.ndarray) -> list[float]:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape != (self.dense_dim,):
            raise ValueError(f"expected dense dim {self.dense_dim}, got {arr.shape}")
        rotated = self.rotation_matrix @ arr
        return rotated.tolist()

    def rotate_dense_batch(self, mat: np.ndarray) -> np.ndarray:
        """Rotate an (N, dense_dim) float32 matrix in one BLAS GEMM call."""
        arr = np.asarray(mat, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dense_dim:
            raise ValueError(f"expected shape (N, {self.dense_dim}), got {arr.shape}")
        return arr @ self.rotation_matrix.T

    def permute_sparse(self, weights: dict[int, float]) -> dict[int, float]:
        perm = self.sparse_permutation
        out: dict[int, float] = {}
        for token_id, weight in weights.items():
            idx = int(token_id)
            if not 0 <= idx < self.sparse_dim:
                raise ValueError(f"sparse token id {idx} out of range 0..{self.sparse_dim - 1}")
            if weight > 0:
                out[int(perm[idx])] = float(weight)
        return out

    def permute_sparse_literal(self, pg_sparse: str) -> str:
        """Permute a Postgres sparsevec literal `{1:0.5,42:1.2}/250002` in place."""
        body, dim_str = pg_sparse.rsplit("/", 1)
        dim = int(dim_str)
        if dim != self.sparse_dim:
            raise ValueError(f"expected sparse dim {self.sparse_dim}, got {dim}")
        inner = body.strip()[1:-1].strip()
        if not inner:
            raise ValueError("empty sparsevec literal")
        weights: dict[int, float] = {}
        for item in inner.split(","):
            k_str, v_str = item.split(":", 1)
            weights[int(k_str) - 1] = float(v_str)
        return sparse_literal(self.permute_sparse(weights), sparse_dim=self.sparse_dim)


def dense_literal(vec: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(x):.6g}" for x in vec) + "]"


def parse_dense_literal(pg_vec: str) -> list[float]:
    inner = pg_vec.strip()[1:-1].strip()
    return [float(x) for x in inner.split(",")] if inner else []


def sparse_literal(weights: dict[int, float], *, sparse_dim: int = SPARSE_DIM, max_nnz: int = 1000) -> str:
    items = sorted((int(t) + 1, float(w)) for t, w in weights.items() if w > 0)
    if not items:
        raise ValueError("empty sparse vector")
    if len(items) > max_nnz:
        items = sorted(sorted(items, key=lambda pair: (-pair[1], pair[0]))[:max_nnz])
    body = ",".join(f"{i}:{w:.7g}" for i, w in items)
    return f"{{{body}}}/{sparse_dim}"


def make_citation_handle(owner: str) -> str:
    slug = "".join(ch for ch in owner.lower() if ch.isalnum() or ch == "-") or "owner"
    return f"clync-{slug}-{secrets.token_hex(6)}"
