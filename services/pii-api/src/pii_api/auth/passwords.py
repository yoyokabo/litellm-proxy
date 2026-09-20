"""Password hashing on ``hashlib.scrypt``.

No passlib, no bcrypt wheel. Three reasons, in order of how much they matter
here:

1. **Air-gapped delivery.** Every dependency is one more wheel to mirror and
   one more thing that can be missing at install time. scrypt has been in the
   standard library since 3.6 and is backed by OpenSSL, which is already in
   the image because Python links against it.
2. **scrypt is memory-hard**, so it resists the GPU attack that makes plain
   PBKDF2 a poor choice for passwords.
3. The encoded form carries its own parameters, so raising the cost later does
   not invalidate existing hashes -- ``verify`` reads N, r and p from the
   stored string rather than assuming today's constants.

The format is ``scrypt$n$r$p$salt_hex$hash_hex``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

__all__ = ["hash_password", "needs_rehash", "verify_password"]

# OWASP's scrypt guidance: N=2^17, r=8, p=1. ~128 MB per hash at r=8, which is
# the point -- it is also ~128 MB per guess for an attacker with the table.
_N: Final = 2**17
_R: Final = 8
_P: Final = 1
_DKLEN: Final = 64
_SALT_BYTES: Final = 16

# hashlib.scrypt enforces maxmem; the default is too small for these
# parameters, so it is set explicitly rather than left to fail at runtime.
_MAXMEM: Final = 256 * 1024 * 1024


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=_MAXMEM
    )
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against a stored hash.

    Returns False rather than raising on a malformed stored value. A corrupt
    row must fail the login, not 500 the endpoint -- the difference is visible
    to an unauthenticated caller and would distinguish "this account exists
    and its hash is broken" from "no such account".
    """
    try:
        scheme, n_raw, r_raw, p_raw, salt_hex, digest_hex = encoded.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(n_raw), int(r_raw), int(p_raw)
        salt, expected = bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False

    try:
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=_MAXMEM,
        )
    except ValueError:
        return False

    return hmac.compare_digest(candidate, expected)


def needs_rehash(encoded: str) -> bool:
    """True when a stored hash uses weaker parameters than today's constants."""
    try:
        scheme, n_raw, r_raw, p_raw, _, _ = encoded.split("$")
    except ValueError:
        return True
    return scheme != "scrypt" or (int(n_raw), int(r_raw), int(p_raw)) != (_N, _R, _P)
