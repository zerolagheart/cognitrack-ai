"""Role-based API-key access control.

Roles are hierarchical: ``viewer`` (aggregate reports) < ``researcher`` (participant-level data,
uploads, inference) < ``admin`` (pipeline publishing). Keys come only from the environment. In
development with no keys configured, auth is disabled and every request acts as ``admin`` — a startup
warning makes this explicit, and production refuses to start without keys.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.app.core.config import Settings, get_settings

ROLE_RANK = {"viewer": 0, "researcher": 1, "admin": 2}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    role: str
    authenticated: bool


def _match(key: str, candidates: set[str]) -> bool:
    return any(hmac.compare_digest(key, c) for c in candidates)


def get_principal(key: str | None = Security(api_key_header), settings: Settings = Depends(get_settings)) -> Principal:
    if not settings.auth_enabled:
        return Principal(role="admin", authenticated=False)
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header", headers={"WWW-Authenticate": "API-Key"})
    for role in ("admin", "researcher", "viewer"):
        if _match(key, settings.keys(role)):
            return Principal(role=role, authenticated=True)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key", headers={"WWW-Authenticate": "API-Key"})


def require(role: str):
    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if ROLE_RANK[principal.role] < ROLE_RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires '{role}' role")
        return principal

    return dependency
