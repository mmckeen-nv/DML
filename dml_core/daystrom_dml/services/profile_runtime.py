"""Runtime admission and public-operation guards for the candidate profile."""
from __future__ import annotations

from contextlib import closing
from functools import wraps
import math
from pathlib import Path
import sqlite3

from ..contracts.profile import ProductionProfileError, validate_profile_authority


def outside_production_profile(method):
    """Keep a legacy capability unavailable on a selected profile instance."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if getattr(self, "_production_profile", None) is not None:
            raise ProductionProfileError("Operation is outside the selected production profile")
        return method(self, *args, **kwargs)
    return guarded


def preflight_profile_authority(profile_id: str | None, storage_dir: Path,
                                *, outbox_enabled: bool) -> None:
    """Check an existing SQLite view before provider initialization.

    Read-only SQLite access must honor committed WAL content. SQLite may use
    transient shared-memory/locking sidecars; this is not an immutable-file read.
    The journal repeats schema admission under its initialization lock.
    """
    if profile_id is None:
        return
    path = storage_dir.expanduser().resolve() / "dml_state.sqlite3"
    if path.with_name(path.name + ".migration.json").exists():
        raise ProductionProfileError("Incomplete journal migration requires recovery")
    if not path.exists():
        if path.with_name(path.name + ".identity.json").exists():
            raise ProductionProfileError("Previously initialized journal is missing")
        return
    if not path.is_file():
        raise ProductionProfileError("Profile authority must be a SQLite file")
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.Error as exc:
        raise ProductionProfileError("Profile authority cannot be inspected") from exc
    validate_profile_authority(profile_id, version, outbox_enabled)


def validate_profile_retrieval(prompt, *, scope, top_k, kinds, phase,
                               include_quarantined, as_of, dpm_identifiers) -> None:
    """Reject malformed scoped read inputs before embedding or ownership."""
    if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 1024 * 1024:
        raise ProductionProfileError("Profile query must be nonempty text of at most 1 MiB")
    for index, value in enumerate(scope):
        if value is None and index:
            continue
        if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256:
            raise ProductionProfileError("Profile scope requires exact nonempty string identifiers")
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 10):
        raise ProductionProfileError("Profile top_k must be an integer from 1 through 10")
    if kinds is not None and (type(kinds) is not list or len(kinds) > 256 or any(
            type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256
            for value in kinds)):
        raise ProductionProfileError("Profile kinds must be a bounded list of string identifiers")
    if phase is not None and (type(phase) is not str or phase not in {"plan", "build", "execute", "debug", "reflect"}):
        raise ProductionProfileError("Profile phase is invalid")
    if type(include_quarantined) is not bool:
        raise ProductionProfileError("Profile quarantine selection must be boolean")
    if as_of is not None:
        try:
            finite_time = type(as_of) in (int, float) and math.isfinite(as_of)
        except OverflowError:
            finite_time = False
        if not finite_time:
            raise ProductionProfileError("Profile as_of must be a finite number")
    if any(value is not None for value in dpm_identifiers):
        raise ProductionProfileError("DPM is outside the selected production profile")
