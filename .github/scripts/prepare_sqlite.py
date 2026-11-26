"""Load a patched SQLite in disposable CI runners, preserving real DB-API tests.

Pinned official archives and SHA3-256 values: https://sqlite.org/download.html
WAL-reset advisory: https://sqlite.org/wal.html#walresetbug
This script changes CI's interpreter/library environment, never an application store.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

VERSION = "3.53.4"
PRODUCT = "3530400"
ARCHIVES = {
    "source": ("sqlite-amalgamation-3530400.zip", "628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e"),
    "windows": ("sqlite-dll-win-x64-3530400.zip", "deddee963c810d1eeac3ce5e15c7c41da21a1c54d7a39cf54fbf577d2f50de3a"),
}
PROBE = """import json, sqlite3
with sqlite3.connect(':memory:') as c:
    print(json.dumps(c.execute('select sqlite_version(), sqlite_source_id()').fetchone()))
"""


def patched(version: str) -> bool:
    if type(version) is not str or len(version) > 64 or re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version) is None:
        return False
    parts = tuple(int(part) for part in version.split("."))
    return (parts >= (3, 51, 3) or
            (parts[:2] == (3, 44) and parts >= (3, 44, 6)) or
            (parts[:2] == (3, 50) and parts >= (3, 50, 7)))


def probe(env=None):
    return json.loads(subprocess.check_output([sys.executable, "-c", PROBE], env=env, text=True))


def archive(kind: str) -> zipfile.ZipFile:
    name, digest = ARCHIVES[kind]
    with urllib.request.urlopen("https://sqlite.org/2026/" + name, timeout=60) as response:
        payload = response.read(20 * 1024 * 1024 + 1)
    if len(payload) > 20 * 1024 * 1024 or hashlib.sha3_256(payload).hexdigest() != digest:
        raise RuntimeError("Pinned SQLite archive integrity mismatch")
    return zipfile.ZipFile(io.BytesIO(payload))


def interpreter_identity(expected_python: str | None = None) -> dict:
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if expected_python is not None:
        if (type(expected_python) is not str or len(expected_python) > 16 or
                re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", expected_python) is None):
            raise ValueError("Expected Python must be a canonical MAJOR.MINOR version")
        if minor != expected_python:
            raise RuntimeError(f"Expected Python {expected_python}, running {minor}: {sys.executable}")
    extension = importlib.util.find_spec("_sqlite3")
    if extension is None or not extension.origin:
        raise RuntimeError("The selected interpreter has no SQLite extension")
    return {"python_executable": sys.executable,
            "python_real_executable": str(Path(sys.executable).resolve(strict=True)),
            "python_version": platform.python_version(), "python_minor": minor,
            "sqlite_extension": (extension.origin if extension.origin == "built-in"
                                 else str(Path(extension.origin).resolve(strict=True)))}


def mac_sqlite_dependency(listing: str) -> str:
    """Identify a real dynamic dependency; DYLD cannot replace embedded SQLite.

    python.org macOS installers statically link SQLite. The CI workflow selects
    Homebrew CPython instead; no extension or framework binary is rewritten here.
    """
    dependencies = set()
    for line in listing.splitlines():
        path, marker, _version = line.strip().partition(" (compatibility version ")
        if marker and re.fullmatch(r"libsqlite3(?:\.[0-9]+)*\.dylib", Path(path).name):
            dependencies.add(path)
    if len(dependencies) != 1:
        raise RuntimeError("Unpatched macOS SQLite is static or ambiguous; select a Homebrew CPython runtime")
    return dependencies.pop()


def main(expected_python: str | None = None):
    identity = interpreter_identity(expected_python)
    observed = probe()
    if patched(observed[0]):
        print(json.dumps({**identity, "sqlite_version": observed[0], "sqlite_source_id": observed[1], "ci_replacement": False}))
        return
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("RUNNER_TEMP"):
        raise RuntimeError("Upgrade the Python SQLite runtime; automatic replacement is limited to disposable GitHub CI")
    if identity["sqlite_extension"] == "built-in":
        raise RuntimeError("Unpatched built-in SQLite cannot be replaced; select a compatible CPython runtime")
    if sys.platform == "darwin":
        dependencies = subprocess.check_output(["otool", "-L", identity["sqlite_extension"]],
                                               text=True, timeout=30)
        identity["mac_sqlite_dependency"] = mac_sqlite_dependency(dependencies)
    build = Path(tempfile.mkdtemp(prefix="dml-sqlite-", dir=os.environ["RUNNER_TEMP"]))
    env = os.environ.copy()
    exported = None
    if sys.platform == "win32":
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            raise RuntimeError("Pinned Windows SQLite CI artifact requires x64")
        # Probe ran in a child: this process never imports/locks the old DLL.
        candidates = [Path(sys.base_prefix) / "DLLs" / "sqlite3.dll", Path(sys.base_prefix) / "sqlite3.dll"]
        targets = [path for path in candidates if path.is_file()]
        if len(targets) != 1:
            raise RuntimeError("Could not identify the disposable interpreter SQLite DLL")
        with archive("windows") as bundle:
            members = [name for name in bundle.namelist() if Path(name).name == "sqlite3.dll"]
            if len(members) != 1:
                raise RuntimeError("Unexpected pinned SQLite DLL archive")
            payload = bundle.read(members[0])
        staged = targets[0].with_suffix(".dml-new.dll")
        staged.write_bytes(payload)
        os.replace(staged, targets[0])
    elif sys.platform in {"linux", "darwin"}:
        with archive("source") as bundle:
            source = bundle.read(f"sqlite-amalgamation-{PRODUCT}/sqlite3.c")
        (build / "sqlite3.c").write_bytes(source)
        library = build / ("libsqlite3.dylib" if sys.platform == "darwin" else "libsqlite3.so.0")
        command = ["cc", "-O2", "-fPIC", "-DSQLITE_THREADSAFE=1", "-DSQLITE_ENABLE_COLUMN_METADATA=1",
                   "-DSQLITE_ENABLE_FTS5=1", "-DSQLITE_ENABLE_RTREE=1"]
        command += (["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-Wl,-soname,libsqlite3.so.0"])
        command += [str(build / "sqlite3.c"), "-o", str(library), "-lpthread", "-lm"]
        if sys.platform == "linux":
            command.append("-ldl")
        subprocess.run(command, check=True, timeout=180)
        if sys.platform == "darwin":
            shutil.copyfile(library, build / "libsqlite3.0.dylib")
        variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        env[variable] = str(build) + (os.pathsep + env[variable] if env.get(variable) else "")
        exported = (variable, env[variable])
    else:
        raise RuntimeError("Unsupported CI SQLite build platform")
    observed = probe(env)
    if observed[0] != VERSION:
        raise RuntimeError(f"Python {identity['python_minor']} at {identity['python_executable']} "
                           f"loaded SQLite {observed[0]} ({observed[1]}); expected pinned SQLite {VERSION}")
    if exported:
        if any("\n" in value or "\r" in value for value in exported):
            raise RuntimeError("Invalid CI environment value")
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as output:
            output.write("=".join(exported) + "\n")
    print(json.dumps({**identity, "sqlite_version": observed[0], "sqlite_source_id": observed[1], "ci_replacement": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-python", metavar="MAJOR.MINOR", help="Require this actual interpreter minor before any effects")
    main(parser.parse_args().expected_python)
