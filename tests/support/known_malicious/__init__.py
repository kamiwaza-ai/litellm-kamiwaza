"""Static guard against known-malicious package versions and container images.

The denylist is security/known-malicious-packages.json. The guard reads every
tracked uv/poetry/pdm lockfile, requirements/constraints file, pyproject.toml
(PEP 621, dependency-groups and uv tables; not [tool.poetry]), setup.py,
setup.cfg, Dockerfile/Containerfile, docker-bake file, shell script, Makefile,
YAML file (compose, Helm values, workflows, manifests) and JSON file
(kamiwaza.json, App Garden and engine catalogs), and reports:

- a lockfile entry that resolves a denylisted version;
- a requirement or install command whose specifier admits a denylisted version
  (an unpinned or open-ended spec admits it too);
- optionally, a direct pyproject requirement on a denylisted package that is
  not an exact ``==`` pin;
- a denylisted image digest, or a watched image (any registry or mirror)
  referenced by a mutable tag instead of an ``@sha256`` digest, in any of those
  files except lockfiles, requirements files, pyproject.toml and setup.py.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .commands import scan_commands
from .denylist import DENYLIST_PATH, Denylist, load_denylist
from .images import scan_images, scan_json, scan_yaml
from .packages import (
    scan_artifacts,
    scan_lock,
    scan_pyproject,
    scan_requirements,
    scan_setup_cfg,
    scan_setup_py,
)

__all__ = ["DENYLIST_PATH", "Denylist", "load_denylist", "scan", "tracked_files"]


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
    ).stdout.decode()
    return [p for p in out.split("\0") if p]


def scan(
    root: Path, files: list[str], deny: Denylist, *, exact_pins: bool
) -> list[str]:
    """Return one human-readable violation per finding, sorted and deduplicated."""
    found: set[str] = set()
    for rel in files:
        path = root / rel
        if rel == DENYLIST_PATH or not path.is_file():
            continue  # the denylist necessarily names what it denies
        text = path.read_text(errors="replace")
        found.update(f"{rel}: {msg}" for msg in _scan_file(rel, text, deny, exact_pins))
    return sorted(found)


def _scan_file(rel: str, text: str, deny: Denylist, exact_pins: bool) -> list[str]:
    for pattern, scanner in _SCANNERS:
        if pattern.search(rel):
            return _run(scanner, text, deny, exact_pins)
    return []


def _run(scanner, text: str, deny: Denylist, exact_pins: bool) -> list[str]:
    try:
        return scanner(text, deny, exact_pins)
    # A manifest that will not parse (bad TOML/INI/requirement/version, or a
    # table of the wrong type) fails closed instead of crashing the suite.
    except (ValueError, TypeError, AttributeError) as exc:
        return [
            f"cannot parse ({exc}), so cannot verify it excludes denylisted versions"
        ]


def _f(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(^|/){pattern}$", re.I)


# First match wins: tracked path -> scanner(text, denylist, exact_pins).
_SCANNERS = (
    (_f(r"(uv|poetry|pdm)\.lock"), lambda t, d, _: scan_lock(t, d)),
    (_f(r"pyproject\.toml"), scan_pyproject),
    (_f(r"setup\.py"), lambda t, d, _: scan_setup_py(t, d)),
    (_f(r"setup\.cfg"), lambda t, d, _: scan_setup_cfg(t, d)),
    (
        # requirements-dev.txt, *.constraints.txt, requirements/base.txt, ci/constraints/x.txt
        _f(
            r"([^/]*(requirements|constraints)[^/]*|(requirements|constraints)/.+)\.(txt|in)"
        ),
        lambda t, d, _: scan_requirements(t, d) + scan_artifacts(t, d),
    ),
    (
        _f(
            r"((docker|container)file[^/]*|[^/]*\.(docker|container)file|docker-bake[^/]*\.hcl)"
        ),
        lambda t, d, _: scan_commands(t, d) + scan_images(t, d),
    ),
    # npm/pnpm lockfiles carry no Python packages or images.
    (_f(r"([^/]*-lock\.(ya?ml|json)|npm-shrinkwrap\.json)"), lambda *_: []),
    (_f(r"[^/]*\.ya?ml"), lambda t, d, _: scan_yaml(t, d)),
    (_f(r"[^/]*\.json"), lambda t, d, _: scan_json(t, d)),
    (
        _f(r"([^/]*\.(sh|mk)|makefile)"),
        lambda t, d, _: scan_commands(t, d) + scan_images(t, d),
    ),
)
