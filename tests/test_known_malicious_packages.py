"""No tracked file may resolve, install or pull a known-malicious artifact.

Denylist: security/known-malicious-packages.json (first entry: the LiteLLM
1.107.0 supply-chain compromise of 2026-10-21). To block a new incident, edit
only that file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.known_malicious import (
    DENYLIST_PATH,
    load_denylist,
    scan,
    tracked_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DENY = load_denylist(REPO_ROOT / DENYLIST_PATH)
BAD_AMD64 = "sha256:bd1aeee419ea36da9aa2b657288da4101887be468f020ac63d22ba6ed84a3ea5"
SAFE_DIGEST = "sha256:" + "a" * 64


def test_repo_references_no_known_malicious_package_or_image() -> None:
    # exact_pins=False: this repo has no lockfile; it caps litellm with a range
    # that excludes every denylisted version instead of an exact == pin.
    assert scan(REPO_ROOT, tracked_files(REPO_ROOT), DENY, exact_pins=False) == []


def test_denylist_covers_litellm_incident() -> None:
    assert "1.107.0" in {str(v) for v in DENY.pypi["litellm"]}
    assert len(DENY.digests) == 6
    assert {"berriai/litellm", "litellm/litellm"} <= DENY.image_paths


# One injected bad entry per file kind the guard reads.
BAD_FILES = {
    "uv.lock": '[[package]]\nname = "litellm"\nversion = "1.107.0"\n',
    "requirements.txt": "litellm==1.107.0\n",
    "deploy/constraints.txt": "LiteLLM[proxy] >= 1.50.0  # open-ended\n",
    "pyproject.toml": '[project]\ndependencies = ["litellm>=1.88.6,!=1.107.0"]\n',
    "tools/pyproject.toml": "[tool.pytest]\nfilterwarnings=ignore::DeprecationWarning\n",
    "setup.py": 'from setuptools import setup\nsetup(install_requires=["litellm>=1.0.0"])\n',
    "Dockerfile": "FROM python:3.12\nRUN pip install --no-cache-dir \\\n    litellm==1.107.0\n",
    "Containerfile": "FROM ghcr.io/berriai/litellm:main-latest\n",
    "Dockerfile.proxy": "FROM docker.litellm.ai/berriai/litellm-non_root:v1.106.2\n",
    "scripts/setup.sh": "uv pip install 'litellm[proxy]'\n",
    "docker-compose.yml": f"services:\n  llm:\n    image: litellm/litellm@{BAD_AMD64}\n",
    "charts/values.yaml": "litellm:\n  image:\n    repository: ghcr.io/berriai/litellm\n    tag: latest\n",
    ".github/workflows/ci.yml": "jobs:\n  t:\n    steps:\n      - run: uvx litellm@1.107.0 --help\n",
    "wheels/requirements.in": "https://example.invalid/litellm-1.107.0-py3-none-any.whl\n",
}

CLEAN_FILES = {
    "uv.lock": '[[package]]\nname = "litellm"\nversion = "1.88.6"\n',
    "requirements.txt": "litellm==1.88.6 \\\n    --hash=sha256:" + "b" * 64 + "\n",
    "constraints.txt": "litellm!=1.107.0,>=1.50.0\n",
    "pyproject.toml": '[project]\ndependencies = ["litellm[proxy]==1.88.6"]\n',
    "Dockerfile": (
        f"FROM ghcr.io/berriai/litellm:v1.106.2@{SAFE_DIGEST}\n"
        "# never `pip install litellm==X.Y.Z` unhashed\n"
        "RUN pip install litellm==1.88.6\n"
    ),
    "docker-compose.yml": "# docs: https://github.com/berriai/litellm\nservices: {}\n",
    "values.yaml": f"image:\n  repository: berriai/litellm\n  digest: {SAFE_DIGEST}\n",
    "setup.py": (
        "from setuptools import setup\n"
        'setup(keywords=["litellm"], install_requires=["litellm>=1.0,<1.107.0"])\n'
    ),
}


def _write(root: Path, files: dict[str, str]) -> list[str]:
    for rel, body in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    return list(files)


@pytest.mark.parametrize("rel", sorted(BAD_FILES))
def test_guard_catches_injected_bad_entry(tmp_path: Path, rel: str) -> None:
    files = _write(tmp_path, {rel: BAD_FILES[rel]})
    violations = scan(tmp_path, files, DENY, exact_pins=True)
    assert violations, f"guard missed the injected entry in {rel}"
    assert all(v.startswith(f"{rel}: ") for v in violations)


def test_guard_accepts_safe_pins_and_digests(tmp_path: Path) -> None:
    files = _write(tmp_path, CLEAN_FILES)
    assert scan(tmp_path, files, DENY, exact_pins=True) == []


def test_exact_pin_rule_is_what_rejects_a_safe_range(tmp_path: Path) -> None:
    files = _write(tmp_path, {"pyproject.toml": BAD_FILES["pyproject.toml"]})
    assert scan(tmp_path, files, DENY, exact_pins=False) == []
