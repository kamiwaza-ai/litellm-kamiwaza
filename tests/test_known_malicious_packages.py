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


LITELLM_DIGESTS = {
    "sha256:bd1aeee419ea36da9aa2b657288da4101887be468f020ac63d22ba6ed84a3ea5",
    "sha256:56b74573eed6e4aafe4b50603dbe7853e0345f188c8e44a09f1098caf1f4bcdf",
    "sha256:24581ddc30767f16ecb69d2b0517556f2cd7dea56971adb42a3d1beb97d69523",
    "sha256:827dd893c74cf621a860a66a3a447a5c732533f087589ce905573e285c0a239c",
    "sha256:645a67d810628252d63df85dcc8824c58bcc77cf8383f4d3e6321f0ccc3ce9e4",
    "sha256:f75c8edae59b72b1ebfd617295833f465c85bb2fab9c1d60f21b496db86cb8da",
}


def test_denylist_covers_litellm_incident() -> None:
    assert {"1.82.7", "1.82.8", "1.107.0"} <= {str(v) for v in DENY.pypi["litellm"]}
    assert {"4.87.1", "4.87.2"} <= {str(v) for v in DENY.pypi["telnyx"]}
    assert LITELLM_DIGESTS <= DENY.digests
    assert {"berriai/litellm", "litellm/litellm"} <= DENY.image_paths


# One injected bad entry per file kind and evasion the guard must catch.
BAD_FILES = {
    "uv.lock": '[[package]]\nname = "litellm"\nversion = "1.107.0"\n',
    "badver/uv.lock": '[[package]]\nname = "litellm"\nversion = "not-a-version"\n',
    "requirements.txt": "litellm==1.107.0\n",
    "ci/constraints/linux-cpu.txt": "litellm==1.82.8\n",
    "requirements/base.in": "telnyx==4.87.1\n",
    "vcs/requirements.txt": "litellm @ git+https://github.com/BerriAI/litellm@v1.107.0\n",
    "lock2/uv.lock": '[[package]]\nname = "litellm"\nversion = "1.82.8"\n',
    "deploy/constraints.txt": "LiteLLM[proxy] >= 1.50.0  # open-ended\n",
    "pyproject.toml": '[project]\ndependencies = ["litellm>=1.88.6,!=1.107.0"]\n',
    "tools/pyproject.toml": "[tool.pytest]\nfilterwarnings=ignore::DeprecationWarning\n",
    "wheel/pyproject.toml": (
        '[project]\ndependencies = ["litellm==1.88.6"]\n[tool.uv.sources]\n'
        'litellm = { url = "https://example.invalid/litellm-1.107.0-py3-none-any.whl" }\n'
    ),
    "setup.py": 'from setuptools import setup\nsetup(install_requires=["litellm>=1.0.0"])\n',
    "url/setup.py": (
        "from setuptools import setup\nsetup(install_requires=["
        '"litellm @ https://example.invalid/litellm-1.107.0-py3-none-any.whl"])\n'
    ),
    "Dockerfile": "FROM python:3.12\nRUN pip install --no-cache-dir \\\n    litellm==1.107.0\n",
    "Dockerfile.comment": "RUN pip install --no-cache-dir \\\n    # LLM deps\n    litellm==1.107.0\n",
    "Containerfile": "FROM ghcr.io/berriai/litellm:main-latest\n",
    "Dockerfile.proxy": "FROM docker.litellm.ai/berriai/litellm-non_root:v1.106.2\n",
    "Dockerfile.mirror": "FROM registry.example.com/mirror/berriai/litellm:v1.106.2\n",
    "scripts/setup.sh": "uv pip install 'litellm[proxy]'\n",
    "scripts/marker.sh": "pip install \"litellm==1.107.0; python_version >= '3.8'\"\n",
    "scripts/pip312.sh": "pip3.12 install litellm\n",
    "scripts/from.sh": "uvx --from=litellm[proxy] litellm --port 4000\n",
    "scripts/pull.sh": "docker pull ghcr.io/berriai/litellm:main-latest\n",
    "scripts/bashc.sh": "bash -c \"pip install 'litellm==1.107.0'\"\n",
    "scripts/download.sh": "pip download litellm==1.107.0 -d wheels\n",
    "static/garden/apps.json": '{"apps": [{"image": "ghcr.io/berriai/litellm:latest"}]}\n',
    "mirror/values.yaml": f"image:\n  repository: internal.io/mirror@{BAD_AMD64}\n",
    "kamiwaza.json": '{"extra_docker_images": ["ghcr.io/berriai/litellm:latest"]}\n',
    "k8s/deploy.yaml": "spec:\n  containers:\n    - image: berriai/litellm:main-stable\n",
    "docker-compose.yml": f"services:\n  llm:\n    image: litellm/litellm@{BAD_AMD64}\n",
    "env.yml": "services:\n  llm:\n    image: ${LITELLM_IMAGE:-ghcr.io/berriai/litellm:latest}\n",
    "charts/values.yaml": "litellm:\n  image:\n    repository: ghcr.io/berriai/litellm\n    tag: latest\n",
    ".github/workflows/ci.yml": "jobs:\n  t:\n    steps:\n      - run: uvx litellm@1.107.0 --help\n",
    "wheels/requirements.in": "https://example.invalid/litellm-1.107.0-py3-none-any.whl\n",
}

CLEAN_FILES = {
    "uv.lock": '[[package]]\nname = "litellm"\nversion = "1.88.6"\n',
    "other/uv.lock": '[[package]]\nname = "odd"\nversion = "not-a-version"\n',
    "requirements.txt": "litellm==1.88.6 \\\n    --hash=sha256:" + "b" * 64 + "\n",
    "constraints.txt": "litellm!=1.82.7,!=1.82.8,!=1.107.0,>=1.50.0\n",
    "pyproject.toml": '[project]\ndependencies = ["litellm[proxy]==1.88.6"]\n',
    "Dockerfile": (
        f"FROM ghcr.io/berriai/litellm:v1.106.2@{SAFE_DIGEST}\n"
        "# never `pip install litellm==X.Y.Z` unhashed\n"
        "# FROM ghcr.io/berriai/litellm:main-latest\n"
        "RUN pip install litellm==1.88.6\n"
        "RUN pip install httpx  # mirrors litellm docs\n"
    ),
    "scripts/docs.sh": "echo 'see https://hub.docker.com/r/berriai/litellm'\n",
    "scripts/proxy.sh": "uvx --from=litellm[proxy]==1.88.6 litellm --port 4000\n",
    "docker-compose.yml": "services:\n  web:\n    labels:\n      docs: github.com/berriai/litellm\n",
    "values.yaml": f"image:\n  repository: berriai/litellm\n  digest: {SAFE_DIGEST}\n",
    "frontend/pnpm-lock.yaml": "packages:\n  /litellm@1.107.0: {}\n",
    "frontend/package-lock.json": '{"packages": {"x": {"resolved": "berriai/litellm:latest"}}}\n',
    "scripts/shc.sh": 'sh -c "pip install litellm==1.88.6"\n',
    "inline/values.yaml": f"image:\n  repository: ghcr.io/berriai/litellm@{SAFE_DIGEST}\n",
    "engine-catalog.json": f'{{"image": "ghcr.io/berriai/litellm@{SAFE_DIGEST}"}}\n',
    "setup.py": (
        "from setuptools import setup\n"
        'setup(keywords=["litellm"], install_requires=["litellm>=1.83.0,<1.107.0"],\n'
        '      extras_require={"litellm": ["litellm==1.88.6"]})\n'
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
