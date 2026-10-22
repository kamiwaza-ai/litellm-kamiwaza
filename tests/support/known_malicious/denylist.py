"""Load security/known-malicious-packages.json."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version

DENYLIST_PATH = "security/known-malicious-packages.json"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class Denylist:
    pypi: dict[str, frozenset[Version]]
    versions_raw: dict[str, tuple[str, ...]]
    digests: frozenset[str]
    image_paths: frozenset[str]


def load_denylist(path: Path) -> Denylist:
    data = json.loads(path.read_text())
    raw = {canonicalize_name(k): tuple(v) for k, v in data["pypi"].items()}
    return Denylist(
        pypi={k: frozenset(Version(v) for v in vs) for k, vs in raw.items()},
        versions_raw=raw,
        digests=_digests(data["images"], path),
        image_paths=frozenset(_image_path(ref) for ref in data["images"]),
    )


def _digests(images: dict[str, list[str]], path: Path) -> frozenset[str]:
    digests = frozenset(d for ds in images.values() for d in ds)
    malformed = sorted(d for d in digests if not DIGEST.fullmatch(d))
    if malformed:
        raise ValueError(f"malformed digests in {path}: {malformed}")
    return digests


def _image_path(ref: str) -> str:
    first, _, rest = ref.partition("/")
    is_registry = rest and ("." in first or ":" in first or first == "localhost")
    return rest if is_registry else ref
