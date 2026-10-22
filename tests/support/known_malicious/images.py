"""Container image checks: denylisted digests and mutable tags on watched images."""

from __future__ import annotations

import re

import yaml

from .denylist import DIGEST, Denylist
from .packages import scan_commands

_NOT_REGISTRIES = {"github.com", "raw.githubusercontent.com", "pypi.org"}


def _image_pattern(deny: Denylist) -> re.Pattern[str]:
    paths = "|".join(
        re.escape(p) for p in sorted(deny.image_paths, key=len, reverse=True)
    )
    return re.compile(
        rf"(?<![\w.-])(?:(?P<registry>[a-z0-9.-]+(?::\d+)?)/)?(?P<path>{paths})(?![\w./-])"
        r"(?P<tag>:[\w.${}-]+)?(?P<digest>@sha256:[0-9a-f]{64})?"
    )


def scan_images(text: str, deny: Denylist) -> list[str]:
    hits = [
        f"references known-malicious image digest {d}"
        for d in deny.digests
        if d[7:] in text
    ]
    for m in _image_pattern(deny).finditer(text):
        if m.group("registry") in _NOT_REGISTRIES:
            continue
        if not m.group("digest"):
            hits.append(
                f"image '{m.group()}' must be pinned by @sha256 digest, not a mutable tag"
            )
    return hits


def scan_yaml(text: str, deny: Denylist) -> list[str]:
    try:
        docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return scan_commands(text, deny) + scan_images(text, deny)
    hits: list[str] = []
    for doc in docs:
        _walk_yaml(doc, deny, hits)
    return hits


def _walk_yaml(node: object, deny: Denylist, hits: list[str]) -> None:
    if isinstance(node, str):
        hits += scan_commands(node, deny) + scan_images(node, deny)
    elif isinstance(node, list):
        for item in node:
            _walk_yaml(item, deny, hits)
    elif isinstance(node, dict):
        if isinstance(node.get("repository"), str):
            hits += _check_split_image(node, deny)
        for key, value in node.items():
            if key != "repository":
                _walk_yaml(value, deny, hits)


def _check_split_image(node: dict, deny: Denylist) -> list[str]:
    """Helm-style ``{registry, repository, tag, digest}`` image blocks."""
    ref = "/".join(str(p) for p in (node.get("registry"), node["repository"]) if p)
    if not _image_pattern(deny).search(ref):
        return []
    digest = str(node.get("digest") or "")
    if not DIGEST.fullmatch(digest):
        return [f"image '{ref}' must set an @sha256 digest, not only a mutable tag"]
    if digest in deny.digests:
        return [f"image '{ref}' uses known-malicious digest {digest}"]
    return []
