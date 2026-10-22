"""Container image checks: denylisted digests and mutable tags on watched images."""

from __future__ import annotations

import json
import re

import yaml

from .commands import scan_commands
from .denylist import DIGEST, Denylist

# Hosts whose berriai/litellm paths are web pages, not pullable images.
_NOT_REGISTRIES = {
    "github.com",
    "raw.githubusercontent.com",
    "pypi.org",
    "hub.docker.com",
    "docs.litellm.ai",
}


def _image_pattern(deny: Denylist) -> re.Pattern[str]:
    paths = "|".join(
        re.escape(p) for p in sorted(deny.image_paths, key=len, reverse=True)
    )
    return re.compile(
        # Start at a token boundary. `:-` (a shell/compose default) is allowed, and so is
        # `}/`, a variable registry prefix such as `${REGISTRY:-ghcr.io}/berriai/...`.
        r"(?:(?<=\}/)|(?<![\w./])(?<![\w.]-))"
        r"(?:(?P<registry>[a-z0-9][a-z0-9.-]*(?::\d+)?(?:/[a-z0-9._-]+)*)/)?"
        rf"(?P<path>{paths})(?![\w./-])"
        r"(?P<tag>:[\w.${}-]+)?(?P<digest>@sha256:[0-9a-f]{64})?"
    )


def _uncommented(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def _is_web_page(match: re.Match[str]) -> bool:
    return (match.group("registry") or "").split("/")[0] in _NOT_REGISTRIES


def denylisted_digests(text: str, deny: Denylist) -> list[str]:
    return [
        f"references known-malicious image digest {d}"
        for d in deny.digests
        if d[7:] in text
    ]


def scan_images(text: str, deny: Denylist) -> list[str]:
    text = _uncommented(text)
    hits = denylisted_digests(text, deny)
    for m in _image_pattern(deny).finditer(text):
        if _is_web_page(m):
            continue
        if not m.group("digest"):
            hits.append(
                f"image '{m.group()}' must be pinned by @sha256 digest, not a mutable tag"
            )
    return hits


def scan_json(text: str, deny: Denylist) -> list[str]:
    try:
        doc = json.loads(text)
    except ValueError:
        return scan_commands(text, deny) + scan_images(text, deny)
    hits: list[str] = []
    _walk_yaml(doc, deny, hits)
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
        _walk_yaml_mapping(node, deny, hits)


def _walk_yaml_mapping(node: dict, deny: Denylist, hits: list[str]) -> None:
    if isinstance(node.get("repository"), str):
        hits += denylisted_digests(node["repository"], deny)
        hits += _check_split_image(node, deny)
    for value in (v for k, v in node.items() if k != "repository"):
        _walk_yaml(value, deny, hits)


def _check_split_image(node: dict, deny: Denylist) -> list[str]:
    """Helm-style ``{registry, repository, tag, digest}`` image blocks."""
    ref = "/".join(str(p) for p in (node.get("registry"), node["repository"]) if p)
    if not _image_pattern(deny).search(ref):
        return []
    parts = (str(node.get(k) or "") for k in ("digest", "repository", "tag"))
    found = DIGEST.search(" ".join(parts))  # a digest key, or an inline @sha256
    digest = found.group() if found else ""
    if not digest:
        return [f"image '{ref}' must set an @sha256 digest, not only a mutable tag"]
    if digest in deny.digests:
        return [f"image '{ref}' uses known-malicious digest {digest}"]
    return []
