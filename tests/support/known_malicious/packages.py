"""Python package checks: lockfiles, requirements, pyproject.toml and setup.py."""

from __future__ import annotations

import ast
import re

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.8-3.10
    import tomli as tomllib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .denylist import Denylist

_SETUP_KWARGS = {
    "install_requires",
    "setup_requires",
    "tests_require",
    "extras_require",
}
# Shell separators outside quotes, so a quoted `"pkg==1; marker"` stays whole.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ARCHIVE = (".whl", ".tar.gz", ".zip", ".tgz")


def _bad_version(req: Requirement, deny: Denylist) -> Version | None:
    if req.url:
        return None
    bad = deny.pypi.get(canonicalize_name(req.name), frozenset())
    return next(
        (v for v in sorted(bad) if req.specifier.contains(v, prereleases=True)), None
    )


def check_requirement(spec: str, deny: Denylist) -> str | None:
    match = _NAME.match(spec)
    if not match or canonicalize_name(match.group()) not in deny.pypi:
        return None
    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return f"cannot verify '{spec}' excludes known-malicious versions"
    bad = _bad_version(req, deny)
    return (
        f"'{spec}' allows known-malicious {canonicalize_name(req.name)}=={bad}"
        if bad
        else None
    )


def _locked_violation(pkg: dict, deny: Denylist) -> str | None:
    bad = deny.pypi.get(canonicalize_name(str(pkg.get("name", ""))))
    if not bad or "version" not in pkg:
        return None
    try:
        version = Version(str(pkg["version"]))
    except InvalidVersion:
        return f"cannot verify locked {pkg['name']} version {pkg['version']!r}"
    if version in bad:
        return f"lock resolves known-malicious {pkg['name']}=={pkg['version']}"
    return None


def scan_lock(text: str, deny: Denylist) -> list[str]:
    pkgs = tomllib.loads(text).get("package", [])
    return [h for h in (_locked_violation(p, deny) for p in pkgs) if h]


def logical_lines(text: str) -> list[str]:
    """Join backslash continuations after dropping full-line comments, as Docker does."""
    kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return re.sub(r"\\\n", " ", "\n".join(kept)).splitlines()


def scan_requirements(text: str, deny: Denylist) -> list[str]:
    hits = []
    for line in logical_lines(text):
        spec = re.split(r"\s+#|^#|\s+--", line, maxsplit=1)[0].strip()
        if spec and not spec.startswith("-"):
            hits.append(check_requirement(spec, deny))
    return [h for h in hits if h]


def _pyproject_requirements(data: dict) -> tuple[list[str], list[str]]:
    project, uv = data.get("project", {}), data.get("tool", {}).get("uv", {})
    direct = list(project.get("dependencies", []))
    direct += [
        r for group in project.get("optional-dependencies", {}).values() for r in group
    ]
    direct += [
        r
        for g in data.get("dependency-groups", {}).values()
        for r in g
        if isinstance(r, str)
    ]
    direct += list(uv.get("dev-dependencies", []))
    indirect = list(uv.get("constraint-dependencies", [])) + list(
        uv.get("override-dependencies", [])
    )
    return direct, indirect + list(data.get("build-system", {}).get("requires", []))


def _exact_pin_violation(spec: str, deny: Denylist) -> str | None:
    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return None  # check_requirement already reports it as unverifiable
    if canonicalize_name(req.name) not in deny.pypi or req.url:
        return None
    if _is_exact_pin(list(req.specifier)):
        return None
    return f"direct requirement '{spec}' must be an exact == pin"


def _is_exact_pin(pins: list) -> bool:
    if len(pins) != 1:
        return False
    return pins[0].operator in ("==", "===") and "*" not in pins[0].version


def scan_pyproject(text: str, deny: Denylist, exact_pins: bool) -> list[str]:
    direct, indirect = _pyproject_requirements(tomllib.loads(text))
    hits = [check_requirement(spec, deny) for spec in direct + indirect]
    if exact_pins:
        hits += [_exact_pin_violation(spec, deny) for spec in direct]
    return [h for h in hits if h] + scan_artifacts(text, deny)


def _setup_kwarg_values(tree: ast.AST) -> list[ast.expr]:
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    return [kw.value for c in calls for kw in c.keywords if kw.arg in _SETUP_KWARGS]


def _string_constants(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Dict):  # extras_require: keys are extra names, not specs
        return [s for v in node.values for s in _string_constants(v)]
    consts = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)]
    return [v for v in consts if isinstance(v, str)]


def _setup_specs(tree: ast.AST) -> list[str]:
    """String literals passed to setup()'s requirement keywords."""
    return [s for v in _setup_kwarg_values(tree) for s in _string_constants(v)]


def scan_setup_py(text: str, deny: Denylist) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [
            f"cannot parse ({exc}), so cannot verify it excludes denylisted versions"
        ]
    hits = [check_requirement(spec, deny) for spec in _setup_specs(tree)]
    return [h for h in hits if h] + scan_artifacts(text, deny)


def scan_artifacts(text: str, deny: Denylist) -> list[str]:
    hits = []
    for name, versions in deny.versions_raw.items():
        name_re = re.escape(name).replace("-", "[-_.]")
        for version in versions:
            pattern = rf"(?<![\w.-]){name_re}-{re.escape(version)}(?=-|\.tar|\.zip)"
            vcs = rf"/{name_re}(?:\.git)?@v?{re.escape(version)}(?![\w.])"
            if re.search(pattern, text, re.I) or re.search(vcs, text, re.I):
                hits.append(f"references a known-malicious {name} {version} artifact")
    return hits
