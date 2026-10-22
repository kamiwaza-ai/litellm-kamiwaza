"""Python package checks: lockfiles, requirements, pyproject, setup.py, installs."""

from __future__ import annotations

import ast
import re
import shlex

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.8-3.10
    import tomli as tomllib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from .denylist import Denylist

_INSTALL = re.compile(
    r"\b(?:pip3?|pipx|poetry|uv(?:\s+tool)?)\s+(?:install|add)\b|\buvx\b"
)
_SETUP_KWARGS = {
    "install_requires",
    "setup_requires",
    "tests_require",
    "extras_require",
}
_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\|")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ARCHIVE = (".whl", ".tar.gz", ".zip", ".tgz")


def _bad_version(req: Requirement, deny: Denylist) -> Version | None:
    if req.url:
        return None
    bad = deny.pypi.get(canonicalize_name(req.name), frozenset())
    return next(
        (v for v in sorted(bad) if req.specifier.contains(v, prereleases=True)), None
    )


def _check_requirement(spec: str, deny: Denylist) -> str | None:
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


def scan_lock(text: str, deny: Denylist) -> list[str]:
    hits = []
    for pkg in tomllib.loads(text).get("package", []):
        bad = deny.pypi.get(canonicalize_name(pkg.get("name", "")), frozenset())
        if pkg.get("version") and Version(pkg["version"]) in bad:
            hits.append(
                f"lock resolves known-malicious {pkg['name']}=={pkg['version']}"
            )
    return hits


def _logical_lines(text: str) -> list[str]:
    return text.replace("\\\r\n", " ").replace("\\\n", " ").splitlines()


def scan_requirements(text: str, deny: Denylist) -> list[str]:
    hits = []
    for line in _logical_lines(text):
        spec = re.split(r"\s+#|^#|\s+--", line, maxsplit=1)[0].strip()
        if spec and not spec.startswith("-"):
            hits.append(_check_requirement(spec, deny))
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
    req = Requirement(spec)
    if canonicalize_name(req.name) not in deny.pypi or req.url:
        return None
    pins = list(req.specifier)
    if (
        len(pins) == 1
        and pins[0].operator in ("==", "===")
        and "*" not in pins[0].version
    ):
        return None
    return f"direct requirement '{spec}' must be an exact == pin"


def scan_pyproject(text: str, deny: Denylist, exact_pins: bool) -> list[str]:
    direct, indirect = _pyproject_requirements(tomllib.loads(text))
    hits = [_check_requirement(spec, deny) for spec in direct + indirect]
    if exact_pins:
        hits += [_exact_pin_violation(spec, deny) for spec in direct]
    return [h for h in hits if h]


def _setup_kwarg_values(tree: ast.AST) -> list[ast.expr]:
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    return [kw.value for c in calls for kw in c.keywords if kw.arg in _SETUP_KWARGS]


def _string_constants(node: ast.AST) -> list[str]:
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
    hits = [_check_requirement(spec, deny) for spec in _setup_specs(tree)]
    return [h for h in hits if h]


def _install_args(segment: str) -> list[str]:
    match = _INSTALL.search(segment)
    if not match:
        return []
    rest = segment[match.end() :]
    try:
        args = shlex.split(rest)
    except ValueError:
        args = rest.split()
    if match.group() == "uvx" and "--from" in args[:-1]:
        return [args[args.index("--from") + 1]]  # the rest is the command name
    return args


def _is_spec_token(token: str) -> bool:
    return not (
        token.startswith("-")
        or "/" in token
        or "*" in token
        or token.endswith(_ARCHIVE)
    )


def _install_specs(text: str) -> list[str]:
    """Package specs named by install commands, skipping full-line comments."""
    lines = [ln for ln in _logical_lines(text) if not ln.lstrip().startswith("#")]
    segments = [seg for ln in lines for seg in _SEGMENT_SPLIT.split(ln)]
    tokens = [tok for seg in segments for tok in _install_args(seg)]
    return [re.sub(r"@(?=\d)", "==", tok) for tok in tokens if _is_spec_token(tok)]


def scan_commands(text: str, deny: Denylist) -> list[str]:
    hits = [_check_requirement(spec, deny) for spec in _install_specs(text)]
    return [h for h in hits if h] + scan_artifacts(text, deny)


def scan_artifacts(text: str, deny: Denylist) -> list[str]:
    hits = []
    for name, versions in deny.versions_raw.items():
        name_re = re.escape(name).replace("-", "[-_.]")
        for version in versions:
            pattern = rf"(?<![\w.-]){name_re}-{re.escape(version)}(?=-|\.tar|\.zip)"
            if re.search(pattern, text, re.I):
                hits.append(f"references a known-malicious {name} {version} artifact")
    return hits
