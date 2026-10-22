"""Install-command checks: pip, uv, uvx, pipx, poetry, pdm and pipenv invocations."""

from __future__ import annotations

import re
import shlex

from .denylist import Denylist
from .packages import ARCHIVE, check_requirement, logical_lines, scan_artifacts

_INSTALL = re.compile(
    r"\b(?:pip(?:3(?:\.\d+)?)?|pipx|poetry|pdm|pipenv|uv(?:\s+tool)?)\s+(?:install|add)\b"
    r"|\buv\s+(?:tool\s+)?run\b|\buvx\b"
)
_RUNNERS = re.compile(r"uvx|.*\brun")  # their positional args after --from are commands
_SEGMENT_SPLIT = re.compile(r"""(?:&&|\|\||;|\|)(?=(?:[^'"]|'[^']*'|"[^"]*")*$)""")
_OPTION_VALUE = re.compile(r"--(?:from|with)=")


def _split_option_value(arg: str) -> list[str]:
    """`--from=spec` -> ["--from", "spec"], so it is handled like `--from spec`."""
    if _OPTION_VALUE.match(arg):
        return arg.split("=", 1)
    return [arg]


def _install_args(segment: str) -> list[str]:
    match = _INSTALL.search(segment)
    if not match:
        return []
    rest = segment[match.end() :]
    try:
        args = shlex.split(rest, comments=True)
    except ValueError:
        args = rest.split()
    args = [part for a in args for part in _split_option_value(a)]
    if _RUNNERS.fullmatch(match.group()) and "--from" in args[:-1]:
        return [args[args.index("--from") + 1]]  # the rest is the command name
    return args


def _is_spec_token(token: str) -> bool:
    return not (
        token.startswith("-") or "/" in token or "*" in token or token.endswith(ARCHIVE)
    )


def _install_specs(text: str) -> list[str]:
    """Package specs named by install commands."""
    segments = [seg for ln in logical_lines(text) for seg in _SEGMENT_SPLIT.split(ln)]
    tokens = [tok for seg in segments for tok in _install_args(seg)]
    return [re.sub(r"@(?=\d)", "==", tok) for tok in tokens if _is_spec_token(tok)]


def scan_commands(text: str, deny: Denylist) -> list[str]:
    hits = [check_requirement(spec, deny) for spec in _install_specs(text)]
    return [h for h in hits if h] + scan_artifacts(text, deny)
