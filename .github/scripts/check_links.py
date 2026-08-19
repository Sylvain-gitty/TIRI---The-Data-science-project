#!/usr/bin/env python3
"""Check that every relative markdown link in the repo points at a file that exists.

Lives under `.github/` rather than `scripts/` on purpose: `scripts/` is analysis code and
experiment drivers, this is repo hygiene. Keeping them apart means `scripts/` stays a
directory a reader can trust to be about the science.

Only relative links are checked. External URLs are not fetched — a CI job that depends on
the availability of a dozen third-party sites fails for reasons that have nothing to do with
the commit under test.

Usage:
    python .github/scripts/check_links.py            # whole repo
    python .github/scripts/check_links.py README.md  # specific files
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Directories that are not ours to lint.
SKIP_DIRS = {".venv", ".git", ".ipynb_checkpoints", "node_modules", "catboost_info"}

LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Links that are deliberately not repo paths.
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")


def markdown_files(args: list[str]) -> list[Path]:
    if args:
        return [Path(a).resolve() for a in args]
    return sorted(
        p for p in REPO.rglob("*.md")
        if not SKIP_DIRS.intersection(p.relative_to(REPO).parts)
    )


def check(path: Path) -> list[str]:
    """Return a list of human-readable problems found in one markdown file."""
    problems: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    for match in LINK_RE.finditer(text):
        target = match.group(2).strip()

        # Strip a link title: [x](path "Title")
        if " " in target and not target.startswith("<"):
            target = target.split(" ", 1)[0]
        target = target.strip("<>")

        if not target or target.startswith(EXTERNAL_PREFIXES):
            continue

        # Drop any anchor; we verify the file exists, not that the heading does.
        file_part = target.split("#", 1)[0]
        if not file_part:
            continue

        resolved = (path.parent / file_part).resolve()
        if not resolved.exists():
            line = text[: match.start()].count("\n") + 1
            rel = path.relative_to(REPO).as_posix()
            problems.append(f"{rel}:{line}: broken link -> {target}")

    return problems


def main() -> int:
    files = markdown_files(sys.argv[1:])
    problems = [p for f in files for p in check(f)]

    for p in problems:
        print(p)

    print(
        f"\nChecked {len(files)} markdown file(s): "
        f"{len(problems)} broken relative link(s)."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
