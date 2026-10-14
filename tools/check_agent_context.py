#!/usr/bin/env python3
"""Validate the coding-agent context structure of this repository.

Layer-1 checks from the AGENTS.md standardization (PR #5500): the context
files other agents rely on must exist and be well-formed. Checks:

1. ``AGENTS.md`` exists at the repository root.
2. ``CLAUDE.md`` exists and references ``AGENTS.md`` (the Claude Code link).
3. ``.agents/skills`` resolves to the shared skill directory.
4. Every ``SKILL.md`` under ``.claude/skills/`` (root and nested subtrees)
   has YAML frontmatter with non-empty ``name`` and single-line
   ``description`` (what the Agent Skills spec keys discovery on).

Run from anywhere; paths are resolved against the repository root found
via ``git rev-parse``. Exits non-zero on the first failing category,
listing every offending file.

Copyright IBM Corp. 2026
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if out.returncode != 0:
        sys.exit("Not inside a git repository.")
    return Path(out.stdout.strip())


def check_context_files(root: Path) -> list[str]:
    errors: list[str] = []
    agents = root / "AGENTS.md"
    if not agents.is_file():
        errors.append("AGENTS.md is missing from the repository root")
    claude = root / "CLAUDE.md"
    if not claude.is_file():
        errors.append(
            "CLAUDE.md is missing (expected one-line @AGENTS.md link for Claude Code)"
        )
    elif "@AGENTS.md" not in claude.read_text():
        errors.append("CLAUDE.md does not reference @AGENTS.md")
    return errors


def check_skills_symlink(root: Path) -> list[str]:
    errors: list[str] = []
    link = root / ".agents" / "skills"
    target = root / ".claude" / "skills"
    if not link.exists():
        errors.append(
            ".agents/skills does not resolve (.claude/skills unreachable via the "
            "convention path; broken symlink on a checkout without symlink support?)"
        )
    elif not target.is_dir():
        errors.append(".claude/skills is not a directory")
    return errors


def check_skill_frontmatter(root: Path) -> list[str]:
    errors: list[str] = []
    # ``**`` matches zero or more directories, so this covers the root
    # ``.claude/skills/`` and nested ones (``torch_spyre/_inductor/.claude/skills/``).
    skill_files = sorted(
        {p for p in root.glob("**/.claude/skills/**/SKILL.md") if ".git" not in p.parts}
    )
    for path in skill_files:
        rel = path.relative_to(root)
        text = path.read_text()
        match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
        if not match:
            errors.append(f"{rel}: no YAML frontmatter block")
            continue
        try:
            meta = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            errors.append(f"{rel}: YAML parse error: {exc}")
            continue
        if not isinstance(meta, dict):
            errors.append(f"{rel}: frontmatter is not a mapping")
            continue
        if not meta.get("name"):
            errors.append(f"{rel}: missing or empty 'name'")
        description = meta.get("description")
        if not description:
            errors.append(f"{rel}: missing or empty 'description'")
        elif re.search(r"^description:\s*[|>]", match.group(1), re.MULTILINE):
            # Pymarkdown mangles block scalars, and a folded (>-) scalar hides
            # the multi-line-ness from a parsed-value newline check; AGENTS.md
            # mandates a quoted single-line description.
            errors.append(f"{rel}: 'description' must be a quoted single-line string")
    return errors


def main() -> int:
    root = repo_root()
    failures = [
        *check_context_files(root),
        *check_skills_symlink(root),
        *check_skill_frontmatter(root),
    ]
    if failures:
        print("agent context validation FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("agent context validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
