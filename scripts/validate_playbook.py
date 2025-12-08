"""Validate ACE playbook files for structural issues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ValidationError(Exception):
    """Raised when a playbook validation fails."""

    pass


def validate_playbook(playbook_path: Path) -> list[str]:
    """Validate a single playbook file and return list of errors."""
    errors: list[str] = []

    try:
        with playbook_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Failed to parse YAML: {e}")
        return errors

    if not isinstance(data, dict):
        errors.append("Playbook must be a YAML dictionary")
        return errors

    # Check for duplicate heuristic IDs
    context = data.get("context", {})
    heuristics = context.get("heuristics", [])

    heuristic_ids: dict[str, int] = {}
    for idx, heuristic in enumerate(heuristics):
        if not isinstance(heuristic, dict):
            continue
        heuristic_id = heuristic.get("id")
        if heuristic_id:
            if heuristic_id in heuristic_ids:
                errors.append(
                    f"Duplicate heuristic ID '{heuristic_id}' found at indices {heuristic_ids[heuristic_id]} and {idx}"
                )
            else:
                heuristic_ids[heuristic_id] = idx

    # Check for repetitive system instructions
    system_instructions = context.get("system_instructions", "")
    if isinstance(system_instructions, str):
        lines = [line.strip() for line in system_instructions.split("\n") if line.strip()]
        seen_lines: dict[str, int] = {}
        for idx, line in enumerate(lines):
            # Check for consecutive duplicates
            if idx > 0 and line == lines[idx - 1]:
                errors.append(f"Consecutive duplicate instruction line at line {idx + 1}: '{line[:50]}...'")
            # Check for multiple occurrences of the same line
            if line in seen_lines:
                seen_lines[line] += 1
            else:
                seen_lines[line] = 1

        # Warn about lines that appear more than twice
        for line, count in seen_lines.items():
            if count > 2:
                errors.append(f"Instruction line appears {count} times: '{line[:50]}...'")

    # Check for near-duplicate heuristics (optional warning)
    if len(heuristics) > 1:
        for i, h1 in enumerate(heuristics):
            if not isinstance(h1, dict):
                continue
            rule1 = h1.get("rule", "").lower().strip()
            if not rule1:
                continue
            for j, h2 in enumerate(heuristics[i + 1 :], start=i + 1):
                if not isinstance(h2, dict):
                    continue
                rule2 = h2.get("rule", "").lower().strip()
                if not rule2:
                    continue
                # Simple similarity check: if one rule contains most of the other
                if len(rule1) > 10 and len(rule2) > 10:
                    if rule1 in rule2 or rule2 in rule1:
                        errors.append(
                            f"Potential duplicate heuristics at indices {i} and {j}: "
                            f"'{rule1[:50]}...' and '{rule2[:50]}...'"
                        )

    return errors


def validate_all_playbooks(playbook_dir: Path) -> tuple[bool, list[tuple[Path, list[str]]]]:
    """Validate all playbook files in a directory."""
    playbooks = sorted(playbook_dir.glob("playbook_*.yaml"))
    if not playbooks:
        return True, []

    all_valid = True
    results: list[tuple[Path, list[str]]] = []

    for playbook_path in playbooks:
        errors = validate_playbook(playbook_path)
        if errors:
            all_valid = False
        results.append((playbook_path, errors))

    return all_valid, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ACE playbook files")
    parser.add_argument(
        "playbooks",
        nargs="*",
        type=Path,
        help="Paths to specific playbook files to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all playbooks in outputs/ace_playbooks/",
    )
    parser.add_argument(
        "--playbook-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ace_playbooks",
        help="Directory containing playbooks (default: outputs/ace_playbooks)",
    )

    args = parser.parse_args()

    if args.all:
        print(f"Validating all playbooks in {args.playbook_dir}...")
        all_valid, results = validate_all_playbooks(args.playbook_dir)
        for playbook_path, errors in results:
            if errors:
                print(f"\n❌ {playbook_path.name}:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print(f"✅ {playbook_path.name}: OK")
        if not all_valid:
            sys.exit(1)
        print("\n✅ All playbooks are valid!")
    elif args.playbooks:
        overall_ok = True
        for pb in args.playbooks:
            errors = validate_playbook(pb)
            if errors:
                overall_ok = False
                print(f"❌ Validation failed for {pb}:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print(f"✅ {pb} is valid!")
        if not overall_ok:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
