#!/usr/bin/env python
"""Validate that example scripts have valid imports and syntax.

This script checks:
1. Syntax is valid (AST parses)
2. All imports from spark_dist_fit resolve to actual exports
"""

import ast
import sys
from pathlib import Path


def get_spark_dist_fit_imports(filepath: Path) -> list[tuple[str, int]]:
    """Extract all imports from spark_dist_fit in a file.

    Returns list of (import_name, line_number) tuples.
    """
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"FAIL: {filepath} - Syntax error: {e}")
        sys.exit(1)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("spark_dist_fit"):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("spark_dist_fit"):
                    imports.append((alias.name, node.lineno))

    return imports


def validate_imports(imports: list[tuple[str, int]], filepath: Path) -> bool:
    """Validate that all imports exist in spark_dist_fit."""
    import spark_dist_fit

    # Get all public exports
    public_exports = set(dir(spark_dist_fit))

    valid = True
    for name, lineno in imports:
        if name not in public_exports:
            print(f"FAIL: {filepath}:{lineno} - '{name}' not found in spark_dist_fit")
            print(f"      Available exports: {sorted(e for e in public_exports if not e.startswith('_'))}")
            valid = False

    return valid


def main():
    examples_dir = Path(__file__).parent.parent / "examples"
    py_files = list(examples_dir.glob("*.py"))

    if not py_files:
        print(f"No Python files found in {examples_dir}")
        sys.exit(1)

    print(f"Validating {len(py_files)} example files...")
    all_valid = True

    for filepath in py_files:
        print(f"\nChecking {filepath.name}...")

        # Get imports
        imports = get_spark_dist_fit_imports(filepath)

        if not imports:
            print("  No spark_dist_fit imports found")
            continue

        print(f"  Found imports: {[name for name, _ in imports]}")

        # Validate imports exist
        if not validate_imports(imports, filepath):
            all_valid = False
        else:
            print("  OK - all imports valid")

    print("\n" + "=" * 60)
    if all_valid:
        print("SUCCESS: All example imports are valid")
        sys.exit(0)
    else:
        print("FAILURE: Some imports are invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
