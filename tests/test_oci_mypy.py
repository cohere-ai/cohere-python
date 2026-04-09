"""Mypy type-checking gate for OCI client code.

Runs mypy on OCI source and test files and fails if any type errors are found.
This prevents type regressions from being introduced silently.

Run with:
    pytest tests/test_oci_mypy.py
"""

import os
import shutil
import subprocess
import unittest

MYPY_BIN = shutil.which("mypy")

# Files that must stay mypy-clean
OCI_SOURCE_FILES = [
    "src/cohere/oci_client.py",
    "src/cohere/manually_maintained/lazy_oci_deps.py",
]

OCI_TEST_FILES = [
    "tests/test_oci_client.py",
]

# --follow-imports=silent prevents mypy from crawling into transitive
# dependencies (e.g. the AWS client) that have pre-existing errors.
_MYPY_BASE = [
    "--config-file", "mypy.ini",
    "--follow-imports=silent",
]


def _run_mypy(files: list[str], extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run mypy on the given files and return (exit_code, output)."""
    assert MYPY_BIN is not None
    env = {**os.environ, **(extra_env or {})}
    result = subprocess.run(
        [MYPY_BIN, *_MYPY_BASE, *files],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


@unittest.skipIf(MYPY_BIN is None, "mypy not found on PATH")
class TestOciMypy(unittest.TestCase):
    """Ensure OCI files pass mypy with no new errors."""

    def test_oci_source_types(self):
        """OCI source files must be free of mypy errors."""
        code, output = _run_mypy(OCI_SOURCE_FILES)
        self.assertEqual(code, 0, f"mypy found type errors in OCI source:\n{output}")

    def test_oci_test_types(self):
        """OCI test files must be free of mypy errors."""
        # PYTHONPATH=src so mypy can resolve `import cohere`
        code, output = _run_mypy(OCI_TEST_FILES, extra_env={"PYTHONPATH": "src"})
        self.assertEqual(code, 0, f"mypy found type errors in OCI tests:\n{output}")


if __name__ == "__main__":
    unittest.main()
