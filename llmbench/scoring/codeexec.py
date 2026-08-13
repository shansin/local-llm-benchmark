"""Run model-generated code against the benchmark's own test suite.

This executes code written by a language model on the host machine. That is the
only way to get a true coding signal — a judge model's opinion of whether code
works is not the same as the code working — but it means the isolation below is
a hard requirement, not a nicety:

  * a fresh temporary directory as the only writable location
  * a scrubbed environment (no credentials, no proxies inherited)
  * `python -I` so the user's site-packages and PYTHON* vars are ignored
  * address-space, CPU-time, file-size, and process-count rlimits
  * a wall-clock timeout, enforced by killing the whole process group
  * network denied via an unprivileged namespace where the kernel allows it

Set `CODE_EXEC=0` to disable execution entirely; tasks then fall back to
judge-only scoring.
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)

MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
CPU_SECONDS = 30
FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_PROCESSES = 64
WALL_TIMEOUT = 60


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one suite against one candidate."""

    passed: int
    total: int
    detail: str

    @property
    def fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0


def extract_code(response: str) -> str | None:
    """Pull the Python source out of a model response.

    Tries fenced blocks labelled python first, then unlabelled fences, then the
    whole response — returning the first candidate that actually parses. A
    response with prose around the code is normal and must not be penalised;
    a response with no valid Python at all returns None.
    """
    labelled: list[str] = []
    unlabelled: list[str] = []
    for language, body in FENCE.findall(response):
        (labelled if language.lower() in ("python", "py", "python3") else unlabelled).append(body)

    # Several fenced blocks often means "the solution" then "the tests"; joining
    # the labelled ones keeps a solution that was split across blocks intact.
    candidates: list[str] = []
    if labelled:
        candidates.append("\n\n".join(labelled))
        candidates.extend(labelled)
    candidates.extend(sorted(unlabelled, key=len, reverse=True))
    candidates.append(response)

    for candidate in candidates:
        text = candidate.strip()
        if not text:
            continue
        try:
            ast.parse(text)
        except SyntaxError:
            continue
        return text
    return None


@lru_cache(maxsize=1)
def _network_sandbox() -> list[str]:
    """Return an `unshare` prefix if this kernel lets us drop networking."""
    unshare = shutil.which("unshare")
    if not unshare:
        return []
    try:
        probe = subprocess.run(
            [unshare, "-rn", "true"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [unshare, "-rn"] if probe.returncode == 0 else []


def _apply_limits() -> None:  # pragma: no cover - runs in the child process
    """Constrain the child before it executes anything."""
    os.setsid()  # own process group, so a timeout can kill any children too
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE_BYTES, FILE_SIZE_BYTES))
    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _kill_group(process: subprocess.Popen[str]) -> None:
    """Terminate the sandboxed process and everything it spawned."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - the group is already SIGKILLed
        process.kill()


def _sandbox_env(workdir: Path) -> dict[str, str]:
    """A minimal environment: nothing inherited that could leak or reach out."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "NO_COLOR": "1",
    }


@dataclass(frozen=True)
class _Report:
    passed: int
    total: int
    errors: int
    first_problem: str


def _parse_junit(report: Path) -> _Report:
    """Read pytest's JUnit XML into pass/total counts and the first problem."""
    try:
        root = ET.parse(report).getroot()
    except (OSError, ET.ParseError):
        return _Report(0, 0, 0, "no test report produced")

    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return _Report(0, 0, 0, "empty test report")

    total = int(suite.get("tests", 0))
    errors = int(suite.get("errors", 0))
    failures = int(suite.get("failures", 0))
    skipped = int(suite.get("skipped", 0))
    passed = max(total - failures - errors - skipped, 0)

    first = ""
    for case in suite.iter("testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            message = (problem.get("message") or "").strip().replace("\n", " ")
            first = f"{case.get('name')}: {message[:160]}"
            break

    return _Report(passed, total, errors, first)


def run_suite(code: str, suite_path: Path, timeout: int = WALL_TIMEOUT) -> ExecResult:
    """Run `suite_path` against `code` in a sandbox and count passing tests."""
    with tempfile.TemporaryDirectory(prefix="llmbench-exec-") as tmp:
        workdir = Path(tmp)
        (workdir / "candidate.py").write_text(code)
        shutil.copy(suite_path, workdir / "test_candidate.py")
        report = workdir / "report.xml"

        command = [
            *_network_sandbox(),
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "test_candidate.py",
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            f"--junit-xml={report}",
        ]

        try:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=_sandbox_env(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_apply_limits,
            )
        except OSError as exc:
            return ExecResult(0, 0, f"could not start sandbox: {exc}")

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The child is its own process group leader (setsid), so killing the
            # group takes any processes it spawned with it. Killing only the
            # direct child would orphan them.
            _kill_group(process)
            return ExecResult(0, 0, f"timed out after {timeout}s (likely non-terminating)")

        outcome = _parse_junit(report)

        # A collection error means the suite never ran — the candidate did not
        # import, or does not define what the suite imports. Reporting that as
        # "0/1 tests passed" would misstate how much was actually attempted.
        if outcome.total == 0 or (outcome.passed == 0 and outcome.errors == outcome.total):
            reason = outcome.first_problem or (stderr or stdout or "").strip()
            reason = reason.replace("\n", " ")[:200]
            return ExecResult(
                0, 0, f"suite could not run: {reason}" if reason else "suite could not run"
            )

        detail = f"{outcome.passed}/{outcome.total} tests passed"
        if outcome.first_problem:
            detail += f" — first failure: {outcome.first_problem}"
        return ExecResult(outcome.passed, outcome.total, detail)


def run_code_check(response: str, suite_path: Path, timeout: int = WALL_TIMEOUT) -> ExecResult:
    """Extract code from a response and run the suite against it."""
    code = extract_code(response)
    if code is None:
        return ExecResult(0, 0, "no valid Python found in the response")
    return run_suite(code, suite_path, timeout)
