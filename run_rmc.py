from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional


VALID_STATUSES = {
    "satisfied",
    "assertion_failed",
    "deadlock",
    "state_explosion",
    "timeout",
    "syntax_error",
    "tool_error",
}


@dataclass
class RMCResult:
    status: str
    project_name: str
    working_dir: str
    translator_log: str = ""
    compile_log: str = ""
    execution_log: str = ""
    output_xml: Optional[str] = None
    output_xml_text: str = ""
    statespace_xml: Optional[str] = None
    statespace_xml_text: str = ""
    progress_text: str = ""
    executable_path: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    notes: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)




def looks_like_java_runtime_mismatch(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in (
            "unsupportedclassversionerror",
            "unsupported major.minor version",
            "class file version",
            "only recognizes class file versions up to",
        )
    )


def probe_rmc_runtime(
    *,
    rmc_jar: str | Path,
    java_path: str | None = None,
    timeout_seconds: int = 15,
) -> tuple[bool, str]:
    rmc_jar = Path(rmc_jar)
    java_path = java_path or shutil.which("java")

    if not java_path:
        return False, "java was not found in PATH."
    if not rmc_jar.exists():
        return False, f"RMC jar not found: {rmc_jar}"

    try:
        probe = subprocess.run(
            [java_path, "-jar", str(rmc_jar), "-h"],
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"RMC runtime probe timed out after {timeout_seconds} seconds."
    except Exception as exc:
        return False, f"RMC runtime probe failed: {exc}"

    output = ((probe.stdout or "") + "\n" + (probe.stderr or "")).strip()
    lowered = output.lower()
    if "usage: rmc" in lowered:
        return True, "RMC jar launches successfully with the current Java runtime."
    if looks_like_java_runtime_mismatch(output):
        return (
            False,
            "Current Java runtime is too old for this RMC jar. Install Java 17+ or place a newer `java` earlier in PATH.",
        )
    if "a jni error has occurred" in lowered:
        return False, "Current Java runtime could not launch the RMC jar. Check Java version and installation."
    if output:
        return False, f"RMC jar failed to launch during preflight: {_compact_probe_message(output)}"
    return False, "RMC jar failed to launch during preflight without diagnostic output."


def run_rmc(
    *,
    rmc_jar: str | Path,
    rebeca_file: str | Path,
    property_file: str | Path,
    working_dir: str | Path,
    project_name: str,
    timeout_seconds: int = 300,
    version: str = "2.1",
    extension: str = "TimedRebeca",
    export_transition_system: bool = True,
) -> RMCResult:
    rmc_jar = Path(rmc_jar)
    rebeca_file = Path(rebeca_file)
    property_file = Path(property_file)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    start = time.time()

    java_path = shutil.which("java")
    gpp_path = shutil.which("g++")

    if not java_path:
        return _result("tool_error",project_name,working_dir,notes="java was not found in PATH.",warnings=warnings,elapsed=time.time() - start,)
    if not gpp_path:
        return _result("tool_error",project_name,working_dir,notes="g++ was not found in PATH.",warnings=warnings,elapsed=time.time() - start,)
    if not rmc_jar.exists():
        return _result("tool_error",project_name,working_dir, notes=f"RMC jar not found: {rmc_jar}", warnings=warnings, elapsed=time.time() - start,)
    if not rebeca_file.exists():
        return _result("tool_error", project_name, working_dir, notes=f"Input .rebeca file does not exist: {rebeca_file}", warnings=warnings, elapsed=time.time() - start, )
    if not property_file.exists():
        return _result("tool_error", project_name, working_dir, notes=f"Input .property file does not exist: {property_file}", warnings=warnings, elapsed=time.time() - start, )

    _clear_previous_artifacts(working_dir)

    normalized_extension = _normalize_extension(extension)
    if normalized_extension != extension:
        warnings.append(
            f"Normalized extension from `{extension}` to `{normalized_extension}` for RMC."
        )

    statespace_xml_path = working_dir / "statespace.xml"
    output_xml_path = working_dir / "output.xml"
    progress_path = working_dir / "progress"

    translator_cmd = [
        java_path,
        "-jar",
        str(rmc_jar),
        "-s",
        str(rebeca_file),
        "-p",
        str(property_file),
        "-o",
        str(working_dir),
        "-v",
        str(version),
        "-e",
        normalized_extension,
    ]
    if export_transition_system:
        translator_cmd.extend(["-x", str(statespace_xml_path)])

    try:
        translator_proc = subprocess.run(
            translator_cmd,
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=timeout_seconds,
            cwd=str(working_dir),
        )
    except subprocess.TimeoutExpired as exc:
        translator_log = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return _result(
            "timeout",
            project_name,
            working_dir,
            translator_log=translator_log,
            output_xml=output_xml_path,
            output_xml_text=_read_if_exists(output_xml_path),
            progress_text=_read_if_exists(progress_path),
            notes=f"RMC translation timed out after {timeout_seconds} seconds.",
            warnings=warnings,
            elapsed=time.time() - start,
        )

    translator_log = ((translator_proc.stdout or "") + "\n" + (translator_proc.stderr or "")).strip()

    cpp_files = sorted(working_dir.glob("*.cpp"))
    if translator_proc.returncode != 0 or not cpp_files:
        status = "syntax_error" if _looks_like_syntax_error(translator_log) else "tool_error"
        note = _translator_failure_note(translator_log)
        return _result(
            status,
            project_name,
            working_dir,
            translator_log=translator_log,
            output_xml=output_xml_path,
            output_xml_text=_read_if_exists(output_xml_path),
            progress_text=_read_if_exists(progress_path),
            notes=note,
            warnings=warnings,
            elapsed=time.time() - start,
        )

    executable_path = working_dir / "executable.exe"
    compile_cmd = [
        gpp_path,
        *[str(p) for p in cpp_files],
        "-w",
        "-o",
        str(executable_path),
    ]

    try:
        compile_proc = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=timeout_seconds,
            cwd=str(working_dir),
        )
    except subprocess.TimeoutExpired as exc:
        compile_log = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return _result(
            "timeout",
            project_name,
            working_dir,
            translator_log=translator_log,
            compile_log=compile_log,
            output_xml=output_xml_path,
            output_xml_text=_read_if_exists(output_xml_path),
            progress_text=_read_if_exists(progress_path),
            notes=f"g++ compilation timed out after {timeout_seconds} seconds.",
            warnings=warnings,
            elapsed=time.time() - start,
        )

    compile_log = ((compile_proc.stdout or "") + "\n" + (compile_proc.stderr or "")).strip()

    if compile_proc.returncode != 0 or not executable_path.exists():
        status = "syntax_error" if _looks_like_syntax_error(compile_log) else "tool_error"
        return _result(
            status,
            project_name,
            working_dir,
            translator_log=translator_log,
            compile_log=compile_log,
            output_xml=output_xml_path,
            output_xml_text=_read_if_exists(output_xml_path),
            progress_text=_read_if_exists(progress_path),
            notes="Generated C++ files could not be compiled.",
            warnings=warnings,
            elapsed=time.time() - start,
        )

    try:
        exec_proc = subprocess.run(
            [str(executable_path)],
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=timeout_seconds,
            cwd=str(working_dir),
        )
    except subprocess.TimeoutExpired as exc:
        execution_log = (exc.stdout or "") + "\n" + (exc.stderr or "")
        output_xml_text = _read_if_exists(output_xml_path)
        statespace_text = _read_if_exists(statespace_xml_path)
        progress_text = _read_if_exists(progress_path)
        return _result(
            "timeout",
            project_name,
            working_dir,
            translator_log=translator_log,
            compile_log=compile_log,
            execution_log=execution_log,
            output_xml=output_xml_path,
            output_xml_text=output_xml_text,
            statespace_xml=statespace_xml_path,
            statespace_xml_text=statespace_text,
            progress_text=progress_text,
            executable_path=executable_path,
            notes=f"Generated executable timed out after {timeout_seconds} seconds.",
            warnings=warnings,
            elapsed=time.time() - start,
        )

    execution_log = ((exec_proc.stdout or "") + "\n" + (exec_proc.stderr or "")).strip()
    output_xml_text = _read_if_exists(output_xml_path)
    statespace_text = _read_if_exists(statespace_xml_path)
    progress_text = _read_if_exists(progress_path)

    status = _classify_execution_log(
        translator_log=translator_log,
        compile_log=compile_log,
        execution_log=execution_log,
        output_xml_text=output_xml_text,
        statespace_text=statespace_text,
        progress_text=progress_text,
        exec_returncode=exec_proc.returncode,
    )

    notes = "RMC translation, C++ compilation, and executable run completed."
    if _completed_without_verdict(
        execution_log=execution_log,
        output_xml_text=output_xml_text,
        statespace_text=statespace_text,
        progress_text=progress_text,
        exec_returncode=exec_proc.returncode,
        status=status,
    ):
        warnings.append(
            "Executable completed successfully, but no recognizable verification verdict was found in output.xml, progress, or logs."
        )
        notes = "Executable completed without a recognized verification verdict."
    return _result(
        status,
        project_name,
        working_dir,
        translator_log=translator_log,
        compile_log=compile_log,
        execution_log=execution_log,
        output_xml=output_xml_path,
        output_xml_text=output_xml_text,
        statespace_xml=statespace_xml_path,
        statespace_xml_text=statespace_text,
        progress_text=progress_text,
        executable_path=executable_path,
        notes=notes,
        warnings=warnings,
        elapsed=time.time() - start,
    )


def _normalize_extension(extension: str) -> str:
    normalized = extension.strip().upper()
    mapping = {
        "COREREBECA": "CORE_REBECA",
        "CORE_REBECA": "CORE_REBECA",
        "TIMEDREBECA": "TIMED_REBECA",
        "TIMED_REBECA": "TIMED_REBECA",
        "PROBABILISTICREBECA": "PROBABILISTIC_REBECA",
        "PROBABILISTIC_REBECA": "PROBABILISTIC_REBECA",
        "PROBABILISTICTIMEREBECA": "PROBABILISTIC_TIME_REBECA",
        "PROBABILISTIC_TIME_REBECA": "PROBABILISTIC_TIME_REBECA",
    }
    return mapping.get(normalized, extension)


def _translator_failure_note(translator_log: str) -> str:
    lowered = (translator_log or "").lower()
    if looks_like_java_runtime_mismatch(translator_log):
        return "RMC translator could not start because the current Java runtime is too old for this RMC jar."
    if "a jni error has occurred" in lowered:
        return "RMC translator failed to launch under the current Java runtime."
    return "RMC translation failed or generated no C++ files."


def _compact_probe_message(text: str, max_chars: int = 220) -> str:
    stripped = " ".join((text or "").split())
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3] + "..."


def _result(
    status: str,
    project_name: str,
    working_dir: Path,
    *,
    translator_log: str = "",
    compile_log: str = "",
    execution_log: str = "",
    output_xml: Path | None = None,
    output_xml_text: str = "",
    statespace_xml: Path | None = None,
    statespace_xml_text: str = "",
    progress_text: str = "",
    executable_path: Path | None = None,
    notes: str = "",
    warnings: list[str] | None = None,
    elapsed: float | None = None,
) -> RMCResult:
    return RMCResult(
        status=status if status in VALID_STATUSES else "tool_error",
        project_name=project_name,
        working_dir=str(working_dir),
        translator_log=translator_log,
        compile_log=compile_log,
        execution_log=execution_log,
        output_xml=str(output_xml) if output_xml and output_xml.exists() else None,
        output_xml_text=output_xml_text,
        statespace_xml=str(statespace_xml) if statespace_xml and statespace_xml.exists() else None,
        statespace_xml_text=statespace_xml_text,
        progress_text=progress_text,
        executable_path=str(executable_path) if executable_path and executable_path.exists() else None,
        elapsed_seconds=elapsed,
        notes=notes,
        warnings=warnings or [],
    )


def _clear_previous_artifacts(working_dir: Path) -> None:
    for pattern in (
        "*.cpp",
        "*.h",
        "*.o",
        "*.obj",
        "executable",
        "executable.exe",
        "statespace.xml",
        "output.xml",
        "progress",
    ):
        for path in working_dir.glob(pattern):
            try:
                path.unlink()
            except Exception:
                pass


def _classify_execution_log(
    *,
    translator_log: str,
    compile_log: str,
    execution_log: str,
    output_xml_text: str,
    statespace_text: str,
    progress_text: str,
    exec_returncode: int,
) -> str:
    xml_result = _classify_output_xml(output_xml_text)
    if xml_result:
        return xml_result

    text = "\n".join(
        [
            translator_log or "",
            compile_log or "",
            execution_log or "",
            output_xml_text or "",
            statespace_text or "",
            progress_text or "",
        ]
    ).lower()

    if _looks_like_syntax_error(text):
        return "syntax_error"

    if "assertion failed" in text:
        return "assertion_failed"

    if "queue overflow" in text:
        return "state_explosion"

    if "deadlock" in text:
        return "deadlock"

    if "analysis result" in text and "satisfied" in text:
        return "satisfied"

    if "satisfied" in text and "analysis result" not in text:
        return "satisfied"

    if exec_returncode != 0 and ("deadlock" in text or "assertion" in text):
        if "assertion failed" in text:
            return "assertion_failed"
        return "deadlock"

    if exec_returncode == 0:
        return "tool_error"

    return "tool_error"


def _completed_without_verdict(
    *,
    execution_log: str,
    output_xml_text: str,
    statespace_text: str,
    progress_text: str,
    exec_returncode: int,
    status: str,
) -> bool:
    if exec_returncode != 0 or status != "tool_error":
        return False
    text = "\n".join(
        [
            execution_log or "",
            output_xml_text or "",
            statespace_text or "",
            progress_text or "",
        ]
    ).lower()
    if _classify_output_xml(output_xml_text):
        return False
    if "assertion failed" in text or "deadlock" in text:
        return False
    if "analysis result" in text and "satisfied" in text:
        return False
    if "satisfied" in text and "analysis result" not in text:
        return False
    return True


def _classify_output_xml(output_xml_text: str) -> str | None:
    lowered = output_xml_text.lower()
    if not lowered.strip():
        return None
    match = re.search(r"<result>\s*([^<]+?)\s*</result>", lowered)
    if not match:
        return None
    result = match.group(1).strip()
    mapping = {
        "satisfied": "satisfied",
        "deadlock": "deadlock",
        "queue overflow": "state_explosion",
        "queue_overflow": "state_explosion",
        "state explosion": "state_explosion",
        "assertion_failed": "assertion_failed",
        "assertion failed": "assertion_failed",
        "property_violation": "assertion_failed",
        "property violation": "assertion_failed",
        "timeout": "timeout",
    }
    return mapping.get(result)


def _looks_like_syntax_error(text: str) -> bool:
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "parse error",
            "syntax error",
            "compiler error",
            "parser error",
            "no viable alternative",
            "mismatched input",
            "unrecognized rebeca extension",
            "is undefined for the type",
            "type binding",
        )
    ):
        return True
    if re.search(r"(^|\n)\s*errors\s*:", lowered):
        return True
    if re.search(r"(^|\n).+\berror:", lowered):
        return True
    return False


def _read_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
