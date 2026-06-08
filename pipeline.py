from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyze_and_repair import build_repair_prompt
from generate_properties import PropertyGenerationResult, generate_properties
from generate_tr_package import TRPackageResult, generate_tr_package
from llm_client import BaseLLMClient, PromptTooLargeError, make_llm_client
from run_rmc import RMCResult, probe_rmc_runtime, run_rmc

ACCEPTED_RMC_STATUSES = {"satisfied", "deadlock"}


@dataclass
class IterationRecord:
    iteration: int
    candidate_property_json: Optional[str]
    candidate_property_raw: Optional[str]
    candidate_property_summary: Optional[str]
    rebeca_path: Optional[str]
    property_path: Optional[str]
    live_rebeca_path: Optional[str]
    live_property_path: Optional[str]
    tr_package_json: Optional[str]
    tr_package_raw: Optional[str]
    rmc_result_json: Optional[str]
    translator_log_path: Optional[str]
    compile_log_path: Optional[str]
    execution_log_path: Optional[str]
    output_xml_copy: Optional[str]
    statespace_xml_copy: Optional[str]
    progress_copy: Optional[str]
    analysis_variant_dir: Optional[str] = None
    pipeline_note: Optional[str] = None
    repair_case: Optional[str] = None
    repair_prompt_path: Optional[str] = None


@dataclass
class PipelineResult:
    project_name: str
    final_status: str
    iterations: List[IterationRecord]
    final_iteration: int
    run_root: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "final_status": self.final_status,
            "iterations": [asdict(iteration) for iteration in self.iterations],
            "final_iteration": self.final_iteration,
            "run_root": self.run_root,
        }


def prepare_project_structure(workspace_dir: Path, project_name: str) -> Path:
    project_dir = workspace_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    return project_dir


def is_accepted_rmc_status(status: str) -> bool:
    return status in ACCEPTED_RMC_STATUSES


def pipeline_final_status_from_rmc_status(status: str) -> str:
    if status == "deadlock":
        return "satisfied_with_deadlock"
    return status


def write_project_files(project_dir: Path, rebeca_text: str, property_text: str, project_name: str) -> Dict[str, str]:
    src_dir = project_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    rebeca_path = src_dir / f"{project_name}.rebeca"
    property_path = src_dir / f"{project_name}.property"
    rebeca_path.write_text(rebeca_text, encoding="utf-8")
    property_path.write_text(property_text, encoding="utf-8")
    return {"rebeca": str(rebeca_path), "property": str(property_path)}


def preflight_check(*, workspace_dir: Path, project_name: str, rmc_jar: Path, llm_client: BaseLLMClient, example_rebeca: str, example_property: str) -> Dict[str, Any]:
    checks: list[str] = []
    warnings: list[str] = []
    ok = True

    try:
        from generate_properties import build_prompt as gp_build
        gp_build("reactor X {}")
        checks.append("Property-generation prompt renders successfully.")
    except Exception as exc:
        ok = False
        warnings.append(f"Property-generation prompt failed to render: {exc}")

    try:
        from generate_tr_package import build_prompt as tr_build
        tr_build("reactor X {}", {"summary": "none", "properties": []})
        checks.append("TR-generation prompt renders successfully.")
    except Exception as exc:
        ok = False
        warnings.append(f"TR-generation prompt failed to render: {exc}")

    if workspace_dir.exists():
        checks.append("Workspace directory exists.")
    else:
        ok = False
        warnings.append(f"Workspace directory does not exist: {workspace_dir}")

    if rmc_jar.exists():
        checks.append("RMC jar exists.")
    else:
        ok = False
        warnings.append(f"RMC jar was not found: {rmc_jar}")

    project_src = workspace_dir / project_name / "src"
    project_src.mkdir(parents=True, exist_ok=True)
    checks.append(f"Project src directory is available at {project_src}.")

    if example_rebeca and example_property:
        checks.append("Example .rebeca and .property inputs are available.")
    else:
        warnings.append("No example .rebeca/.property files were provided; syntax quality may be less stable.")

    provider = llm_client.provider_name.lower().strip()
    explicit_key = getattr(llm_client, "api_key", None)
    java_path = shutil.which("java")
    gpp_path = shutil.which("g++")

    if java_path:
        checks.append("java is available in PATH.")
    else:
        ok = False
        warnings.append("java was not found in PATH.")

    if gpp_path:
        checks.append("g++ is available in PATH.")
    else:
        ok = False
        warnings.append("g++ was not found in PATH.")

    if java_path and rmc_jar.exists():
        runtime_ok, runtime_message = probe_rmc_runtime(rmc_jar=rmc_jar, java_path=java_path)
        if runtime_ok:
            checks.append(runtime_message)
        else:
            ok = False
            warnings.append(runtime_message)

    if provider == "openai":
        if explicit_key or os.getenv("OPENAI_API_KEY"):
            checks.append("OpenAI API key is configured.")
        else:
            ok = False
            warnings.append("OpenAI API key is missing from OPENAI_API_KEY and no explicit api_key was validated.")
    else:
        if explicit_key or os.getenv("ANTHROPIC_API_KEY"):
            checks.append("Anthropic API key is configured.")
        else:
            ok = False
            warnings.append("Anthropic API key is missing from ANTHROPIC_API_KEY and no explicit api_key was validated.")

    return {"ok": ok, "checks": checks, "warnings": warnings}


def discover_example_pair(workspace_dir: Path, lf_file: Path) -> tuple[str, str]:
    candidates: list[tuple[int, Path, Path]] = []
    lf_stem = lf_file.stem.lower()
    for rebeca_path in workspace_dir.rglob("*.rebeca"):
        property_path = rebeca_path.with_suffix(".property")
        if not property_path.exists():
            continue
        score = 0
        project_name = rebeca_path.stem.lower()
        full_path = str(rebeca_path).lower()
        if "baseline" in full_path:
            score += 10
        if lf_stem and lf_stem in project_name:
            score += 8
        if rebeca_path.parent.name.lower() == "src":
            score += 3
        candidates.append((score, rebeca_path, property_path))

    if not candidates:
        return "", ""

    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    best_rebeca, best_property = candidates[0][1], candidates[0][2]
    return (
        best_rebeca.read_text(encoding="utf-8"),
        best_property.read_text(encoding="utf-8"),
    )


def run_pipeline(
    lf_code: str,
    project_name: str,
    workspace_dir: str | Path,
    rmc_jar: str | Path,
    llm_client: BaseLLMClient,
    example_rebeca: str = "",
    example_property: str = "",
    timeout_seconds: int = 300,
    max_iterations: int = 3,
    rmc_version: str = "2.1",
    rmc_extension: str = "TimedRebeca",
) -> PipelineResult:
    workspace_dir = Path(workspace_dir)
    rmc_jar = Path(rmc_jar)
    project_dir = prepare_project_structure(workspace_dir, project_name)
    run_root = project_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    shared_debug_dir = run_root / "shared"
    shared_debug_dir.mkdir(parents=True, exist_ok=True)

    preflight = preflight_check(
        workspace_dir=workspace_dir,
        project_name=project_name,
        rmc_jar=rmc_jar,
        llm_client=llm_client,
        example_rebeca=example_rebeca,
        example_property=example_property,
    )
    print(json.dumps({"preflight": preflight}, indent=2))
    if not preflight["ok"]:
        result = PipelineResult(project_name=project_name, final_status="tool_error", iterations=[], final_iteration=0, run_root=str(run_root))
        (project_dir / f"{project_name}.run_summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    try:
        prop_result = generate_properties(
            lf_code,
            llm_client,
            debug_dir=shared_debug_dir,
            debug_name=f"{project_name}.candidate_properties",
        )
    except PromptTooLargeError:
        result = PipelineResult(
            project_name=project_name,
            final_status="prompt_too_large",
            iterations=[],
            final_iteration=0,
            run_root=str(run_root),
        )
        (project_dir / f"{project_name}.run_summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result
    candidate_payload = {"summary": prop_result.to_prompt_summary(), "properties": [asdict(prop) for prop in prop_result.properties]}

    iterations: list[IterationRecord] = []
    failure_context = "None"
    final_status = "tool_error"

    for iteration in range(1, max_iterations + 1):
        iter_dir = run_root / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        property_paths = _write_candidate_property_artifacts(iter_dir, project_name, prop_result)

        try:
            tr_result: TRPackageResult = generate_tr_package(
                lf_code=lf_code,
                candidate_properties=candidate_payload,
                llm_client=llm_client,
                example_rebeca=example_rebeca,
                example_property=example_property,
                failure_context=failure_context,
                debug_dir=iter_dir,
                debug_name=project_name,
                allow_property_normalization=True,
            )
        except PromptTooLargeError as exc:
            record = IterationRecord(
                iteration=iteration,
                candidate_property_json=property_paths["json"],
                candidate_property_raw=property_paths["raw"],
                candidate_property_summary=property_paths["summary"],
                rebeca_path=None,
                property_path=None,
                live_rebeca_path=None,
                live_property_path=None,
                tr_package_json=None,
                tr_package_raw=None,
                rmc_result_json=None,
                translator_log_path=None,
                compile_log_path=None,
                execution_log_path=None,
                output_xml_copy=None,
                statespace_xml_copy=None,
                progress_copy=None,
                analysis_variant_dir=None,
                pipeline_note=str(exc),
                repair_case="repair_context_too_large",
                repair_prompt_path=None,
            )
            (iter_dir / "iteration_record.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
            iterations.append(record)
            final_status = "prompt_too_large"
            break
        except ValueError as exc:
            local_result = RMCResult(
                status="syntax_error",
                project_name=project_name,
                working_dir=str(iter_dir / "rmc"),
                translator_log=str(exc),
                notes="Local validation rejected the generated TR package before invoking RMC.",
            )
            rmc_paths = _write_rmc_artifacts(iter_dir, project_name, local_result)
            repair = build_repair_prompt(
                result=local_result,
                lf_code=lf_code,
                previous_rebeca=(iter_dir / f"{project_name}.rebeca.extracted.txt").read_text(encoding="utf-8", errors="ignore") if (iter_dir / f"{project_name}.rebeca.extracted.txt").exists() else "",
                previous_property=(iter_dir / f"{project_name}.property.extracted.txt").read_text(encoding="utf-8", errors="ignore") if (iter_dir / f"{project_name}.property.extracted.txt").exists() else "",
                baseline_analysis="",
            )
            repair_prompt_path = str(iter_dir / f"{project_name}.repair_prompt.txt")
            Path(repair_prompt_path).write_text(repair.repair_prompt, encoding="utf-8")
            record = IterationRecord(
                iteration=iteration,
                candidate_property_json=property_paths["json"],
                candidate_property_raw=property_paths["raw"],
                candidate_property_summary=property_paths["summary"],
                rebeca_path=str(iter_dir / f"{project_name}.rebeca.extracted.txt") if (iter_dir / f"{project_name}.rebeca.extracted.txt").exists() else None,
                property_path=str(iter_dir / f"{project_name}.property.extracted.txt") if (iter_dir / f"{project_name}.property.extracted.txt").exists() else None,
                live_rebeca_path=None,
                live_property_path=None,
                tr_package_json=None,
                tr_package_raw=str(iter_dir / f"{project_name}.raw_response.txt") if (iter_dir / f"{project_name}.raw_response.txt").exists() else None,
                rmc_result_json=rmc_paths["json"],
                translator_log_path=rmc_paths["translator_log"],
                compile_log_path=rmc_paths["compile_log"],
                execution_log_path=rmc_paths["execution_log"],
                output_xml_copy=rmc_paths["output_xml"],
                statespace_xml_copy=rmc_paths["statespace_xml"],
                progress_copy=rmc_paths["progress"],
                analysis_variant_dir=None,
                pipeline_note=None,
                repair_case=repair.detail_reason or repair.failure_case,
                repair_prompt_path=repair_prompt_path,
            )
            (iter_dir / "iteration_record.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
            iterations.append(record)
            failure_context = repair.repair_prompt
            final_status = local_result.status
            continue

        tr_paths = tr_result.save(iter_dir, project_name)

        live_files = write_project_files(project_dir=project_dir, rebeca_text=tr_result.rebeca_code, property_text=tr_result.property_code, project_name=project_name)

        rmc_work_dir = iter_dir / "rmc"
        rmc_result: RMCResult = run_rmc(
            rmc_jar=rmc_jar,
            rebeca_file=Path(tr_paths["rebeca"]),
            property_file=Path(tr_paths["property"]),
            working_dir=rmc_work_dir,
            project_name=project_name,
            timeout_seconds=timeout_seconds,
            version=rmc_version,
            extension=rmc_extension,
            export_transition_system=True,
        )
        rmc_paths = _write_rmc_artifacts(iter_dir, project_name, rmc_result)

        repair_prompt_path = None
        repair_case = None
        pipeline_note = None
        stop_after_iteration = False
        if is_accepted_rmc_status(rmc_result.status):
            failure_context = "None"
            final_status = pipeline_final_status_from_rmc_status(rmc_result.status)
            if rmc_result.status == "deadlock":
                repair_case = "accepted_deadlock"
                pipeline_note = (
                    "Pipeline policy accepted the raw RMC deadlock result as a terminal successful outcome. "
                    "The raw RMC status remains `deadlock`."
                )
            else:
                pipeline_note = "Pipeline policy accepted the raw RMC satisfied result as a terminal successful outcome."
        else:
            repair = build_repair_prompt(result=rmc_result, lf_code=lf_code, previous_rebeca=tr_result.rebeca_code, previous_property=tr_result.property_code, baseline_analysis=tr_result.model_analysis)
            repair_case = repair.detail_reason or repair.failure_case
            repair_prompt_path = str(iter_dir / f"{project_name}.repair_prompt.txt")
            Path(repair_prompt_path).write_text(repair.repair_prompt, encoding="utf-8")
            final_status = rmc_result.status
            if repair.should_repair_baseline:
                failure_context = repair.repair_prompt
            else:
                failure_context = "None"
                stop_after_iteration = True
                pipeline_note = (
                    "Strict LF-faithful baseline was preserved. "
                    "The failure was classified as a non-repairable baseline limitation, so no further baseline-repair iteration was started. "
                    "Any analysis_bounded artifact remains separate and is not counted as the baseline result."
                )

        record = IterationRecord(
            iteration=iteration,
            candidate_property_json=property_paths["json"],
            candidate_property_raw=property_paths["raw"],
            candidate_property_summary=property_paths["summary"],
            rebeca_path=tr_paths["rebeca"],
            property_path=tr_paths["property"],
            live_rebeca_path=live_files["rebeca"],
            live_property_path=live_files["property"],
            tr_package_json=tr_paths["json"],
            tr_package_raw=tr_paths["raw_output"],
            rmc_result_json=rmc_paths["json"],
            translator_log_path=rmc_paths["translator_log"],
            compile_log_path=rmc_paths["compile_log"],
            execution_log_path=rmc_paths["execution_log"],
            output_xml_copy=rmc_paths["output_xml"],
            statespace_xml_copy=rmc_paths["statespace_xml"],
            progress_copy=rmc_paths["progress"],
            analysis_variant_dir=tr_paths.get("analysis_variant_dir"),
            pipeline_note=pipeline_note,
            repair_case=repair_case,
            repair_prompt_path=repair_prompt_path,
        )
        (iter_dir / "iteration_record.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        iterations.append(record)
        if is_accepted_rmc_status(rmc_result.status):
            break
        if stop_after_iteration:
            break

    result = PipelineResult(project_name=project_name, final_status=final_status, iterations=iterations, final_iteration=len(iterations), run_root=str(run_root))
    (project_dir / f"{project_name}.run_summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _write_candidate_property_artifacts(iter_dir: Path, project_name: str, prop_result: PropertyGenerationResult) -> Dict[str, str]:
    json_path = iter_dir / f"{project_name}.candidate_properties.json"
    raw_path = iter_dir / f"{project_name}.candidate_properties.raw.txt"
    summary_path = iter_dir / f"{project_name}.candidate_properties.summary.txt"
    prop_result.save_json(json_path)
    raw_path.write_text(prop_result.raw_response, encoding="utf-8")
    summary_path.write_text(prop_result.to_prompt_summary(), encoding="utf-8")
    return {"json": str(json_path), "raw": str(raw_path), "summary": str(summary_path)}


def _write_rmc_artifacts(iter_dir: Path, project_name: str, rmc_result: RMCResult) -> Dict[str, Optional[str]]:
    result_json_path = iter_dir / f"{project_name}.rmc_result.json"
    translator_log_path = iter_dir / f"{project_name}.rmc.translator.log.txt"
    compile_log_path = iter_dir / f"{project_name}.rmc.compile.log.txt"
    execution_log_path = iter_dir / f"{project_name}.rmc.execution.log.txt"
    output_xml_copy = iter_dir / f"{project_name}.output.xml.txt"
    statespace_xml_copy = iter_dir / f"{project_name}.statespace.xml.txt"
    progress_copy = iter_dir / f"{project_name}.progress.txt"

    result_json_path.write_text(json.dumps(rmc_result.to_dict(), indent=2), encoding="utf-8")
    translator_log_path.write_text(rmc_result.translator_log or "", encoding="utf-8")
    compile_log_path.write_text(rmc_result.compile_log or "", encoding="utf-8")
    execution_log_path.write_text(rmc_result.execution_log or "", encoding="utf-8")
    output_xml_copy.write_text(rmc_result.output_xml_text or "", encoding="utf-8")
    statespace_xml_copy.write_text(rmc_result.statespace_xml_text or "", encoding="utf-8")
    progress_copy.write_text(rmc_result.progress_text or "", encoding="utf-8")

    return {
        "json": str(result_json_path),
        "translator_log": str(translator_log_path),
        "compile_log": str(compile_log_path),
        "execution_log": str(execution_log_path),
        "output_xml": str(output_xml_copy),
        "statespace_xml": str(statespace_xml_copy),
        "progress": str(progress_copy),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the LF -> properties -> TR -> RMC iteration pipeline.")
    parser.add_argument("lf_file", help="Path to LF source file")
    parser.add_argument("project_name", help="Project/output name")
    parser.add_argument("workspace_dir", help="Workspace directory where artifacts and live project live")
    parser.add_argument("rmc_jar", help="Path to rmc-2.14.jar")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-5.4"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--example-rebeca", default="")
    parser.add_argument("--example-property", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--rmc-version", default="2.1")
    parser.add_argument("--rmc-extension", default="TimedRebeca")
    args = parser.parse_args()

    lf_code = Path(args.lf_file).read_text(encoding="utf-8")
    lf_file = Path(args.lf_file)
    workspace_dir = Path(args.workspace_dir)
    example_rebeca = Path(args.example_rebeca).read_text(encoding="utf-8") if args.example_rebeca else ""
    example_property = Path(args.example_property).read_text(encoding="utf-8") if args.example_property else ""
    if not example_rebeca or not example_property:
        discovered_rebeca, discovered_property = discover_example_pair(workspace_dir, lf_file)
        if not example_rebeca:
            example_rebeca = discovered_rebeca
        if not example_property:
            example_property = discovered_property
    client = make_llm_client(args.provider, args.model, api_key=args.api_key)
    result = run_pipeline(
        lf_code=lf_code,
        project_name=args.project_name,
        workspace_dir=workspace_dir,
        rmc_jar=args.rmc_jar,
        llm_client=client,
        example_rebeca=example_rebeca,
        example_property=example_property,
        timeout_seconds=args.timeout,
        max_iterations=args.max_iterations,
        rmc_version=args.rmc_version,
        rmc_extension=args.rmc_extension,
    )
    print(json.dumps(result.to_dict(), indent=2))
