# LF -> Timed Rebeca -> RMC Pipeline

This pipeline automates the workflow from Lingua Franca (LF) programs to Timed Rebeca verification with RMC.

The goal is to:

1. Read and analyze an LF program
2. Derive candidate safety properties from the LF code
3. Generate a baseline Timed Rebeca `.rebeca` model
4. Generate a matching `.property` file
5. Run RMC directly instead of using the AFRA GUI manually
6. Compile the generated C++ code with `g++`
7. Run the generated executable
8. Save all logs and artifacts
9. If verification fails, classify the failure and generate a repair prompt for the next iteration

The pipeline is designed to be benchmark-independent. It should reason from the LF structure instead of hardcoding names such as ADAS, Pipe, Camera, Door, Vision, or other specific examples.

## Baseline Policy

- The pipeline always generates and checks the strict LF-faithful baseline first.
- It does not add artificial loops, artificial bounds, saturation, modulo, clamping, or queue-size workarounds to force the baseline toward `satisfied`.
- Failures are classified before any repair step is attempted.
- The strict baseline is only repaired when the evidence points to a mapping/translation problem, such as timer mapping or presence/value mapping errors.
- If the failure is caused by periodic unbounded behavior, closed-loop feedback, or state-space limitations, the baseline semantics are preserved and the pipeline does not silently rewrite the model just to make model checking finish.
- Any bounded abstraction must be treated as a separate analysis-oriented artifact, not as the baseline result.

## Optional Bounded Variant

- When useful, the generator may emit a separate `analysis_bounded` artifact alongside the baseline.
- That variant must carry the warning: `This is an analysis-oriented bounded abstraction, not the strict LF-faithful baseline.`
- Results from `analysis_bounded` are never counted as the strict baseline verification result.

## Pipeline Success Policy

The pipeline distinguishes between the raw RMC result and the pipeline-level final interpretation.

- Raw RMC status `satisfied` is accepted directly and the pipeline stops.
- Raw RMC status `deadlock` is also accepted by pipeline policy and the pipeline stops immediately.
- When raw RMC status is `deadlock`, the pipeline final status is written as `satisfied_with_deadlock`.
- The raw RMC result JSON still keeps the original RMC status `deadlock`.
- This is a pipeline policy decision, not a change to RMC semantics.

The accepted-status policy only applies to:
- `satisfied`
- `deadlock`

All other statuses still use the normal repair/iteration logic, including:
- `assertion_failed`
- `syntax_error`
- `tool_error`
- `timeout`
- `state_explosion`

## Required tools
- Python 3.10+
- Java in PATH (`java -version`)
  - The current preflight checks that the active `java` can actually launch the configured RMC jar.
  - With the current `rmc-2.14.jar`, this effectively means Java 17+.
- g++ in PATH (`g++ --version`)
- `rmc-2.14.jar`
- OpenAI or Anthropic SDK installed

## Example run (Windows CMD)

```bat
cd /d "C:\path\to\lf-tr-rmc-pipeline"
set OPENAI_API_KEY=YOUR_KEY
python pipeline.py "C:\path\to\lf_cases\Pipe.lf" autoPipe "C:\path\to\afra\workspace" "C:\path\to\rmc-2.14.jar" --provider openai --model YOUR_MODEL --timeout 300 --max-iterations 2 --rmc-version 2.1 --rmc-extension TimedRebeca
```

## Output
Artifacts are saved under:
- `<workspace>/<project>/runs/iter_01/`
- `<workspace>/<project>/runs/iter_02/`

Each iteration includes:
- candidate properties JSON/raw/summary
- generated `.rebeca` and `.property`
- RMC translator log
- g++ compile log
- execution log
- statespace dump
- repair prompt
