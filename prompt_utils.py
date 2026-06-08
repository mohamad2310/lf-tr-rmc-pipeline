from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

APPROX_CHARS_PER_TOKEN = 4
MAX_REPAIR_CONTEXT_CHARS = 24_000
MAX_PROMPT_CHARS = 120_000
MAX_DEBUG_PROMPT_SAVE_CHARS = 80_000


def approximate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text or "") / APPROX_CHARS_PER_TOKEN))


def truncate_text(text: str, max_chars: int, *, keep_head: float = 0.75) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head_chars = max(0, int(max_chars * keep_head))
    tail_chars = max(0, max_chars - head_chars - 64)
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f"\n\n... [truncated {omitted} chars] ...\n\n"
        + (text[-tail_chars:] if tail_chars > 0 else "")
    )


def build_prompt_diagnostics(prompt: str, sections: Mapping[str, str]) -> dict[str, object]:
    return {
        "char_count": len(prompt),
        "approx_token_count": approximate_token_count(prompt),
        "sections": [
            {
                "name": name,
                "char_count": len(content or ""),
                "approx_token_count": approximate_token_count(content or ""),
            }
            for name, content in sections.items()
        ],
    }


def write_prompt_debug(
    *,
    debug_dir: str | Path,
    debug_name: str,
    prompt: str,
    sections: Mapping[str, str],
    max_prompt_save_chars: int = MAX_DEBUG_PROMPT_SAVE_CHARS,
) -> dict[str, object]:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    prompt_to_save = truncate_text(prompt, max_prompt_save_chars)
    diagnostics = build_prompt_diagnostics(prompt, sections)
    diagnostics["prompt_was_truncated_for_debug"] = len(prompt_to_save) < len(prompt)
    diagnostics["saved_prompt_char_count"] = len(prompt_to_save)

    (debug_dir / f"{debug_name}.prompt.txt").write_text(prompt_to_save, encoding="utf-8")
    (debug_dir / f"{debug_name}.prompt.metrics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    return diagnostics
