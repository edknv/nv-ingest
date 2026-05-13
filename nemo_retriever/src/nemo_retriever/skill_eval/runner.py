# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trial runner: build sandboxed workdir, spawn `claude -p`, parse outputs."""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import stat
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from nemo_retriever.skill_eval.dataset import DatasetEntry

logger = logging.getLogger(__name__)

CONDITIONS = ("c1_base", "c2_retriever", "c3_retriever_skill")


@functools.lru_cache(maxsize=8)
def _load_prompt_template(name: str) -> str:
    return Path(str(pkg_files("nemo_retriever.skill_eval").joinpath(f"prompts/{name}"))).read_text(encoding="utf-8")


@dataclass
class TrialResult:
    trial_id: str
    condition: str
    entry_id: int
    query_id: int
    status: str
    extraction_method: str
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    total_cost_usd: float
    model_id: str
    session_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0
    final_answer: str = ""
    ranked_retrieved: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retriever_first_use_turn: int | None = None
    retriever_used_ever: bool = False
    skill_fired: bool | None = None
    is_setup: bool = False


def _render_prompt(entry: DatasetEntry, condition: str) -> str:
    tpl_name = "trial_user_slash.j2" if condition == "c3_retriever_skill" else "trial_user_nl.j2"
    text = _load_prompt_template(tpl_name)
    return text.replace("{{ paraphrased_prompt }}", entry.paraphrased_prompt).replace(
        "{{ original_query }}", entry.original_query
    )


def _render_setup_prompt(condition: str) -> str:
    tpl_name = "setup_slash.j2" if condition == "c3_retriever_skill" else "setup_nl.j2"
    return _load_prompt_template(tpl_name)


def _build_pdf_symlinks(pdf_source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(pdf_source.glob("*.pdf")):
        target = dest / pdf.name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(pdf.resolve())


def _write_shim(shim_dir: Path, name: str) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / name
    shim.write_text(
        "#!/usr/bin/env bash\n" f"echo 'skill_eval shim: {name} not available in this trial' >&2\n" "exit 127\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _copy_skill(skill_source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "SKILL.md").exists():
        return
    shutil.copy2(skill_source / "SKILL.md", dest / "SKILL.md")
    ref_src = skill_source / "references"
    if ref_src.is_dir():
        shutil.copytree(ref_src, dest / "references", dirs_exist_ok=True)


def _build_condition_workdir(
    condition: str,
    root: Path,
    pdf_source: Path,
    skill_source: Path,
) -> Path:
    """Build one workdir per condition. Shared across all turns in the session.

    Workdir contents:
      - pdfs/ symlink farm into the source PDF folder
      - .claude/ sandbox (settings + per-condition skill copy)
      - .bin/retriever shim (c1 only) so retriever is unavailable on PATH

    The agent itself creates any retrieval artifacts (e.g., ./lancedb/) inside the
    workdir on the setup turn.
    """
    workdir = root / f"{condition}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    _build_pdf_symlinks(pdf_source, workdir / "pdfs")
    (workdir / ".claude").mkdir(parents=True, exist_ok=True)
    (workdir / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    # c2 and c3 both have retriever installed AND the nemo-retriever skill loaded.
    # The c2/c3 distinction is purely the prompt style (NL vs explicit slash command).
    if condition in ("c2_retriever", "c3_retriever_skill"):
        _copy_skill(skill_source, workdir / ".claude" / "skills" / "nemo-retriever")
    if condition == "c1_base":
        _write_shim(workdir / ".bin", "retriever")
    return workdir


def cleanup_condition_workdir(workdir: Path) -> None:
    """Remove a condition's scratch workdir (PDFs symlinks, .claude/, agent-built
    artifacts like .venv/, lancedb/, scratch scripts). Called after a session
    completes and its results have been persisted to the artifact dir.
    """
    if not workdir.exists():
        return
    shutil.rmtree(workdir, ignore_errors=True)
    logger.info("cleaned up workdir %s", workdir)


def _env_for(condition: str, workdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    if condition == "c1_base":
        env["PATH"] = f"{workdir / '.bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def _build_command(
    condition: str,
    model: str,
    budget_usd: float,
    session_uuid: str,
    workdir: Path,
    *,
    resume: bool = False,
) -> list[str]:
    """Build the `claude -p` command. First turn uses --session-id; subsequent turns use --resume.

    We deliberately do NOT pass --no-session-persistence because multi-turn requires
    the session to persist between subprocess invocations.
    """
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--add-dir",
        str(workdir),
        "--permission-mode",
        "bypassPermissions",
        "--max-budget-usd",
        str(budget_usd),
        "--setting-sources",
        "project",
        "--allow-dangerously-skip-permissions",
    ]
    if resume:
        cmd.extend(["--resume", session_uuid])
    else:
        cmd.extend(["--session-id", session_uuid])
    # Only c1 disables skills entirely. c2 has the skill loaded but uses NL prompt
    # (relying on description-based auto-discovery); c3 explicitly invokes via slash.
    if condition == "c1_base":
        cmd.append("--disable-slash-commands")
    return cmd


def _parse_envelope(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        logger.warning("could not parse claude envelope (first 200 chars): %r", raw[:200])
        return {}


def _populate_tokens(result: TrialResult, envelope: dict[str, Any]) -> None:
    usage = envelope.get("usage") or {}
    result.input_tokens = int(usage.get("input_tokens") or 0)
    result.output_tokens = int(usage.get("output_tokens") or 0)
    result.cache_read_input_tokens = int(usage.get("cache_read_input_tokens") or 0)
    result.cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    cache_detail = usage.get("cache_creation") or {}
    result.ephemeral_5m_input_tokens = int(cache_detail.get("ephemeral_5m_input_tokens") or 0)
    result.ephemeral_1h_input_tokens = int(cache_detail.get("ephemeral_1h_input_tokens") or 0)


def _parse_output_json(workdir: Path) -> tuple[str, list[dict[str, Any]], str, list[str]]:
    out_path = workdir / "output.json"
    errors: list[str] = []
    if not out_path.exists():
        return "", [], "missing", ["output.json not written"]
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return "", [], "invalid_json", [f"invalid JSON: {e}"]
    if not isinstance(payload, dict):
        return "", [], "invalid_json", [f"top-level must be an object, got {type(payload).__name__}"]
    for required in ("final_answer", "ranked_retrieved"):
        if required not in payload:
            errors.append(f"missing required key '{required}'")
    ranked = payload.get("ranked_retrieved") or []
    if not isinstance(ranked, list):
        errors.append(f"ranked_retrieved must be a list, got {type(ranked).__name__}")
        ranked = []
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(ranked, start=1):
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        page = item.get("page_number")
        if doc_id is None or page is None:
            continue
        cleaned.append({"doc_id": str(doc_id), "page_number": int(page), "rank": int(item.get("rank") or i)})
    return str(payload.get("final_answer") or ""), cleaned, ("ok" if not errors else "schema_warning"), errors


def _extract_model_id(envelope: dict[str, Any], fallback: str) -> str:
    model_usage = envelope.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        return next(iter(model_usage.keys()))
    return str(envelope.get("model") or fallback)


def _scan_transcript_for_signals(envelope: dict[str, Any]) -> tuple[int | None, bool]:
    """Inspect the envelope's `result` text for `retriever` CLI usage."""
    text = str(envelope.get("result") or "")
    used = "retriever " in text or "\nretriever\n" in text
    first_use = 1 if used else None
    return first_use, used


def _run_one_turn(
    *,
    condition: str,
    prompt: str,
    trial_id: str,
    entry_id: int,
    query_id: int,
    is_setup: bool,
    turn_idx: int,
    workdir: Path,
    session_uuid: str,
    cmd: list[str],
    env: dict[str, str],
    timeout_s: int,
    model: str,
) -> TrialResult:
    """Execute one turn. Query turns (is_setup=False) expect the agent to write
    ./output.json; the setup turn does not."""
    out_path = workdir / "output.json"
    if out_path.exists():
        out_path.unlink()

    label = "setup" if is_setup else f"entry_id={entry_id}, query_id={query_id}"
    logger.info("turn %d for %s (%s)", turn_idx + 1, condition, label)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(workdir),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TrialResult(
            trial_id=trial_id,
            condition=condition,
            entry_id=entry_id,
            query_id=query_id,
            status="timeout",
            extraction_method="none",
            duration_ms=int((time.monotonic() - t0) * 1000),
            duration_api_ms=0,
            num_turns=turn_idx + 1,
            total_cost_usd=0.0,
            model_id=model,
            session_id=session_uuid,
            errors=[f"turn exceeded {timeout_s}s wall timeout"],
            is_setup=is_setup,
        )

    envelope = _parse_envelope(proc.stdout)
    stderr = proc.stderr.strip()
    result = TrialResult(
        trial_id=trial_id,
        condition=condition,
        entry_id=entry_id,
        query_id=query_id,
        status="ok" if proc.returncode == 0 and not envelope.get("is_error", False) else "error",
        extraction_method="n/a" if is_setup else "output_json",
        duration_ms=int(envelope.get("duration_ms") or (time.monotonic() - t0) * 1000),
        duration_api_ms=int(envelope.get("duration_api_ms") or 0),
        num_turns=turn_idx + 1,
        total_cost_usd=float(envelope.get("total_cost_usd") or 0.0),
        model_id=_extract_model_id(envelope, fallback=model),
        session_id=str(envelope.get("session_id") or session_uuid),
        is_setup=is_setup,
    )
    _populate_tokens(result, envelope)
    if proc.returncode != 0:
        result.errors.append(f"non-zero exit {proc.returncode}")
    if envelope.get("is_error"):
        result.errors.append(f"envelope is_error: {envelope.get('subtype') or '?'}")
    if stderr:
        result.errors.append(f"stderr: {stderr[:500]}")

    if not is_setup:
        answer, ranked, extract_status, extract_errors = _parse_output_json(workdir)
        if extract_status in ("missing", "invalid_json"):
            result.extraction_method = extract_status
            if result.status == "ok":
                result.status = "extraction_failed"
        elif extract_status == "schema_warning":
            result.extraction_method = "schema_warning"
        result.final_answer = answer
        result.ranked_retrieved = ranked
        result.errors.extend(extract_errors)
        if out_path.exists():
            out_path.rename(workdir / f"output_e{entry_id}.json")

    first_use, used = _scan_transcript_for_signals(envelope)
    result.retriever_first_use_turn = first_use
    result.retriever_used_ever = used
    # c1 has the skill unavailable; leave skill_fired=None to distinguish from "loaded but didn't fire".
    if condition in ("c2_retriever", "c3_retriever_skill"):
        result.skill_fired = used and (first_use is not None) and first_use <= 2
    return result


def run_condition(
    *,
    condition: str,
    entries: list[DatasetEntry],
    workdir_root: Path,
    pdf_source: Path,
    skill_source: Path,
    model: str,
    budget_usd: float,
    timeout_s: int,
) -> tuple[Path, list[TrialResult]]:
    """Run one Claude Code session covering setup + all `entries` for `condition`.

    Turn 1 creates the session via --session-id; subsequent turns resume it. The
    first TrialResult has is_setup=True; the rest are query results, one per entry.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    workdir = _build_condition_workdir(condition, workdir_root, pdf_source, skill_source)
    session_uuid = str(uuid.uuid4())
    env = _env_for(condition, workdir)
    logger.info("starting session for %s: workdir=%s session_id=%s", condition, workdir, session_uuid)

    results: list[TrialResult] = []

    setup_cmd = _build_command(condition, model, budget_usd, session_uuid, workdir, resume=False)
    setup_result = _run_one_turn(
        condition=condition,
        prompt=_render_setup_prompt(condition),
        trial_id=f"{condition}_setup_t1",
        entry_id=0,
        query_id=0,
        is_setup=True,
        turn_idx=0,
        workdir=workdir,
        session_uuid=session_uuid,
        cmd=setup_cmd,
        env=env,
        timeout_s=timeout_s,
        model=model,
    )
    results.append(setup_result)

    resume_cmd = _build_command(condition, model, budget_usd, session_uuid, workdir, resume=True)
    for i, entry in enumerate(entries):
        turn_idx = i + 1
        result = _run_one_turn(
            condition=condition,
            prompt=_render_prompt(entry, condition),
            trial_id=f"{condition}_e{entry.entry_id}_t{turn_idx + 1}",
            entry_id=entry.entry_id,
            query_id=entry.query_id,
            is_setup=False,
            turn_idx=turn_idx,
            workdir=workdir,
            session_uuid=session_uuid,
            cmd=resume_cmd,
            env=env,
            timeout_s=timeout_s,
            model=model,
        )
        results.append(result)
    return workdir, results


def save_trial(result: TrialResult, session_dir: Path) -> Path:
    out = session_dir / "trials" / result.condition / f"{result.trial_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return out
