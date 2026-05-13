# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-trial results into a per-condition session summary."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from nemo_retriever.harness.artifacts import write_session_summary
from nemo_retriever.skill_eval.dataset import DatasetEntry
from nemo_retriever.skill_eval.runner import CONDITIONS, TrialResult
from nemo_retriever.skill_eval.score import recall_at_k


METRIC_KS = (1, 5, 10)
RECALL_KEYS = tuple(f"recall_{k}" for k in METRIC_KS)


def _relevant_set(entry: DatasetEntry) -> set[tuple[str, int]]:
    return {(p.doc_id, p.page_number) for p in entry.ground_truth_pages}


def _ranked_pairs(result: TrialResult) -> list[tuple[str, int]]:
    items = sorted(result.ranked_retrieved, key=lambda x: x.get("rank", 999))
    return [(str(item["doc_id"]), int(item["page_number"])) for item in items]


def overall_recall(
    results: Iterable[TrialResult],
    entries_by_id: dict[int, DatasetEntry],
    ks: tuple[int, ...] = METRIC_KS,
) -> dict[str, float]:
    """Macro-averaged recall@k: mean of per-query recall@k across query turns.

    Matches the aggregation used by `recall/beir.py:compute_beir_metrics` (which
    `retriever harness` runs), so skill_eval numbers are directly comparable to
    harness BEIR output.
    """
    per_query: dict[int, list[float]] = {k: [] for k in ks}
    for r in results:
        if r.is_setup:
            continue
        entry = entries_by_id.get(r.entry_id)
        if entry is None:
            continue
        relevant = _relevant_set(entry)
        if not relevant:
            continue
        ranked = _ranked_pairs(r)
        for k in ks:
            per_query[k].append(recall_at_k(ranked, relevant, k))
    return {f"recall_{k}": (sum(v) / len(v)) if v else 0.0 for k, v in per_query.items()}


def aggregate_condition(results: Iterable[TrialResult], entries_by_id: dict[int, DatasetEntry]) -> dict[str, Any]:
    rs = list(results)
    if not rs:
        return {}

    # Setup turn is recorded for token accounting but is excluded from recall
    # (no output.json) and from per-query means. Query turns are scored against qrels.
    query_results = [r for r in rs if not r.is_setup]
    setup_results = [r for r in rs if r.is_setup]

    # Overall (micro-averaged) recall: hits and GT-page counts pooled across queries.
    metrics: dict[str, Any] = dict(overall_recall(query_results, entries_by_id))
    # Per-query means (over the 5 query turns only).
    if query_results:
        metrics["input_tokens"] = mean(r.input_tokens for r in query_results)
        metrics["output_tokens"] = mean(r.output_tokens for r in query_results)
        metrics["cache_read_input_tokens"] = mean(r.cache_read_input_tokens for r in query_results)
        metrics["cache_creation_input_tokens"] = mean(r.cache_creation_input_tokens for r in query_results)
        metrics["total_cost_usd"] = mean(r.total_cost_usd for r in query_results)
        metrics["duration_ms"] = mean(r.duration_ms for r in query_results)
    # Setup is reported separately so the one-time cost isn't lost.
    if setup_results:
        s = setup_results[0]
        metrics["setup_input_tokens"] = s.input_tokens
        metrics["setup_output_tokens"] = s.output_tokens
        metrics["setup_cache_read_input_tokens"] = s.cache_read_input_tokens
        metrics["setup_cache_creation_input_tokens"] = s.cache_creation_input_tokens
        metrics["setup_cost_usd"] = s.total_cost_usd
        metrics["setup_duration_ms"] = s.duration_ms
        metrics["setup_status"] = s.status
    # Session totals (setup + queries).
    metrics["session_input_tokens"] = sum(r.input_tokens for r in rs)
    metrics["session_output_tokens"] = sum(r.output_tokens for r in rs)
    metrics["session_cache_read_input_tokens"] = sum(r.cache_read_input_tokens for r in rs)
    metrics["session_cache_creation_input_tokens"] = sum(r.cache_creation_input_tokens for r in rs)
    metrics["session_total_cost_usd"] = sum(r.total_cost_usd for r in rs)
    metrics["num_turns"] = rs[-1].num_turns if rs else 0
    metrics["success_rate"] = sum(1 for r in rs if r.status == "ok") / len(rs)
    metrics["retriever_used_rate"] = sum(1 for r in rs if r.retriever_used_ever) / len(rs)
    skill_fired = [r.skill_fired for r in rs if r.skill_fired is not None]
    if skill_fired:
        metrics["skill_fired_rate"] = sum(1 for x in skill_fired if x) / len(skill_fired)
    return {
        "run_name": rs[0].condition,
        "success": all(r.status == "ok" for r in rs),
        "metrics": metrics,
        "tags": [rs[0].condition, f"n_queries={len(query_results)}"],
        "artifact_dir": str(Path("trials") / rs[0].condition),
    }


def write_summary_md(session_dir: Path, condition_rows: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    lines = [
        f"# skill_eval session summary — `{session_dir.name}`",
        "",
        f"- Agent model: `{config.get('agent_model', '?')}`",
        f"- Per-trial budget: ${config.get('per_trial_budget_usd', '?')}",
        f"- Per-trial timeout: {config.get('per_trial_timeout_s', '?')}s",
        "",
        "_Agent-session tokens only. Pipeline-side LLM calls (embeddings, VLM, etc.) are not instrumented._",
        "_Each row reflects one Claude session: turn 1 = setup (in-session ingest), turns 2-6 = the 5 queries._",
        "",
        "## Recall (mean over 5 query turns) and per-query token means",
        "",
        (
            "| condition | success_rate | recall@1 | recall@5 | recall@10 | q_input | q_output "
            "| q_cache_read | q_cache_create | q_cost |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in condition_rows:
        m = row.get("metrics", {})
        lines.append(
            (
                "| {cond} | {sr:.2f} | {r1:.3f} | {r5:.3f} | {r10:.3f} | {ipt:.0f} | {opt:.0f} "
                "| {cr:.0f} | {cc:.0f} | ${cost:.3f} |"
            ).format(
                cond=row.get("run_name", "?"),
                sr=m.get("success_rate", 0.0),
                r1=m.get("recall_1", 0.0),
                r5=m.get("recall_5", 0.0),
                r10=m.get("recall_10", 0.0),
                ipt=m.get("input_tokens", 0.0),
                opt=m.get("output_tokens", 0.0),
                cr=m.get("cache_read_input_tokens", 0.0),
                cc=m.get("cache_creation_input_tokens", 0.0),
                cost=m.get("total_cost_usd", 0.0),
            )
        )
    lines.append("")
    lines.append("## Setup turn (one-time cost per condition)")
    lines.append("")
    lines.append("| condition | status | setup_input | setup_output | setup_cache_read | setup_cost | setup_ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in condition_rows:
        m = row.get("metrics", {})
        lines.append(
            "| {cond} | {st} | {ipt:.0f} | {opt:.0f} | {cr:.0f} | ${cost:.3f} | {ms:.0f} |".format(
                cond=row.get("run_name", "?"),
                st=m.get("setup_status", "?"),
                ipt=m.get("setup_input_tokens", 0),
                opt=m.get("setup_output_tokens", 0),
                cr=m.get("setup_cache_read_input_tokens", 0),
                cost=m.get("setup_cost_usd", 0.0),
                ms=m.get("setup_duration_ms", 0),
            )
        )
    lines.append("")
    lines.append("## Session totals (setup + all 5 queries)")
    lines.append("")
    lines.append(
        "| condition | turns | total_input | total_output | total_cache_read | total_cache_create | total_cost |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in condition_rows:
        m = row.get("metrics", {})
        lines.append(
            "| {cond} | {turns} | {ipt} | {opt} | {cr} | {cc} | ${cost:.3f} |".format(
                cond=row.get("run_name", "?"),
                turns=m.get("num_turns", 0),
                ipt=m.get("session_input_tokens", 0),
                opt=m.get("session_output_tokens", 0),
                cr=m.get("session_cache_read_input_tokens", 0),
                cc=m.get("session_cache_creation_input_tokens", 0),
                cost=m.get("session_total_cost_usd", 0.0),
            )
        )
    lines.append("")
    lines.append("## Diagnostics")
    for row in condition_rows:
        m = row.get("metrics", {})
        extras = [
            f"retriever_used_rate={m.get('retriever_used_rate', 0.0):.2f}",
        ]
        if "skill_fired_rate" in m:
            extras.append(f"skill_fired_rate={m['skill_fired_rate']:.2f}")
        lines.append(f"- **{row['run_name']}**: " + ", ".join(extras))
    out = session_dir / "session_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_summary(
    session_dir: Path,
    results_by_condition: dict[str, list[TrialResult]],
    entries: list[DatasetEntry],
    config: dict[str, Any],
    config_path: str,
) -> tuple[Path, Path]:
    entries_by_id = {e.entry_id: e for e in entries}
    rows = []
    for cond in CONDITIONS:
        results = results_by_condition.get(cond, [])
        if not results:
            continue
        rows.append(aggregate_condition(results, entries_by_id))

    json_path = write_session_summary(
        session_dir=session_dir,
        run_results=rows,
        session_type="skill_eval",
        config_path=config_path,
    )
    md_path = write_summary_md(session_dir, rows, config)
    return json_path, md_path
