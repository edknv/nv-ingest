# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`retriever skill-eval run` benchmark."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import typer

from nemo_retriever.harness.artifacts import create_session_dir
from nemo_retriever.harness.config import REPO_ROOT
from nemo_retriever.skill_eval.dataset import load_config, load_dataset
from nemo_retriever.skill_eval.report import overall_recall, write_summary
from nemo_retriever.skill_eval.runner import (
    CONDITIONS,
    cleanup_condition_workdir,
    run_condition,
    save_trial,
)

DEFAULT_ORDER = ("c1_base", "c2_retriever", "c3_retriever_skill")

app = typer.Typer(help="Benchmark Claude with vs. without the /nemo-retriever skill on a folder of PDFs.")
logger = logging.getLogger(__name__)


@app.command("run")
def run_command(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to run.yaml; defaults to the packaged config."),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", help="Path to dataset.yaml; defaults to the packaged 5-entry dataset."
    ),
    conditions: str = typer.Option(
        ",".join(DEFAULT_ORDER),
        "--conditions",
        help=(
            "Comma-separated conditions in execution order. Each condition's workdir is deleted after it runs, "
            "so only one LanceDB is on disk at a time."
        ),
    ),
    artifacts_root: Optional[Path] = typer.Option(
        None, "--artifacts-root", help="Override the artifact root; defaults to <repo>/nemo_retriever/artifacts/"
    ),
) -> None:
    """Run the v1 benchmark: 5 entries × selected conditions, sequential."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if shutil.which("claude") is None:
        typer.echo("Error: `claude` CLI is not on PATH; install Claude Code first.", err=True)
        raise typer.Exit(code=2)

    cfg = load_config(config)
    selected = [c.strip() for c in conditions.split(",") if c.strip()]
    for c in selected:
        if c not in CONDITIONS:
            typer.echo(f"Error: unknown condition '{c}'. Choose from {CONDITIONS}.", err=True)
            raise typer.Exit(code=2)

    entries = load_dataset(dataset)
    typer.echo(f"Loaded {len(entries)} dataset entries.")

    if not cfg.get("pdf_dir"):
        typer.echo("Error: config is missing required key 'pdf_dir'.", err=True)
        raise typer.Exit(code=2)
    pdf_source = Path(str(cfg["pdf_dir"])).expanduser().resolve()
    skill_source = Path(
        str(cfg.get("skill_source_dir") or REPO_ROOT / ".claude" / "skills" / "nemo-retriever")
    ).expanduser()
    workdir_root = Path(str(cfg.get("per_trial_workdir_root", "/tmp/skill_eval"))).expanduser()
    workdir_root.mkdir(parents=True, exist_ok=True)
    model = str(cfg.get("agent_model", "claude-opus-4-7"))
    budget = float(cfg.get("per_trial_budget_usd", 5.0))
    timeout = int(cfg.get("per_trial_timeout_s", 600))

    base_dir = str(artifacts_root) if artifacts_root else None
    session_dir = create_session_dir("skilleval", base_dir=base_dir)
    typer.echo(f"Session dir: {session_dir}")

    (session_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in cfg.items()) + "\n", encoding="utf-8")

    results_by_condition: dict[str, list] = {c: [] for c in selected}
    for cond in selected:
        typer.echo(f"Starting session for {cond} — setup + {len(entries)} query turns")
        workdir, results = run_condition(
            condition=cond,
            entries=entries,
            workdir_root=workdir_root,
            pdf_source=pdf_source,
            skill_source=skill_source,
            model=model,
            budget_usd=budget,
            timeout_s=timeout,
        )
        for r in results:
            save_trial(r, session_dir)
            kind = "setup" if r.is_setup else f"entry_id={r.entry_id} query_id={r.query_id}"
            typer.echo(
                f"  turn {r.num_turns} {kind}: status={r.status} "
                f"tokens(in/out/cache_r)={r.input_tokens}/{r.output_tokens}/{r.cache_read_input_tokens} "
                f"cost=${r.total_cost_usd:.3f} retrieved={len(r.ranked_retrieved)}"
            )
            results_by_condition[cond].append(r)

        entries_by_id = {e.entry_id: e for e in entries}
        scores = overall_recall(results, entries_by_id)
        typer.echo(
            f"\nOverall recall for {cond}: "
            f"recall@1={scores['recall_1']:.3f}  "
            f"recall@5={scores['recall_5']:.3f}  "
            f"recall@10={scores['recall_10']:.3f}"
        )

        cleanup_condition_workdir(workdir)
        typer.echo(f"Cleaned up workdir for {cond}\n")

    json_path, md_path = write_summary(
        session_dir=session_dir,
        results_by_condition=results_by_condition,
        entries=entries,
        config=cfg,
        config_path=str(config) if config else "<packaged default>",
    )
    typer.echo(f"\nWrote {json_path}")
    typer.echo(f"Wrote {md_path}")
    typer.echo("\nDone.")
