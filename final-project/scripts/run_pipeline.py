#!/usr/bin/env python
"""End-to-end pipeline: raw data → clean → features → models → outputs.

Run from the final-project/ directory:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-llm      # skip Claude API calls
    python scripts/run_pipeline.py --skip-embed    # skip embedding computation

The pipeline is idempotent: if intermediate parquet files already exist,
those steps are skipped (use --force to re-run).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.ingest import (
    load_fomc_metadata,
    load_intraday_bars,
    load_policy_rates,
    load_transcripts,
    load_transcripts_json,
    save_transcripts_json,
)
from src.clean import qa_intraday_bars, read_parquet, write_parquet
from src.targets import compute_targets, windows_from_config
from src.features_structured import build_structured_features
from src.features_text import score_transcripts_keywords, score_transcripts_llm, make_embeddings
from src.eval import assemble_panel, run_ladder
from src.plots import ladder_chart, pnl_chart, save_fig

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _exists(path: pathlib.Path, force: bool) -> bool:
    """Return True if the path exists and we should skip recomputation."""
    if path.exists() and not force:
        logger.info("Skip (exists): %s", path.name)
        return True
    return False


# ── Pipeline steps ────────────────────────────────────────────────────────────


def step_ingest(cfg, data_raw, data_clean, force):
    """Phase 1: Load and clean raw data."""
    logger.info("=== Phase 1: Data Ingestion ===")

    # FOMC metadata
    meta_path = data_clean / "fomc_metadata.parquet"
    if not _exists(meta_path, force):
        fomc = load_fomc_metadata(data_raw)
        write_parquet(fomc, meta_path)
    else:
        fomc = read_parquet(meta_path)

    meeting_ids = fomc["meeting_id"].tolist()

    # Policy rates
    rates_path = data_clean / "policy_rates.parquet"
    if not _exists(rates_path, force):
        policy = load_policy_rates(data_raw)
        write_parquet(policy.reset_index(), rates_path)

    # Intraday bars
    bars_path = data_clean / "intraday_bars.parquet"
    if not _exists(bars_path, force):
        bars_raw = load_intraday_bars(data_raw, meeting_ids)
        bars_clean, qa_report = qa_intraday_bars(bars_raw)
        logger.info("QA: %d raw → %d clean", qa_report["n_raw"], qa_report["n_clean"])
        write_parquet(bars_clean, bars_path)

    # Transcripts
    transcript_path = data_clean / "transcripts.json"
    if not _exists(transcript_path, force):
        logger.info("Extracting transcripts from PDFs (slow) ...")
        transcripts = load_transcripts(data_raw, meeting_ids)
        save_transcripts_json(transcripts, transcript_path)

    return fomc, meeting_ids


def step_targets(cfg, data_clean, fomc, force):
    """Phase 2: Compute log-return targets."""
    logger.info("=== Phase 2: Target Construction ===")
    targets_path = data_clean / "targets.parquet"
    if _exists(targets_path, force):
        return read_parquet(targets_path)

    bars = read_parquet(data_clean / "intraday_bars.parquet")
    windows = windows_from_config(cfg)
    panel = compute_targets(bars, fomc, cfg.pairs, windows)
    write_parquet(panel, targets_path)
    return panel


def step_features_structured(cfg, data_clean, fomc, force):
    """Phase 3: Build structured features (rungs 1–3)."""
    logger.info("=== Phase 3: Structured Features ===")
    feat_path = data_clean / "features_structured.parquet"
    if _exists(feat_path, force):
        return read_parquet(feat_path)

    bars = read_parquet(data_clean / "intraday_bars.parquet")
    import pandas as pd
    policy = read_parquet(data_clean / "policy_rates.parquet")
    policy = policy.set_index("date") if "date" in policy.columns else policy
    policy.index = pd.to_datetime(policy.index)

    feat = build_structured_features(
        fomc_meta=fomc,
        policy_rates=policy,
        bars=bars,
        pair_cb_map=cfg.pair_cb_map,
        pre_start=cfg.pre_window.start,
        pre_end=cfg.pre_window.end,
    )
    write_parquet(feat, feat_path)
    return feat


def step_features_text(cfg, data_clean, meeting_ids, outputs, skip_llm, skip_embed, force):
    """Phase 4: Build text features (keyword + LLM + embeddings)."""
    logger.info("=== Phase 4: Text Features ===")
    feat_path = data_clean / "features_text.parquet"
    if _exists(feat_path, force):
        return read_parquet(feat_path)

    transcripts = load_transcripts_json(data_clean / "transcripts.json")

    # Stage 4a: keywords (always run)
    kw = score_transcripts_keywords(transcripts, meeting_ids)

    # Stage 4b: LLM
    import pandas as pd
    llm_cache = data_clean / "llm_scores_cache.parquet"
    if _exists(llm_cache, force) or skip_llm:
        if llm_cache.exists():
            llm = read_parquet(llm_cache)
        else:
            logger.warning("LLM cache not found and --skip-llm set; text features will lack rubric scores")
            llm = pd.DataFrame(columns=["meeting_id", "text_source"])
    else:
        prompt_path = outputs / "prompts" / f"rubric_{cfg.text.prompt_version}.txt"
        llm = score_transcripts_llm(
            transcripts, meeting_ids, prompt_path, cfg.text.llm_model, log_dir=outputs
        )
        write_parquet(llm, llm_cache)

    # Stage 4c: embeddings
    emb_cache = data_clean / "embeddings_cache.parquet"
    if _exists(emb_cache, force) or skip_embed:
        if emb_cache.exists():
            emb = read_parquet(emb_cache)
        else:
            logger.warning("Embeddings cache not found and --skip-embed set; skipping")
            emb = pd.DataFrame(columns=["meeting_id", "text_source"])
    else:
        figures_dir = outputs / "figures"
        emb = make_embeddings(
            transcripts, meeting_ids,
            n_components_min=cfg.text.embedding_dim_min,
            n_components_max=cfg.text.embedding_dim_max,
            figures_dir=figures_dir,
        )
        write_parquet(emb, emb_cache)

    # Merge text features
    feat = kw.merge(llm, on=["meeting_id", "text_source"], how="outer")
    feat = feat.merge(emb, on=["meeting_id", "text_source"], how="outer")
    write_parquet(feat, feat_path)
    return feat


def step_model(cfg, data_clean, outputs, force):
    """Phase 5: Run the full model ladder."""
    logger.info("=== Phase 5: Model Ladder ===")
    results_path = outputs / "model_results.csv"
    if _exists(results_path, force):
        import pandas as pd
        return pd.read_csv(results_path)

    targets   = read_parquet(data_clean / "targets.parquet")
    feat_struct = read_parquet(data_clean / "features_structured.parquet")
    feat_text_path = data_clean / "features_text.parquet"
    feat_text = read_parquet(feat_text_path) if feat_text_path.exists() else None

    panel = assemble_panel(targets, feat_struct, feat_text)
    write_parquet(panel, data_clean / "panel_final.parquet")

    results = run_ladder(panel, cfg)
    results.to_csv(results_path, index=False)
    logger.info("Results saved → %s", results_path)
    return results


def step_plots(cfg, data_clean, outputs, results):
    """Phase 6: Generate and save all output figures."""
    logger.info("=== Phase 6: Plots ===")
    figures_dir = outputs / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rung_order = [r for r in [
        "Rung 1 – Predict Zero", "Rung 1 – Hist Mean",
        "Rung 2 – OLS Theory",
        "Rung 3a – Ridge", "Rung 3a – LASSO", "Rung 3a – ElasticNet",
        "Rung 3a – Huber", "Rung 3b – GBR",
        "Rung 4 – +Keywords", "Rung 4 – +LLM Rubric", "Rung 4 – +Embeddings",
    ] if r in results["rung"].values]

    for window in ["statement", "digestion"]:
        for metric in ["rmse", "dir_acc"]:
            fig = ladder_chart(results, metric=metric, window=window, rung_order=rung_order)
            save_fig(fig, figures_dir / f"ladder_{metric}_{window}.png")

    logger.info("Figures saved to %s", figures_dir)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="FX @ FOMC end-to-end pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Claude API calls (use cached if available)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding computation (use cached if available)")
    parser.add_argument("--force", action="store_true", help="Recompute all steps even if outputs exist")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config(PROJECT_ROOT / args.config)

    data_raw   = PROJECT_ROOT / cfg.paths.data_raw
    data_clean = PROJECT_ROOT / cfg.paths.data_clean
    outputs    = PROJECT_ROOT / cfg.paths.outputs

    data_clean.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    logger.info("Pipeline start | root=%s", PROJECT_ROOT)

    fomc, meeting_ids = step_ingest(cfg, data_raw, data_clean, args.force)
    step_targets(cfg, data_clean, fomc, args.force)
    step_features_structured(cfg, data_clean, fomc, args.force)
    step_features_text(cfg, data_clean, meeting_ids, outputs, args.skip_llm, args.skip_embed, args.force)
    results = step_model(cfg, data_clean, outputs, args.force)
    step_plots(cfg, data_clean, outputs, results)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
