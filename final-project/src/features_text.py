"""Text feature engineering — Rung 4 of the model ladder.

Three stages of increasing sophistication:

Stage 4a — Keyword/rule-based scoring (deterministic, no API cost)
    Hawkish/dovish word counts, balance sheet terms, uncertainty phrases.

Stage 4b — LLM rubric scoring via Claude API
    Structured 7-dimension rubric scored by claude-opus-4-6.
    Requires ANTHROPIC_API_KEY env variable.
    Prompt template: outputs/prompts/rubric_v1.txt (version-controlled).

Stage 4c — Text embeddings + PCA
    Sentence-level embeddings via sentence-transformers.
    PCA reduction to n_components in [embedding_dim_min, embedding_dim_max].
    Separate embeddings for statement and press_conf text.

Output: a single DataFrame with one row per (meeting_id, text_source)
where text_source ∈ {"statement", "press_conf"}.
The modeling layer joins this to the target panel on (meeting_id, window):
    statement window → statement text features
    digestion window → press_conf text features
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Stage 4a: Keyword scoring ─────────────────────────────────────────────────

# Hawkish words/phrases (suggest tighter policy, less accommodation)
_HAWKISH_TERMS = [
    "inflation", "inflationary", "price stability", "above target",
    "overshoot", "tighten", "tightening", "restrictive", "elevated",
    "strong labor", "robust", "accelerat", "overheat", "wage growth",
    "hike", "increase the target",
]

# Dovish words/phrases (suggest looser policy, more accommodation)
_DOVISH_TERMS = [
    "below target", "low inflation", "disinflation", "deflat",
    "slowdown", "weakness", "accommodat", "easing", "cut", "reduce",
    "support", "cautious", "gradual", "patient", "data-dependent",
    "global headwinds", "downside risk", "recession",
]

# Balance sheet terms
_BALANCE_SHEET_TERMS = [
    "balance sheet", "asset purchase", "quantitative", "taper",
    "tapering", " QT", " QE", "reinvestment",
]

# Uncertainty phrases
_UNCERTAINTY_TERMS = [
    "uncertain", "uncertainty", "monitor", "assess", "evolv",
    "data-dependent", "remain attentive", "remain vigilant",
    "as appropriate", "if warranted",
]


def _count_terms(text: str, terms: list[str]) -> int:
    """Case-insensitive count of term occurrences in text."""
    text_lower = text.lower()
    return sum(text_lower.count(term.lower()) for term in terms)


def score_transcripts_keywords(
    transcripts: dict[str, dict[str, str]],
    meeting_ids: list[str],
) -> pd.DataFrame:
    """Stage 4a: Compute keyword-based text scores.

    Returns DataFrame with columns:
        meeting_id, text_source,
        hawkish_count, dovish_count, net_hawkish,
        net_hawkish_norm  (normalised by total word count),
        balance_sheet_kw  (int: count of balance sheet terms),
        uncertainty_kw    (int: count of uncertainty terms)
    """
    rows: list[dict] = []

    for mid in meeting_ids:
        if mid not in transcripts:
            continue
        for text_source in ("statement", "press_conf"):
            text = transcripts[mid].get(text_source, "")
            if not text:
                rows.append(
                    {
                        "meeting_id": mid,
                        "text_source": text_source,
                        **{k: np.nan for k in [
                            "hawkish_count", "dovish_count", "net_hawkish",
                            "net_hawkish_norm", "balance_sheet_kw", "uncertainty_kw",
                        ]},
                    }
                )
                continue

            word_count = len(text.split())
            h = _count_terms(text, _HAWKISH_TERMS)
            d = _count_terms(text, _DOVISH_TERMS)
            bs = _count_terms(text, _BALANCE_SHEET_TERMS)
            unc = _count_terms(text, _UNCERTAINTY_TERMS)

            rows.append(
                {
                    "meeting_id": mid,
                    "text_source": text_source,
                    "hawkish_count": h,
                    "dovish_count": d,
                    "net_hawkish": h - d,
                    "net_hawkish_norm": (h - d) / max(word_count, 1),
                    "balance_sheet_kw": bs,
                    "uncertainty_kw": unc,
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Keyword scores: %d rows", len(df))
    return df


# ── Stage 4b: LLM rubric scoring ─────────────────────────────────────────────

_RUBRIC_KEYS = [
    "hawkish_dovish",
    "inflation_focus",
    "labor_focus",
    "recession_risk",
    "uncertainty_score",
    "forward_guidance_strength",
    "balance_sheet_mention",
]

_VALID_RANGES = {
    "hawkish_dovish": (-2, 2),
    "inflation_focus": (-2, 2),
    "labor_focus": (-2, 2),
    "recession_risk": (-2, 2),
    "uncertainty_score": (0, 2),
    "forward_guidance_strength": (0, 2),
    "balance_sheet_mention": (0, 1),
}


def _load_prompt_template(prompt_path: pathlib.Path) -> str:
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


def _call_claude(client, model: str, prompt: str, max_retries: int = 3) -> str | None:
    """Call Claude API and return the text response, with retry on rate-limit."""
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "overload" in err_str:
                wait = 2 ** attempt * 5
                logger.warning("Rate limit hit; waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.error("Claude API error: %s", e)
                return None
    return None


def _parse_llm_response(response: str) -> dict[str, float] | None:
    """Extract and validate the JSON rubric from Claude's response."""
    # Extract JSON object from the response (handles any surrounding text)
    match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
    if not match:
        logger.warning("No JSON found in LLM response: %s", response[:200])
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s — raw: %s", e, match.group()[:200])
        return None

    # Validate all keys present and in range
    result = {}
    for key in _RUBRIC_KEYS:
        val = parsed.get(key)
        if val is None:
            logger.warning("Missing rubric key '%s' in response", key)
            return None
        lo, hi = _VALID_RANGES[key]
        if not (lo <= float(val) <= hi):
            logger.warning("Out-of-range value for '%s': %s", key, val)
            val = max(lo, min(hi, float(val)))  # clip rather than discard
        result[key] = float(val)

    return result


def score_transcripts_llm(
    transcripts: dict[str, dict[str, str]],
    meeting_ids: list[str],
    prompt_path: pathlib.Path,
    llm_model: str = "claude-opus-4-6",
    log_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Stage 4b: Score transcripts via Claude API rubric.

    Requires ANTHROPIC_API_KEY environment variable.

    Returns DataFrame with columns:
        meeting_id, text_source, llm_model_id,
        hawkish_dovish, inflation_focus, labor_focus, recession_risk,
        uncertainty_score, forward_guidance_strength, balance_sheet_mention
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("anthropic package required: pip install anthropic") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    prompt_template = _load_prompt_template(prompt_path)

    # Log model version and prompt path for auditability
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "llm_run_metadata.json", "w") as f:
            json.dump(
                {
                    "llm_model": llm_model,
                    "prompt_path": str(prompt_path),
                    "prompt_version": prompt_path.stem,
                    "meeting_ids": meeting_ids,
                },
                f, indent=2,
            )

    rows: list[dict] = []
    for i, mid in enumerate(meeting_ids):
        if mid not in transcripts:
            continue
        for text_source in ("statement", "press_conf"):
            text = transcripts[mid].get(text_source, "")
            if not text or len(text.strip()) < 50:
                rows.append(
                    {
                        "meeting_id": mid,
                        "text_source": text_source,
                        "llm_model_id": llm_model,
                        **{k: np.nan for k in _RUBRIC_KEYS},
                    }
                )
                continue

            prompt = prompt_template.replace("{TEXT}", text[:8000])  # truncate to ~8k chars
            response = _call_claude(client, llm_model, prompt)

            if response is None:
                scores = {k: np.nan for k in _RUBRIC_KEYS}
            else:
                parsed = _parse_llm_response(response)
                scores = parsed if parsed is not None else {k: np.nan for k in _RUBRIC_KEYS}

            rows.append(
                {
                    "meeting_id": mid,
                    "text_source": text_source,
                    "llm_model_id": llm_model,
                    **scores,
                }
            )

            logger.info("[%d/%d] %s / %s → %s", i + 1, len(meeting_ids), mid, text_source, scores)
            time.sleep(0.5)  # polite rate limiting

    df = pd.DataFrame(rows)
    logger.info("LLM scores: %d rows", len(df))
    return df


# ── Stage 4c: Embeddings + PCA ────────────────────────────────────────────────


def make_embeddings(
    transcripts: dict[str, dict[str, str]],
    meeting_ids: list[str],
    n_components_min: int = 5,
    n_components_max: int = 20,
    model_name: str = "all-MiniLM-L6-v2",
    figures_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Stage 4c: Compute sentence embeddings and reduce via PCA.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default).
    Separately embeds statement and press_conf text.
    PCA n_components chosen by explained-variance elbow in [min, max] range.

    Returns DataFrame with columns:
        meeting_id, text_source, emb_pc_1, emb_pc_2, ..., emb_pc_N
    """
    # Use transformers directly to avoid sentence-transformers' dependency on
    # `datasets` (which requires _lzma, a C extension often missing in pyenv builds).
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers and torch required: pip install transformers torch"
        ) from e

    hf_model_name = f"sentence-transformers/{model_name}"
    logger.info("Loading embedding model: %s", hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    embed_model = AutoModel.from_pretrained(hf_model_name)
    embed_model.eval()

    def _mean_pool(model_output, attention_mask):
        token_emb = model_output.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

    def _encode_texts(texts: list[str], batch_size: int = 16) -> "np.ndarray":
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
            with torch.no_grad():
                output = embed_model(**encoded)
            emb = _mean_pool(output, encoded["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.numpy())
        return np.vstack(all_emb)

    all_rows: list[dict] = []

    for text_source in ("statement", "press_conf"):
        texts_in_order: list[str] = []
        ids_in_order: list[str] = []

        for mid in meeting_ids:
            text = transcripts.get(mid, {}).get(text_source, "")
            texts_in_order.append(text if text else "")
            ids_in_order.append(mid)

        logger.info("Encoding %d %s texts ...", len(texts_in_order), text_source)
        embeddings = _encode_texts(texts_in_order)
        # embeddings shape: (n_meetings, embedding_dim)

        # Standardise before PCA
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(embeddings)

        # Choose n_components by explained variance elbow
        pca_full = PCA(n_components=min(n_components_max, emb_scaled.shape[0], emb_scaled.shape[1]))
        pca_full.fit(emb_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)

        # Select n_components: smallest k >= n_components_min that explains >= 80%,
        # capped at n_components_max
        n_comp = n_components_min
        for k in range(n_components_min, n_components_max + 1):
            if k - 1 < len(cum_var) and cum_var[k - 1] >= 0.80:
                n_comp = k
                break
        else:
            n_comp = min(n_components_max, len(cum_var))

        logger.info("%s: selected n_components=%d (%.1f%% variance explained)", text_source, n_comp, cum_var[n_comp - 1] * 100)

        if figures_dir:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(1, len(cum_var) + 1), pca_full.explained_variance_ratio_ * 100, alpha=0.6, label="Incremental")
            ax.plot(range(1, len(cum_var) + 1), cum_var * 100, "r-o", ms=4, label="Cumulative")
            ax.axvline(n_comp, color="green", lw=1.5, ls="--", label=f"Selected k={n_comp}")
            ax.axhline(80, color="gray", lw=0.8, ls=":")
            ax.set_title(f"PCA Explained Variance — {text_source}")
            ax.set_xlabel("Component")
            ax.set_ylabel("Variance explained (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(figures_dir / f"pca_variance_{text_source}.png", dpi=120)
            plt.close(fig)
            logger.info("Saved PCA plot → %s", figures_dir / f"pca_variance_{text_source}.png")

        # Final PCA with selected n_components
        pca = PCA(n_components=n_comp)
        components = pca.fit_transform(emb_scaled)

        for mid, comp_row in zip(ids_in_order, components):
            row = {"meeting_id": mid, "text_source": text_source}
            for j, val in enumerate(comp_row):
                row[f"emb_pc_{j+1}"] = float(val)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    logger.info("Embeddings DataFrame: %d rows, %d columns", len(df), df.shape[1])
    return df


# ── Combine all text features ─────────────────────────────────────────────────


def build_text_features(
    transcripts: dict[str, dict[str, str]],
    meeting_ids: list[str],
    prompt_path: pathlib.Path,
    llm_model: str = "claude-opus-4-6",
    n_components_min: int = 5,
    n_components_max: int = 20,
    log_dir: pathlib.Path | None = None,
    figures_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Run all three text feature stages and merge into a single DataFrame.

    Returns a DataFrame with one row per (meeting_id, text_source).
    Join to the target panel:
        statement window ← text_source == 'statement'
        digestion window ← text_source == 'press_conf'
    """
    logger.info("=== Stage 4a: Keyword scoring ===")
    kw = score_transcripts_keywords(transcripts, meeting_ids)

    logger.info("=== Stage 4b: LLM rubric scoring ===")
    llm = score_transcripts_llm(transcripts, meeting_ids, prompt_path, llm_model, log_dir)

    logger.info("=== Stage 4c: Embeddings + PCA ===")
    emb = make_embeddings(
        transcripts, meeting_ids, n_components_min, n_components_max, figures_dir=figures_dir
    )

    # Merge all three on (meeting_id, text_source)
    df = kw.merge(llm, on=["meeting_id", "text_source"], how="outer")
    df = df.merge(emb, on=["meeting_id", "text_source"], how="outer")

    logger.info("Text features: %d rows, %d columns", len(df), df.shape[1])
    return df
