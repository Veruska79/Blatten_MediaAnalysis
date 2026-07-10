"""
Evaluation of LLM-generated frames against human-annotated reference frames:
embedding cosine similarity, BERTScore, and an LLM-as-judge score.

Usage from another notebook:
    from llm_output_evaluation import evaluate_frames

    result_df, summary = evaluate_frames(
        df=df2,
        human_col='frames_manual',
        llm_col='frames',
        output_path='../../Blatten/data/evaluation_20260214.csv',
        plot_dir='../../Blatten/data/Figures',
    )
    print(summary)

The LLM-judge step calls the OpenAI API and needs an API key. If
OPENAI_API_KEY isn't already set in the environment, judge_dataframe /
evaluate_frames will prompt for it interactively (via getpass) --
never hardcode a key in this file or in a notebook that imports it.
"""

import getpass
import json
import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Embedding cosine similarity
# ---------------------------------------------------------------------------

def compute_cosine_similarity(
    df: pd.DataFrame,
    human_col: str,
    llm_col: str,
    model_name: str = "BAAI/bge-large-en-v1.5",
    out_col: str = "narrative_cosine",
) -> pd.DataFrame:
    """
    Encode *human_col* and *llm_col* with a sentence-transformer and add their
    row-wise cosine similarity as *out_col*. Returns a copy of *df*.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    human_emb = model.encode(df[human_col].fillna("").tolist(), normalize_embeddings=True)
    llm_emb = model.encode(df[llm_col].fillna("").tolist(), normalize_embeddings=True)

    result = df.copy()
    result[out_col] = (human_emb * llm_emb).sum(axis=1)
    return result


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(
    df: pd.DataFrame,
    human_col: str,
    llm_col: str,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
) -> pd.DataFrame:
    """
    Compute BERTScore precision/recall/F1 of *llm_col* (candidate) against
    *human_col* (reference), for rows where both are non-empty.

    Returns a filtered copy of *df* with 'bertscore_P', 'bertscore_R',
    'bertscore_F1' columns added.
    """
    import evaluate

    mask = (
        df[human_col].fillna("").astype(str).str.strip().ne("") &
        df[llm_col].fillna("").astype(str).str.strip().ne("")
    )
    result = df.loc[mask].copy()

    bertscore = evaluate.load("bertscore")
    scores = bertscore.compute(
        predictions=result[llm_col].fillna("").astype(str).tolist(),
        references=result[human_col].fillna("").astype(str).tolist(),
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=True,
    )

    result["bertscore_P"] = np.array(scores["precision"])
    result["bertscore_R"] = np.array(scores["recall"])
    result["bertscore_F1"] = np.array(scores["f1"])
    return result


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_TOOL = [{
    "type": "function",
    "name": "grade_frames",
    "description": "Grade whether the LLM frame matches the human frame in meaning and framing (do not reward length).",
    "parameters": {
        "type": "object",
        "properties": {
            "overall_score_0_to_10": {"type": "number", "minimum": 0, "maximum": 10},
            "A_core_meaning_0_to_2": {"type": "integer", "minimum": 0, "maximum": 2},
            "B_key_elements_0_to_2": {"type": "integer", "minimum": 0, "maximum": 2},
            "C_contradictions_0_to_2": {"type": "integer", "minimum": 0, "maximum": 2},
            "D_tone_frame_0_to_2": {"type": "integer", "minimum": 0, "maximum": 2},
            "E_irrelevant_additions_0_to_2": {"type": "integer", "minimum": 0, "maximum": 2},
            "missing_points": {"type": "array", "items": {"type": "string"}},
            "extra_or_wrong_points": {"type": "array", "items": {"type": "string"}},
            "one_sentence_verdict": {"type": "string"},
        },
        "required": [
            "overall_score_0_to_10", "A_core_meaning_0_to_2", "B_key_elements_0_to_2",
            "C_contradictions_0_to_2", "D_tone_frame_0_to_2", "E_irrelevant_additions_0_to_2",
            "missing_points", "extra_or_wrong_points", "one_sentence_verdict",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}]

_JUDGE_PROMPT = """
You are an evaluator. Compare two narratives for semantic agreement.
Do NOT reward length. Focus on meaning, key elements, contradictions, and framing.

Rubric (0-2 each; total 0-10):
A) Core meaning match
B) Key elements covered
C) Contradictions (2 = none)
D) Tone/frame match
E) Irrelevant additions (2 = none)

Return your judgment by calling the tool grade_frames.

HUMAN FRAME (reference):
<<<{human_text}>>>

LLM FRAME (candidate):
<<<{llm_text}>>>
""".strip()


def _ensure_openai_api_key() -> None:
    """Prompt for OPENAI_API_KEY via getpass if it isn't already set in the environment."""
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


def judge_pair(client, human_text: str, llm_text: str, model: str = "gpt-5-mini") -> dict:
    """Ask the LLM judge to grade one (human, llm) frame pair. Returns the parsed tool-call arguments."""
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": _JUDGE_PROMPT.format(human_text=human_text, llm_text=llm_text)}],
        tools=JUDGE_TOOL,
        tool_choice={"type": "function", "name": "grade_frames"},
    )
    for item in resp.output:
        if item.type == "function_call" and item.name == "grade_frames":
            return json.loads(item.arguments)
    raise RuntimeError("No grade_frames function call found in response output.")


def judge_dataframe(
    df: pd.DataFrame,
    human_col: str,
    llm_col: str,
    client=None,
    model: str = "gpt-5-mini",
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 20,
    sleep_between_calls: float = 0.05,
) -> pd.DataFrame:
    """
    Run the LLM judge over every row where both *human_col* and *llm_col* are
    non-empty, and join the resulting grades onto a filtered copy of *df*.

    *client* defaults to `OpenAI()`. If OPENAI_API_KEY isn't already set in
    the environment, you'll be prompted for it via getpass. Rows where the
    judge call fails get an 'error' column instead of scores, so one bad row
    doesn't stop the run.
    """
    from openai import OpenAI
    from tqdm import tqdm

    if client is None:
        _ensure_openai_api_key()
        client = OpenAI()

    mask = (
        df[human_col].fillna("").astype(str).str.strip().ne("") &
        df[llm_col].fillna("").astype(str).str.strip().ne("")
    )
    result = df.loc[mask].copy()

    grades = []
    for i, (idx, row) in enumerate(tqdm(result.iterrows(), total=len(result), desc="Judging")):
        try:
            grade = judge_pair(client, str(row[human_col]), str(row[llm_col]), model=model)
            grade["index"] = idx
        except Exception as e:
            grade = {"index": idx, "error": str(e)}
        grades.append(grade)

        if sleep_between_calls:
            time.sleep(sleep_between_calls)

        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            pd.DataFrame(grades).set_index("index").to_parquet(checkpoint_path)

    grades_df = pd.DataFrame(grades).set_index("index")
    return result.join(grades_df)


# ---------------------------------------------------------------------------
# Summary and plots
# ---------------------------------------------------------------------------

def summarize_scores(df: pd.DataFrame, score_col: str = "overall_score_0_to_10", threshold: float = 6) -> dict:
    """Return count/mean/median/std and the percentage of rows at or above *threshold*."""
    scores = df[score_col].dropna()
    return {
        "n": len(scores),
        "mean": scores.mean(),
        "median": scores.median(),
        "std": scores.std(),
        "pct_at_or_above_threshold": (scores >= threshold).mean() * 100,
    }


def plot_score_distribution(
    df: pd.DataFrame,
    score_col: str = "overall_score_0_to_10",
    bins: int = 10,
    value_range: tuple = (0, 10),
    xlabel: str = "Overall score",
    save_path: Optional[str] = None,
):
    """Histogram + KDE of *score_col*, optionally saved to *save_path*."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    s = df[score_col].dropna().to_numpy()
    kde = gaussian_kde(s)
    xs = np.linspace(*value_range, 400)

    fig, ax = plt.subplots()
    ax.hist(s, bins=bins, range=value_range, density=True, alpha=0.4, linewidth=0.5, edgecolor="white")
    ax.plot(xs, kde(xs), linewidth=2)
    ax.set_xlim(*value_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_score_comparison(
    series_dict: dict,
    bins: int = 10,
    value_range: tuple = (0, 10),
    xlabel: str = "Value",
    save_path: Optional[str] = None,
):
    """
    Overlay density histograms for multiple named score series, e.g.
    {'Cosine similarity': df['frames_cosine'] * 10, 'LLM Judge': df['overall_score_0_to_10']}.
    """
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(*value_range, bins + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, series in series_dict.items():
        ax.hist(pd.Series(series).dropna().to_numpy(), bins=bin_edges, density=True,
                alpha=0.45, linewidth=0.5, edgecolor="white", label=label)
    ax.set_xlim(*value_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def evaluate_frames(
    df: pd.DataFrame,
    human_col: str = "frames_manual",
    llm_col: str = "frames",
    run_cosine: bool = True,
    run_bertscore: bool = True,
    run_judge: bool = True,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    judge_model: str = "gpt-5-mini",
    judge_client=None,
    judge_checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
    plot_dir: Optional[str] = None,
) -> tuple:
    """
    Run cosine similarity, BERTScore, and LLM-judge evaluation of *llm_col*
    against *human_col*; optionally save the result CSV and summary plots.

    Returns
    -------
    result  : DataFrame with narrative_cosine / bertscore_* / judge columns added
    summary : dict from summarize_scores(), or {} if run_judge is False
    """
    result = df.copy()

    if run_cosine:
        result = compute_cosine_similarity(result, human_col, llm_col, model_name=embedding_model)

    if run_bertscore:
        result = compute_bertscore(result, human_col, llm_col, model_type=bertscore_model)

    if run_judge:
        result = judge_dataframe(
            result, human_col, llm_col,
            client=judge_client, model=judge_model,
            checkpoint_path=judge_checkpoint_path,
        )

    summary = summarize_scores(result) if run_judge else {}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    if plot_dir and run_judge:
        os.makedirs(plot_dir, exist_ok=True)
        plot_score_distribution(
            result, save_path=os.path.join(plot_dir, "hist_kde_overall_score_llm_judge_frames.png")
        )
        if run_cosine:
            plot_score_comparison(
                {"Cosine similarity": result["frames_cosine"] * 10, "LLM Judge": result["overall_score_0_to_10"]},
                save_path=os.path.join(plot_dir, "two_histograms_density_frames.png"),
            )

    print(summary)
    return result, summary


# -------------------------------------------------------------------------------------------
# CLI convenience to run from terminal if needed: python llm_output_evaluation.py data.csv out.csv
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python llm_output_evaluation.py <input_csv> [output_csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    df_in = pd.read_csv(input_csv)
    result_df, summary_stats = evaluate_frames(df_in, output_path=output_csv)
