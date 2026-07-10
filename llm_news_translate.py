"""
LLM-based news translation pipeline (Ollama Cloud).

Translates a text column in a CSV from German / French / Italian to English
using an Ollama Cloud model, with checkpointing and parallel processing.

Usage from another notebook
---------------------------
    from llm_news_translate import translate_csv

    translate_csv(
        input_csv="data/news.csv",
        output_csv="data/news_translated.csv",
        checkpoint_pkl="data/news_translated.checkpoint.pkl",
        text_col="content",
        lang_col="language",
        out_col="content_en",
    )

Command-line usage
------------------
    python llm_news_translate.py \\
        --input data/news.csv \\
        --output data/news_translated.csv \\
        --text-col content \\
        --lang-col language \\
        --out-col content_en

    # Pass API key via env var (preferred):
    export OLLAMA_API_KEY=<your-key>

    # Or pass inline:
    python llm_news_translate.py --input ... --api-key <your-key>
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import pandas as pd
from ollama import Client
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Defaults (all overridable at call time)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-oss:120b-cloud"
MAX_CHARS_PER_CHUNK = 4096
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
DEFAULT_MAX_WORKERS = 6
DEFAULT_BATCH_SIZE = 40
DEFAULT_MAX_RETRY_ATTEMPTS = 2
DEFAULT_CHECKPOINT_FREQUENCY = 40

EMPTY_MARKERS = {
    "__EMPTY_TRANSLATION__",
    "--EMPTY TRANSLATION--",
    "--EMPTY TRANSLATION--,",
}


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def get_ollama_client(api_key: Optional[str] = None) -> Client:
    key = api_key or os.environ.get("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError(
            "Ollama API key not found. Set OLLAMA_API_KEY or pass --api-key."
        )
    return Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + key},
    )


# ---------------------------------------------------------------------------
# Core translation helpers
# ---------------------------------------------------------------------------

def chat_with_retries(
    client: Client,
    messages: List[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    backoff = INITIAL_BACKOFF_SECONDS
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat(model=model, messages=messages, options={"temperature": 0})
        except Exception as e:
            last_exc = e
            if attempt == MAX_RETRIES:
                break
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"Ollama chat failed after {MAX_RETRIES} attempts") from last_exc


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if current_len + len(p) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


_SYSTEM_PROMPT = (
    "You are a professional translator specialized in translating German, French, and Italian "
    "news articles to British English.\n"
    "Your job is to translate the all text (including dialogues or people quotes) into clear and accurate English.\n"
    "\n"
    "STRICT RULES:\n"
    "- Preserve *all punctuation exactly as it appears* in the source text.\n"
    "  This includes: « … », » … «, „ … ", —, –, …, ;, :, ?, !, %, CHF, etc.\n"
    "- Do NOT change quotation marks to another style.\n"
    "- Do NOT escape punctuation. No backslashes (\\') or (\\\").\n"
    "- Do NOT add or remove quotation marks.\n"
    "- Do NOT convert quotes to English format unless they already appear that way.\n"
    "- Do NOT introduce Markdown formatting (**, *, ``, etc.).\n"
    "- Translate ONLY the words — punctuation must remain identical.\n"
    "- Output pure plain text only.\n"
    "\n"
    "ADDITIONAL RULES:\n"
    "- Keep all numbers, dates, names, organisations, and places unchanged.\n"
    "- Do NOT summarise or omit content.\n"
    "- Do NOT add explanations.\n"
)


def translate_chunk(
    client: Client,
    text: str,
    src_lang: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Translate a single chunk (no splitting)."""
    user_prompt = (
        f"Source language code: {src_lang}\n\n"
        "Translate the following news text into English:\n\n"
        f"{text}"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = chat_with_retries(client, messages, model=model)
    return response.get("message", {}).get("content", "").strip()


def translate_text(
    client: Client,
    text: str,
    src_lang: Optional[str],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Translate *text* from *src_lang* into English.

    - If src_lang is 'en' or not in {de, fr, it}: returns text unchanged.
    - Splits long texts into paragraph chunks and re-joins them.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if not isinstance(src_lang, str):
        return text
    src_lang = src_lang.strip().lower()
    if src_lang == "en":
        return text
    if src_lang not in {"de", "fr", "it"}:
        return text

    chunks = chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK)
    translated = [
        translate_chunk(client, chunk, src_lang=src_lang, model=model)
        for chunk in chunks
    ]
    return "\n\n".join(t.strip() for t in translated if t.strip()) or text


def clean_translation(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks."""
    if not isinstance(text, str):
        return text
    text = text.replace(" ", " ").replace(" ", " ")
    text = text.replace("\\n", "\n")
    placeholder = "<<<PARA_BREAK>>>"
    text = text.replace("\n\n", placeholder)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(placeholder, "\n\n")
    return text.strip()


# ---------------------------------------------------------------------------
# Batch / checkpoint processing
# ---------------------------------------------------------------------------

def _needs_translation(
    row: pd.Series,
    text_col: str,
    out_col: str,
    retry_count_col: str,
    retry_failed: bool,
    max_retries: int,
) -> bool:
    src = row[text_col]
    if pd.isna(src) or str(src).strip() == "":
        return False
    retry = row.get(retry_count_col, 0)
    retry = 0 if pd.isna(retry) else int(retry)
    if retry >= max_retries:
        return False
    tgt = row[out_col]
    if pd.isna(tgt) or str(tgt).strip() == "":
        return True
    if retry_failed and str(tgt).strip() in EMPTY_MARKERS:
        return True
    return False


def _translate_row_worker(
    client: Client,
    idx: int,
    text: str,
    src_lang: Optional[str],
    current_retry: int,
    model: str,
) -> Tuple[int, object, int]:
    try:
        translated = translate_text(client, text, src_lang, model=model)
        if not translated or str(translated).strip() == "":
            return idx, pd.NA, current_retry + 1
        return idx, str(translated), 0
    except Exception as e:
        print(f"  Row {idx} error: {e}")
        return idx, pd.NA, current_retry + 1


def _translate_batch(
    client: Client,
    df_batch: pd.DataFrame,
    text_col: str,
    lang_col: str,
    retry_count_col: str,
    model: str,
    max_workers: int,
) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in df_batch.iterrows():
            retry = row.get(retry_count_col, 0)
            retry = 0 if pd.isna(retry) else int(retry)
            futures.append(
                executor.submit(
                    _translate_row_worker,
                    client, idx, row[text_col], row[lang_col], retry, model,
                )
            )
        for f in tqdm(as_completed(futures), total=len(futures), desc="  batch", leave=False):
            idx, translated, retry_count = f.result()
            results[idx] = (translated, retry_count)
    return results


def _stats(
    df: pd.DataFrame,
    text_col: str,
    out_col: str,
    retry_count_col: str,
    retry_failed: bool,
    max_retries: int,
) -> dict:
    total = len(df)
    has_src = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    done = (
        df[out_col].notna()
        & (df[out_col].astype(str).str.strip() != "")
        & (~df[out_col].astype(str).str.strip().isin(EMPTY_MARKERS))
    ).sum()
    needs = sum(
        _needs_translation(row, text_col, out_col, retry_count_col, retry_failed, max_retries)
        for _, row in df.iterrows()
    )
    return {
        "total": total,
        "translatable": int(has_src.sum()),
        "done": int(done),
        "needs": int(needs),
        "empty_source": int(total - has_src.sum()),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_csv(
    input_csv: str,
    output_csv: str,
    checkpoint_pkl: Optional[str] = None,
    text_col: str = "content",
    lang_col: str = "language",
    out_col: str = "content_en",
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retry_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS,
    retry_failed: bool = True,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Translate *text_col* in *input_csv* and write results to *output_csv*.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    output_csv : str
        Path where the output CSV will be written.
    checkpoint_pkl : str, optional
        Path for the pickle checkpoint. Defaults to ``<output_csv>.checkpoint.pkl``.
    text_col : str
        Column containing the text to translate.
    lang_col : str
        Column containing language codes ('de', 'fr', 'it', 'en').
    out_col : str
        Column name for the English translation output.
    model : str
        Ollama model identifier.
    max_workers : int
        Number of parallel translation threads.
    batch_size : int
        Rows processed per checkpoint cycle.
    max_retry_attempts : int
        Maximum retries per row before giving up.
    retry_failed : bool
        Retry rows that previously returned empty translations.
    api_key : str, optional
        Ollama API key. Falls back to OLLAMA_API_KEY env var.

    Returns
    -------
    pd.DataFrame
        The translated dataframe (also written to *output_csv*).
    """
    retry_count_col = "_translation_retries"

    if checkpoint_pkl is None:
        checkpoint_pkl = output_csv + ".checkpoint.pkl"

    client = get_ollama_client(api_key)

    # ---- load or initialise ----
    if os.path.exists(checkpoint_pkl):
        print(f"Loading checkpoint: {checkpoint_pkl}")
        df = pd.read_pickle(checkpoint_pkl)
    else:
        print(f"Starting fresh from: {input_csv}")
        df = pd.read_csv(input_csv)
        if out_col not in df.columns:
            df[out_col] = pd.NA
        df[retry_count_col] = 0
        df.to_pickle(checkpoint_pkl)

    if retry_count_col not in df.columns:
        df[retry_count_col] = 0

    s = _stats(df, text_col, out_col, retry_count_col, retry_failed, max_retry_attempts)
    print(
        f"Rows: {s['total']}  |  translatable: {s['translatable']}  |  "
        f"done: {s['done']}  |  remaining: {s['needs']}  |  empty source: {s['empty_source']}"
    )

    if s["needs"] == 0:
        print("Nothing to translate — saving output.")
        df.to_csv(output_csv, index=False)
        return df

    total_done = 0
    t0 = time.time()

    while True:
        pending = [
            idx for idx, row in df.iterrows()
            if _needs_translation(row, text_col, out_col, retry_count_col, retry_failed, max_retry_attempts)
        ]
        if not pending:
            break

        batch_idx = pending[:batch_size]
        print(f"\nBatch {len(batch_idx)} rows  (rows {batch_idx[0]}–{batch_idx[-1]})  |  {len(pending) - len(batch_idx)} remaining after")

        t_batch = time.time()
        results = _translate_batch(
            client, df.loc[batch_idx],
            text_col, lang_col, retry_count_col, model, max_workers,
        )
        batch_secs = time.time() - t_batch

        for idx, (text, retries) in results.items():
            df.at[idx, out_col] = text
            df.at[idx, retry_count_col] = retries

        total_done += len(batch_idx)
        elapsed = time.time() - t0
        rps = total_done / elapsed if elapsed > 0 else 0
        print(f"  {batch_secs:.1f}s  |  overall {rps:.1f} rows/s", end="")
        remaining_rows = len(pending) - len(batch_idx)
        if rps > 0 and remaining_rows > 0:
            print(f"  |  ETA ~{remaining_rows / rps / 60:.1f} min", end="")
        print()

        df.to_pickle(checkpoint_pkl)

        s = _stats(df, text_col, out_col, retry_count_col, retry_failed, max_retry_attempts)
        if s["translatable"] > 0:
            pct = 100 * s["done"] / s["translatable"]
            print(f"  Progress: {s['done']}/{s['translatable']} ({pct:.1f}%)")

    df.to_csv(output_csv, index=False)
    elapsed_min = (time.time() - t0) / 60
    print(f"\nDone. {total_done} rows in {elapsed_min:.1f} min  →  {output_csv}")
    return df


def reset_failed(checkpoint_pkl: str, max_retry_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS) -> pd.DataFrame:
    """Reset failed/empty translation rows so they are retried on next run."""
    retry_count_col = "_translation_retries"
    df = pd.read_pickle(checkpoint_pkl)
    out_col = next(
        (c for c in df.columns if "content_en" in c or c.endswith("_en")), None
    )
    if out_col is None:
        raise ValueError("Cannot detect translation output column — pass it manually.")

    s = df[out_col].astype("string")
    mask = s.isna() | s.str.strip().eq("") | s.str.strip().isin(EMPTY_MARKERS)
    mask &= df.get(retry_count_col, 0) < max_retry_attempts
    n = int(mask.sum())
    df.loc[mask, out_col] = pd.NA
    if retry_count_col in df.columns:
        df.loc[mask, retry_count_col] = 0
    df.to_pickle(checkpoint_pkl)
    print(f"Reset {n} rows for retry  →  {checkpoint_pkl}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate a CSV text column to English using Ollama Cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Input CSV path.")
    p.add_argument("--output", "-o", required=True, help="Output CSV path.")
    p.add_argument("--checkpoint", default=None, help="Checkpoint .pkl path (default: <output>.checkpoint.pkl).")
    p.add_argument("--text-col", default="content", help="Column with text to translate.")
    p.add_argument("--lang-col", default="language", help="Column with language codes (de/fr/it/en).")
    p.add_argument("--out-col", default="content_en", help="Output column name for English text.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model identifier.")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel translation threads.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Rows per checkpoint cycle.")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRY_ATTEMPTS, help="Max retry attempts per row.")
    p.add_argument("--no-retry-failed", action="store_true", help="Do not retry previously empty translations.")
    p.add_argument("--api-key", default=None, help="Ollama API key (default: OLLAMA_API_KEY env var).")
    p.add_argument("--reset-failed", action="store_true", help="Reset failed rows in checkpoint and exit.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.reset_failed:
        ckpt = args.checkpoint or (args.output + ".checkpoint.pkl")
        reset_failed(ckpt, max_retry_attempts=args.max_retries)
        return

    translate_csv(
        input_csv=args.input,
        output_csv=args.output,
        checkpoint_pkl=args.checkpoint,
        text_col=args.text_col,
        lang_col=args.lang_col,
        out_col=args.out_col,
        model=args.model,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_retry_attempts=args.max_retries,
        retry_failed=not args.no_retry_failed,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
