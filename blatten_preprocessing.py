"""
Blatten news preprocessing pipeline.

Usage from another notebook:
    from blatten_preprocessing import preprocess_news
    df = preprocess_news(
        file_path='../../Blatten/data/NewsArticles/',
        file_name='Blatten_20250511to20251127.tsv.xz',
        output_path='../../Blatten/data/cleaned_news.csv',
        plot_dir='../../Blatten/data/plots',
    )
"""

import os
import re
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_invisible_chars(text):
    if text is None:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[​-\u200F﻿⁠]', '', text)
    text = re.sub(r'[‐-―−⁃]', '-', text)
    text = re.sub(r'^\s*-\s*', '', text)
    return text.strip()


def clean_text(text):
    if text is None or pd.isna(text):
        return text
    try:
        text = text.encode('utf-8').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    text = remove_invisible_chars(text)
    text = re.sub(r'\s*>{1,}\.*\s*', ' ', text)
    text = re.sub(r'\b\d{1,2}h\d{1,2}\b', ' ', text)
    text = re.sub(r'\b\d{1,2}:\d{2}\b', ' ', text)
    text = re.sub(r'\b\d{1,4}\b', ' ', text)
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([A-Z][a-z]+)([A-Z][a-z]+)', r'\1 \2', text)
    text = re.sub(r'(^|\s)-\s*', r'\1', text)
    text = re.sub(r'\s*\.\s*\.', '.', text)
    text = re.sub(r'\s*([,;:.!?])\s*', r'\1 ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_weekly_counts(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    weekly_counts = df.groupby("week").size().reset_index(name="news_count")
    weekly_counts["rolling_avg"] = weekly_counts["news_count"].rolling(window=4, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=weekly_counts, x="week", y="news_count", marker="o", label="Weekly News")
    sns.lineplot(data=weekly_counts, x="week", y="rolling_avg", marker="o", linestyle=":", label="4-week Rolling Avg")
    plt.xticks(weekly_counts["week"], rotation=45)
    plt.title("News Articles per Week")
    plt.xlabel("")
    plt.ylabel("News Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "NewsArticlesWeek.png"), bbox_inches='tight', dpi=400)
    plt.close()

    for doc_type in df["doctype_description"].unique():
        df_doc = df[df["doctype_description"] == doc_type]
        wc = df_doc.groupby("week").size().reset_index(name="news_count")
        wc["rolling_avg"] = wc["news_count"].rolling(window=4, min_periods=1).mean()

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=wc, x="week", y="news_count", marker="o", label="Weekly News")
        sns.lineplot(data=wc, x="week", y="rolling_avg", marker="o", linestyle=":", label="4-week Rolling Avg")
        plt.xticks(wc["week"], rotation=45)
        plt.title(f"News Articles per Week – {doc_type}")
        plt.xlabel("")
        plt.ylabel("News Count")
        plt.tight_layout()
        safe_name = doc_type.replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"weekly_news_{safe_name}.png"), dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_news(
    file_path: str,
    file_name: str,
    output_path: str | None = None,
    plot_dir: str | None = None,
    drop_romansh: bool = True,
) -> pd.DataFrame:
    """
    Load, filter, clean, and optionally save a Swissdox news TSV.

    Parameters
    ----------
    file_path   : directory containing the TSV/TSV.XZ file
    file_name   : file name (e.g. 'Blatten_20250511to20251127.tsv.xz')
    output_path : if given, saves the cleaned DataFrame to this CSV path
    plot_dir    : if given, generates and saves weekly-count plots here
    drop_romansh: drop rows with language == 'rm' (default True)

    Returns
    -------
    Cleaned pd.DataFrame
    """
    df = pd.read_csv(os.path.join(file_path, file_name), delimiter='\t', encoding='utf-8')

    if drop_romansh:
        df = df.query('language != "rm"')

    df = df.drop_duplicates(subset=['head'], keep='first')
    df['content'] = df['content'].apply(remove_html_tags)
    df['clean_content'] = df['content'].apply(clean_text)

    df["pubtime"] = pd.to_datetime(df["pubtime"], utc=True)
    df = df.sort_values(by='pubtime')
    df['date'] = df["pubtime"].dt.date
    df["week"] = df["pubtime"].dt.to_period("W").apply(lambda r: r.start_time)

    if plot_dir:
        plot_weekly_counts(df, plot_dir)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df.to_csv(output_path, encoding='utf-8', index=False)

    return df


if __name__ == "__main__":
    df = preprocess_news(
        file_path='../../Blatten/data/NewsArticles/',
        file_name='Blatten_20250511to20251127.tsv.xz',
        output_path='../../Blatten/data/cleaned_news.csv',
        plot_dir='../../Blatten/data/plots',
    )
    print(df['language'].value_counts())
    print(f"Total articles: {len(df)}")
