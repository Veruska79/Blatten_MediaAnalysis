# Blatten_MediaAnalysis
This is the repository for the Blatten news analysis
# Blatten Media Analysis — Pipeline Overview

This repo processes news coverage of the Blatten (Lötschental) rockfall/debris-flow
event: cleaning raw articles, filtering for relevance, extracting structured
information with an LLM, and running downstream NLP analysis (topic modelling,
co-occurrence, summarisation).

The pipeline runs as a sequence of stages. Each stage is a standalone `.py`
module (importable from a notebook) or a notebook itself, and each stage reads
the CSV/pickle output of the previous one.

## Pipeline stages

### 1. Preprocessing — [`blatten_preprocessing.py`](blatten_preprocessing.py)
Cleans the raw news export (HTML stripping, invisible-character/dash
normalization, deduplication, etc.).
```python
from blatten_preprocessing import preprocess_news
df = preprocess_news(file_path=..., file_name=..., output_path=..., plot_dir=...)
```

### 2. Relevance filtering — [`news_classifier.py`](news_classifier.py)
A binary SetFit (few-shot) classifier that separates news actually relevant to
the Blatten event from unrelated matches.
```python
from news_classifier import classify_news
relevant_df, eval_df = classify_news(df=preprocessed_df, output_dir=..., output_suffix=...)
```

### 3. Open-vocabulary structured extraction — [`StructuredOutputWithLLM_NewsBlatten_OpenVocab.ipynb`](StructuredOutputWithLLM_NewsBlatten_OpenVocab.ipynb)
First LLM structured-output pass (LangChain + OpenAI, Pydantic schema, free
text/open vocabulary) extracting per article: event type, sentiment/tone,
locations and names, response strategy, stated causes, and personal-account
summaries ("stories").

The extraction output is saved, and **10% of the data is set aside for manual
annotation** — this manual set becomes both an evaluation reference and the
basis for the closed vocabulary used in stage 5.

### 4. Translation — [`llm_news_translate.py`](llm_news_translate.py)
Translates non-English article content (German/French/Italian → English) via
an Ollama Cloud model, with checkpointing and parallel processing.
```python
from llm_news_translate import translate_csv
translate_csv(input_csv=..., output_csv=..., checkpoint_pkl=..., text_col="content",
               lang_col="language", out_col="content_en")
```
Also runnable from the CLI — see the module docstring.

### 5. Closed-vocabulary structured extraction — [`StructuredOutputWIthLLM_NewsBlatten_Vocab.ipynb`](StructuredOutputWIthLLM_NewsBlatten_Vocab.ipynb)
Re-runs structured extraction on the translated content, this time against a
**closed vocabulary** (enum-constrained `event` / `strategy` fields) derived
from the manual annotation exercise in stage 3. This notebook also contains
the evaluation of the closed-vocabulary extraction.

### 6. Open-text evaluation — [`llm_output_evaluation.py`](llm_output_evaluation.py)
Evaluates the open-vocabulary/free-text extraction (e.g. the `stories` /
narrative fields) against the manually annotated 10% subset, using:
- embedding cosine similarity (`sentence-transformers`)
- BERTScore
- an LLM-as-judge rubric (0–10, via OpenAI structured tool calls)
```python
from llm_output_evaluation import evaluate_frames
result_df, summary = evaluate_frames(df=df2, human_col="frames_manual", llm_col="frames",
                                      output_path=..., plot_dir=...)
```

### 7. Downstream analysis — [`ContentAnalysis&Summarisation_ver2.ipynb`](ContentAnalysis&Summarisation_ver2.ipynb)
Takes the final tagged/translated dataset and runs:
- corpus characteristics (language, media type)
- hazard/event and DRM-strategy co-occurrence analysis
- TF-IDF + NMF topic modelling, t-SNE visualization, keyword tagging, temporal analysis

### 8. Topic number and media source — [`MediaSource_TopicNumber_Evaluation.ipynb`](MediaSource_TopicNumber_Evaluation.ipynb)
Analysis of topic models
- implements various test to establish a robust number of topics
- analyses the media outlets hazard/event and DRM-strategy co-occurrence analysis



## Data flow summary

```
raw news (tsv.xz)
   │  blatten_preprocessing.py
   ▼
cleaned news
   │  news_classifier.py (SetFit)
   ▼
relevant news only
   │  StructuredOutputWithLLM_NewsBlatten_OpenVocab.ipynb
   ▼
open-vocab structured extraction ──► 10% held out for manual annotation
   │  llm_news_translate.py
   ▼
translated (English) content
   │  StructuredOutputWIthLLM_NewsBlatten_Vocab.ipynb (closed vocab, using manual annotation categories)
   ▼
closed-vocab structured extraction
   │  llm_output_evaluation.py (cosine similarity / BERTScore / LLM judge, vs. manual annotation)
   ▼
final tagged dataset
   │  ContentAnalysis&Summarisation.ipynb
   ▼
topic modelling, co-occurrence analysis, summarisation, temporal evolution
```

## Requirements
See [`requirements.txt`](requirements.txt).
