"""
SetFit-based binary news classifier.

Usage from another notebook:
    from news_classifier import classify_news
    relevant_df, eval_df = classify_news(
        df=preprocessed_df,            # output of blatten_preprocessing.preprocess_news()
        output_dir='../../Blatten/data',
        output_suffix='_20251201', #adapted to your needs
    )
"""

import os
import warnings
from collections import Counter
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# NewsClassifier
# ---------------------------------------------------------------------------

class NewsClassifier:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.trainer = None

    def prepare_data(self, df, text_columns=None, label_column='label', separator=' [SEP] '):
        if text_columns is None:
            text_columns = ['content']

        available_text_cols = [c for c in text_columns if c in df.columns]
        print(f"Using text columns: {available_text_cols}")

        def combine_texts(row):
            parts = []
            for col in available_text_cols:
                val = row[col]
                if pd.notna(val) and str(val).strip() and str(val).lower() not in ('none', 'nan', ''):
                    parts.append(f"{col}: {str(val).strip()}")
            return separator.join(parts) if parts else "no_content_available"

        df = df.copy()
        df['combined_text'] = df.apply(combine_texts, axis=1)
        df = df.drop_duplicates(subset=['combined_text']).reset_index(drop=True)

        labeled = df[df[label_column].notna()].copy()
        unlabeled = df[df[label_column].isna()].copy()

        print(f"Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}")
        if len(labeled):
            print(f"Class distribution: {dict(labeled[label_column].value_counts())}")

        return labeled, unlabeled

    def _balance(self, df, label_column='label'):
        counts = Counter(df[label_column])
        minority = min(counts, key=counts.get)
        majority = max(counts, key=counts.get)

        minority_df = df[df[label_column] == minority]
        majority_df = df[df[label_column] == majority]

        oversampled = minority_df.sample(n=len(majority_df), replace=True, random_state=42)
        balanced = pd.concat([majority_df, oversampled], ignore_index=True).sample(frac=1, random_state=42)
        print(f"After balancing: {dict(Counter(balanced[label_column]))}")
        return balanced

    def train(self, train_df, text_column='combined_text', label_column='label',
              num_iterations=20, num_epochs=1, batch_size=16):
        from datasets import Dataset
        from setfit import SetFitModel, SetFitTrainer

        os.environ["WANDB_MODE"] = "disabled"
        self.model = SetFitModel.from_pretrained(self.model_name)

        dataset = Dataset.from_dict({
            "text": train_df[text_column].tolist(),
            "label": train_df[label_column].astype(int).tolist(),
        })

        self.trainer = SetFitTrainer(
            model=self.model,
            train_dataset=dataset,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        print("Training SetFit model...")
        self.trainer.train()
        print("Training complete.")

    def evaluate(self, test_df, text_column='combined_text', label_column='label',
                 plot_dir=None):
        if self.model is None:
            raise ValueError("Model not trained.")

        texts = test_df[text_column].tolist()
        labels = test_df[label_column].astype(int).tolist()
        preds = self._to_numpy(self.model(texts))

        print(classification_report(labels, preds, target_names=['Class 0', 'Class 1']))
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:\n", cm)

        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Class 0', 'Class 1'],
                        yticklabels=['Class 0', 'Class 1'], ax=ax)
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')
            ax.set_title('Confusion Matrix')
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'confusion_matrix.png'), dpi=300)
            plt.close(fig)

        try:
            proba = self._to_numpy(self.model.predict_proba(texts))
            if len(np.unique(labels)) > 1:
                p = proba[:, 1] if proba.ndim > 1 else proba
                print(f"ROC AUC: {roc_auc_score(labels, p):.4f}")
        except Exception:
            pass

        return preds

    def predict(self, unlabeled_df, text_column='combined_text'):
        if self.model is None:
            raise ValueError("Model not trained.")

        texts = unlabeled_df[text_column].tolist()
        preds = self._to_numpy(self.model(texts))

        result = unlabeled_df.copy()
        result['predicted_label'] = preds

        try:
            proba = self._to_numpy(self.model.predict_proba(texts))
            if proba.ndim > 1 and proba.shape[1] > 1:
                result['probability_0'] = proba[:, 0]
                result['probability_1'] = proba[:, 1]
                result['confidence'] = proba.max(axis=1)
            else:
                result['probability'] = proba
                result['confidence'] = np.abs(proba - 0.5) + 0.5
        except Exception:
            pass

        print(f"Predictions: {dict(Counter(preds))}")
        if 'confidence' in result.columns:
            print(f"Avg confidence: {result['confidence'].mean():.3f}")

        return result

    @staticmethod
    def _to_numpy(data):
        if hasattr(data, 'detach'):
            return data.detach().cpu().numpy()
        if hasattr(data, 'numpy'):
            return data.numpy()
        return np.array(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def classify_news(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    text_columns: list | None = None,
    label_column: str = 'label',
    test_size: float = 0.2,
    num_iterations: int = 20,
    num_epochs: int = 1,
    batch_size: int = 16,
    eval_sample_size: int = 215,
    output_dir: str | None = None,
    output_suffix: str | None = None,
    plot_dir: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train a SetFit classifier on the labeled rows of *df*, predict the unlabeled
    rows, and return the filtered relevant articles together with an evaluation
    sample.

    Parameters
    ----------
    df               : preprocessed DataFrame (output of blatten_preprocessing)
    model_name       : sentence-transformer checkpoint for SetFit
    text_columns     : columns to combine into the classifier input text
    label_column     : column that holds human labels (1=relevant, 0=irrelevant, NaN=unlabeled)
    test_size        : fraction of labeled data held out for evaluation
    num_iterations   : SetFit contrastive training iterations
    num_epochs       : SetFit classifier epochs
    batch_size       : training batch size
    eval_sample_size : number of rows to draw for the evaluation CSV
    output_dir       : if given, CSVs are written here
    output_suffix    : appended to output file names; defaults to today's date as '_YYYYMMDD'
    plot_dir         : if given, evaluation plots are written here
    random_state     : reproducibility seed

    Returns
    -------
    relevant_df : articles with label==1 or predicted_label==1
    eval_df     : random sample of *eval_sample_size* rows from relevant_df
    """
    if text_columns is None:
        text_columns = ['strategy', 'stories', 'analysis', 'topic']
    if output_suffix is None:
        output_suffix = '_' + date.today().strftime('%Y%m%d')

    clf = NewsClassifier(model_name=model_name)

    # --- prepare ---
    labeled, unlabeled = clf.prepare_data(df, text_columns=text_columns,
                                          label_column=label_column)

    if len(labeled) < 2:
        raise ValueError("Not enough labeled examples to train (need at least 2).")

    # --- split ---
    train_data, test_data = train_test_split(
        labeled,
        test_size=test_size,
        stratify=labeled[label_column],
        random_state=random_state,
    )
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # --- balance training set ---
    balanced_train = clf._balance(train_data, label_column=label_column)

    # --- train ---
    clf.train(balanced_train, num_iterations=num_iterations,
              num_epochs=num_epochs, batch_size=batch_size)

    # --- evaluate ---
    clf.evaluate(test_data, plot_dir=plot_dir)

    # --- predict unlabeled ---
    predictions_df = clf.predict(unlabeled)

    # --- combine train / test / predicted ---
    extra_cols = ['probability_0', 'probability_1', 'confidence', 'predicted_label']
    train_tagged = train_data.copy()
    train_tagged['source'] = 'train'
    test_tagged = test_data.copy()
    test_tagged['source'] = 'test'
    predictions_df['source'] = 'predicted'

    for col in extra_cols:
        for part in (train_tagged, test_tagged):
            if col not in part.columns:
                part[col] = pd.NA

    combined_df = pd.concat([train_tagged, test_tagged, predictions_df],
                             ignore_index=True)

    # --- filter relevant ---
    relevant_df = combined_df.query(
        f"{label_column} == 1 or predicted_label == 1"
    ).copy()

    # --- evaluation sample ---
    n = min(eval_sample_size, len(relevant_df))
    eval_df = relevant_df.sample(n=n, random_state=random_state)

    print(f"\nRelevant articles: {len(relevant_df)}")
    print(f"Evaluation sample: {len(eval_df)}")

    # --- save outputs ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        relevant_path = os.path.join(output_dir, f"news_predictions_all_relevant{output_suffix}.csv")
        eval_path = os.path.join(output_dir, f"news_eval{output_suffix}.csv")
        relevant_df.to_csv(relevant_path, index=False)
        eval_df.to_csv(eval_path, index=False)
        print(f"Saved: {relevant_path}")
        print(f"Saved: {eval_path}")

    return relevant_df, eval_df


# -------------------------------------------------------------------------------------------
# CLI convenience to run from terminal if needed: python news_classifier.py data.csv outputs/
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python news_classifier.py <input_csv> [output_dir]") #we used the following file: 2025_Blatten_News_selected_tagged_20251128 - 2025_Blatten_News_selected_tagged_20251201.csv
        sys.exit(1)

    input_csv = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(input_csv)

    df_in = pd.read_csv(input_csv)
    relevant, evaluation = classify_news(df_in, output_dir=out_dir)
    print(f"Done. {len(relevant)} relevant articles, {len(evaluation)} eval rows.")
