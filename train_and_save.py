"""
Train and serialize both the benchmark and operational models.

Benchmark Model (Model 3): Composite Token TF-IDF + LinearSVC -> Target A (team_label)
Operational Model (Model 2): TF-IDF + Categorical OneHot + LinearSVC -> Target B (final_team)
"""

import pandas as pd
import numpy as np
import unicodedata
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

CANONICAL_MAP = {
    'Installations': 'Installs & Demo',
    'Installs & Demo': 'Installs & Demo',
    'Consumables': 'Filters & Consumables',
    'Filters & Consumables': 'Filters & Consumables',
    'Repairs': 'Repairs',
    'Billing': 'Billing',
    'Returns & Replacement': 'Returns & Replacement',
    'Warranty Claims': 'Warranty Claims',
    'Product Advice': 'Product Advice',
}

CANONICAL_TEAMS = sorted(set(CANONICAL_MAP.values()))

CAT_COLS = ['channel', 'product_family', 'warranty_status']


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ''
    t = t.replace('\u00e2\u0080\u00a6', '...')
    t = t.replace('\ufffd', ' ')
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii')
    t = re.sub(r'\bko\d+\b', '<ORDER_ID>', t, flags=re.IGNORECASE)
    t = re.sub(r'\bsr\d+\b', '<REG_ID>', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def make_composite(df: pd.DataFrame) -> pd.Series:
    return (
        'channel_' + df['channel'].astype(str) + ' '
        + 'product_' + df['product_family'].astype(str).str.replace(' ', '_') + ' '
        + 'warranty_' + df['warranty_status'].astype(str) + ' '
        + df['cleaned_text']
    )


def main():
    print('Loading data...', flush=True)
    train_df = pd.read_csv('train.csv')
    res_df = pd.read_csv('resolution_log.csv')
    merged = pd.merge(train_df, res_df, on='request_id')

    merged['target_A'] = merged['team_label'].map(CANONICAL_MAP)
    merged['target_B'] = merged['final_team'].map(CANONICAL_MAP)
    merged['cleaned_text'] = merged['request_text'].apply(clean_text)
    merged['composite_text'] = make_composite(merged)

    os.makedirs('models', exist_ok=True)

    # ── Benchmark Model (Model 3): Composite Token LinearSVC ─────────
    print('Training benchmark model (Model 3 / Target A)...', flush=True)
    benchmark_pipe = Pipeline([
        ('tfidf', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True)),
        ])),
        ('clf', LinearSVC(C=1.0, random_state=42)),
    ])
    benchmark_pipe.fit(merged['composite_text'], merged['target_A'])
    joblib.dump(benchmark_pipe, 'models/benchmark_model.joblib')
    print('  Saved models/benchmark_model.joblib', flush=True)

    # ── Operational Model (Model 2): TF-IDF + Categoricals ───────────
    print('Training operational model (Model 2 / Target B)...', flush=True)
    operational_pipe = Pipeline([
        ('prep', ColumnTransformer(
            transformers=[
                ('text', FeatureUnion([
                    ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                    ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True)),
                ]), 'cleaned_text'),
                ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
            ]
        )),
        ('clf', LinearSVC(C=1.0, random_state=42)),
    ])
    operational_pipe.fit(merged, merged['target_B'])
    joblib.dump(operational_pipe, 'models/operational_model.joblib')
    print('  Saved models/operational_model.joblib', flush=True)

    # Save class ordering for decision_function interpretation
    joblib.dump(operational_pipe.classes_, 'models/operational_classes.joblib')
    print('  Saved models/operational_classes.joblib', flush=True)

    print('Done. Both model artifacts saved to models/', flush=True)


if __name__ == '__main__':
    main()
