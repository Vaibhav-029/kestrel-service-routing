import pandas as pd
import numpy as np
import unicodedata, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

train = pd.read_csv('train.csv')
res = pd.read_csv('resolution_log.csv')
merged = pd.merge(train, res, on='request_id')

CANONICAL_MAP = {
    'Installations': 'Installs & Demo',
    'Installs & Demo': 'Installs & Demo',
    'Consumables': 'Filters & Consumables',
    'Filters & Consumables': 'Filters & Consumables',
    'Repairs': 'Repairs',
    'Billing': 'Billing',
    'Returns & Replacement': 'Returns & Replacement',
    'Warranty Claims': 'Warranty Claims',
    'Product Advice': 'Product Advice'
}

merged['target_B'] = merged['final_team'].map(CANONICAL_MAP)

def clean_text(t):
    if not isinstance(t, str):
        return ''
    t = t.replace('â€¦', '...')
    t = t.replace('\ufffd', ' ')
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii')
    t = re.sub(r'\bko\d+\b', '<ORDER_ID>', t, flags=re.IGNORECASE)
    t = re.sub(r'\bsr\d+\b', '<REG_ID>', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

merged['cleaned_text'] = merged['request_text'].apply(clean_text)

train_idx = merged['created_at_ist'] < '2026-04-01'
val_idx = merged['created_at_ist'] >= '2026-04-01'

cat_cols = ['channel', 'product_family', 'warranty_status']
preprocessor_m2 = ColumnTransformer(
    transformers=[
        ('text', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True))
        ]), 'cleaned_text'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

pipe_b = Pipeline([
    ('prep', preprocessor_m2),
    ('clf', LinearSVC(C=1.0, random_state=42))
])

pipe_b.fit(merged.loc[train_idx], merged.loc[train_idx, 'target_B'])
val_df = merged.loc[val_idx].copy()
val_df['pred_B'] = pipe_b.predict(val_df)

print("=== 1. Returns & Replacement predicted as Repairs ===")
for _, r in val_df[(val_df['target_B']=='Returns & Replacement') & (val_df['pred_B']=='Repairs')].head(10).iterrows():
    print(f"[{r['product_family']} | {r['warranty_status']}] {r['cleaned_text']}")

print("\n=== 2. Product Advice predicted as Repairs ===")
for _, r in val_df[(val_df['target_B']=='Product Advice') & (val_df['pred_B']=='Repairs')].head(10).iterrows():
    print(f"[{r['product_family']} | {r['warranty_status']}] {r['cleaned_text']}")

print("\n=== 3. Installs & Demo predicted as Repairs ===")
for _, r in val_df[(val_df['target_B']=='Installs & Demo') & (val_df['pred_B']=='Repairs')].head(10).iterrows():
    print(f"[{r['product_family']} | {r['warranty_status']}] {r['cleaned_text']}")

print("\n=== 4. Repairs predicted as Returns & Replacement ===")
for _, r in val_df[(val_df['target_B']=='Repairs') & (val_df['pred_B']=='Returns & Replacement')].head(10).iterrows():
    print(f"[{r['product_family']} | {r['warranty_status']}] {r['cleaned_text']}")

print("\n=== 5. Repairs predicted as Installs & Demo ===")
for _, r in val_df[(val_df['target_B']=='Repairs') & (val_df['pred_B']=='Installs & Demo')].head(10).iterrows():
    print(f"[{r['product_family']} | {r['warranty_status']}] {r['cleaned_text']}")
