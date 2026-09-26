import pandas as pd
import numpy as np
import unicodedata
import re
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set random seed
RANDOM_STATE = 42

print("=== 1. LOADING AND PREPARING DATA ===", flush=True)
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_unlabelled.csv')
res_df = pd.read_csv('resolution_log.csv')

# Canonical Team Mapping
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

CANONICAL_TEAMS = sorted(list(set(CANONICAL_MAP.values())))
print(f"Canonical 7 Teams: {CANONICAL_TEAMS}", flush=True)

# Merge train with resolution_log
merged = pd.merge(train_df, res_df, on='request_id')

# Create Target A (team_label / first_team mapped) and Target B (final_team mapped)
merged['target_A'] = merged['team_label'].map(CANONICAL_MAP)
merged['target_B'] = merged['final_team'].map(CANONICAL_MAP)

# Text Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        return ''
    t = text
    # Fix known encoding corruptions
    t = t.replace('â€¦', '...')
    t = t.replace('\ufffd', ' ')
    # Normalize unicode accents (e.g. urgént -> urgent, thé -> the)
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii')
    # Normalize order IDs (KO followed by digits)
    t = re.sub(r'\bko\d+\b', '<ORDER_ID>', t, flags=re.IGNORECASE)
    # Normalize registration numbers (SR followed by digits)
    t = re.sub(r'\bsr\d+\b', '<REG_ID>', t, flags=re.IGNORECASE)
    # Normalize multiple whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

merged['cleaned_text'] = merged['request_text'].apply(clean_text)
test_df['cleaned_text'] = test_df['request_text'].apply(clean_text)

# Composite text representation (structured metadata prepended to text)
def make_composite(df):
    return (
        'channel_' + df['channel'].astype(str) + ' ' +
        'product_' + df['product_family'].astype(str).str.replace(' ', '_') + ' ' +
        'warranty_' + df['warranty_status'].astype(str) + ' ' +
        df['cleaned_text']
    )

merged['composite_text'] = make_composite(merged)
test_df['composite_text'] = make_composite(test_df)
merged['month'] = merged['created_at_ist'].str[:7]

# Date masks
train_mask = merged['created_at_ist'] < '2026-04-01'
val_mask = merged['created_at_ist'] >= '2026-04-01'

print(f"Primary Split: Train rows = {train_mask.sum()} (Apr 2025 - Mar 2026), Val rows = {val_mask.sum()} (Apr 2026 - Jun 2026)", flush=True)

# 2. FEATURE PIPELINES
print("\n=== 2. INITIALIZING PIPELINES ===", flush=True)

# Text-only Union
text_union = FeatureUnion([
    ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True))
])

# Multi-Modal ColumnTransformer (Text TF-IDF + OneHot Categoricals)
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

# Pipeline definitions
models = {
    'Model 1 (LinearSVC Pure TF-IDF)': {
        'make_pipe': lambda: Pipeline([
            ('tfidf', FeatureUnion([
                ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True))
            ])),
            ('clf', LinearSVC(C=1.0, random_state=RANDOM_STATE))
        ]),
        'get_X': lambda df: df['cleaned_text']
    },
    'Model 2 (LinearSVC TF-IDF + Categoricals)': {
        'make_pipe': lambda: Pipeline([
            ('prep', ColumnTransformer(
                transformers=[
                    ('text', FeatureUnion([
                        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                        ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True))
                    ]), 'cleaned_text'),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
                ]
            )),
            ('clf', LinearSVC(C=1.0, random_state=RANDOM_STATE))
        ]),
        'get_X': lambda df: df
    },
    'Model 2B (LogisticRegression TF-IDF + Categoricals)': {
        'make_pipe': lambda: Pipeline([
            ('prep', ColumnTransformer(
                transformers=[
                    ('text', FeatureUnion([
                        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                        ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True))
                    ]), 'cleaned_text'),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
                ]
            )),
            ('clf', LogisticRegression(C=2.0, max_iter=200, solver='lbfgs', random_state=RANDOM_STATE))
        ]),
        'get_X': lambda df: df
    },
    'Model 3 (LinearSVC Composite Tokens)': {
        'make_pipe': lambda: Pipeline([
            ('tfidf', FeatureUnion([
                ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=5, sublinear_tf=True))
            ])),
            ('clf', LinearSVC(C=1.0, random_state=RANDOM_STATE))
        ]),
        'get_X': lambda df: df['composite_text']
    }
}

# 3. RUN EVALUATIONS
print("\n=== 3. RUNNING CHRONOLOGICAL VALIDATION ===", flush=True)

results = []
detailed_reports = {}
detailed_cms = {}
val_predictions = {}

months = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06']

for target_key, target_name in [('target_A', 'TARGET A (team_label)'), ('target_B', 'TARGET B (final_team)')]:
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING {target_name}", flush=True)
    print(f"=======================================================", flush=True)
    
    for m_name, m_info in models.items():
        print(f"Evaluating {m_name} on {target_key}...", flush=True)
        pipe = m_info['make_pipe']()
        X_tr = m_info['get_X'](merged.loc[train_mask])
        y_tr = merged.loc[train_mask, target_key]
        X_val = m_info['get_X'](merged.loc[val_mask])
        y_val = merged.loc[val_mask, target_key]
        
        # Fit on primary train
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='macro')
        report = classification_report(y_val, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_val, preds, labels=CANONICAL_TEAMS)
        
        # Misroutes & Cost
        misroutes = (preds != y_val).sum()
        misroute_pct = (misroutes / len(y_val)) * 100
        # Rs 305 handling + Rs 260 extra contact = Rs 565 per misroute
        waste_cost = misroutes * 565
        
        # Rolling monthly validation
        monthly_accs = {}
        for m in months:
            m_tr_mask = merged['month'] < m
            m_te_mask = merged['month'] == m
            
            m_pipe = m_info['make_pipe']()
            m_X_tr = m_info['get_X'](merged.loc[m_tr_mask])
            m_y_tr = merged.loc[m_tr_mask, target_key]
            m_X_te = m_info['get_X'](merged.loc[m_te_mask])
            m_y_te = merged.loc[m_te_mask, target_key]
            
            m_pipe.fit(m_X_tr, m_y_tr)
            m_pred = m_pipe.predict(m_X_te)
            monthly_accs[m] = accuracy_score(m_y_te, m_pred)
            
        min_monthly = min(monthly_accs.values())
        
        res_row = {
            'Model': m_name,
            'Target': target_key,
            'Target_Name': target_name,
            'Validation Accuracy': round(acc * 100, 2),
            'Macro F1': round(f1 * 100, 2),
            'Min Monthly Accuracy': round(min_monthly * 100, 2),
            'Misroutes': int(misroutes),
            'Misroute %': round(misroute_pct, 2),
            'Waste Cost (INR)': int(waste_cost),
            'Monthly_Accs': {k: round(v * 100, 2) for k, v in monthly_accs.items()}
        }
        results.append(res_row)
        
        run_key = f"{m_name}__{target_key}"
        detailed_reports[run_key] = report
        detailed_cms[run_key] = cm.tolist()
        val_predictions[run_key] = preds
        
        print(f"  -> Val Acc: {acc*100:.2f}% | Macro F1: {f1*100:.2f}% | Min Monthly: {min_monthly*100:.2f}% | Misroutes: {misroutes} | Waste: Rs {waste_cost:,}", flush=True)

# 4. CROSS-EVALUATION
# Evaluate Target A model against Target B ground truth, and vice versa
print("\n=== 4. CROSS-TARGET EVALUATION (Apr-Jun 2026 Val Set) ===", flush=True)
y_val_A = merged.loc[val_mask, 'target_A']
y_val_B = merged.loc[val_mask, 'target_B']

# Best Target A model (Model 3) predicting on Target B ground truth
pred_A_best = val_predictions['Model 3 (LinearSVC Composite Tokens)__target_A']
acc_A_on_B = accuracy_score(y_val_B, pred_A_best)
f1_A_on_B = f1_score(y_val_B, pred_A_best, average='macro')
misroutes_A_on_B = (pred_A_best != y_val_B).sum()
waste_A_on_B = misroutes_A_on_B * 565

# Best Target B model (Model 2) predicting on Target B ground truth vs Target A
pred_B_best = val_predictions['Model 2 (LinearSVC TF-IDF + Categoricals)__target_B']
acc_B_on_A = accuracy_score(y_val_A, pred_B_best)
f1_B_on_A = f1_score(y_val_A, pred_B_best, average='macro')
misroutes_B_on_B = (pred_B_best != y_val_B).sum()
waste_B_on_B = misroutes_B_on_B * 565

print(f"Target A Model (Bot Replica) evaluated against Target B (True Resolution Ground Truth):", flush=True)
print(f"  Accuracy on True Resolution: {acc_A_on_B*100:.2f}% | True Misroutes: {misroutes_A_on_B} | Operational Waste: Rs {waste_A_on_B:,}", flush=True)

print(f"Target B Model (True Resolution) evaluated against Target B Ground Truth:", flush=True)
print(f"  Accuracy on True Resolution: {accuracy_score(y_val_B, pred_B_best)*100:.2f}% | True Misroutes: {misroutes_B_on_B} | Operational Waste: Rs {waste_B_on_B:,}", flush=True)
print(f"  Operational Savings of Target B model over Target A model: Rs {waste_A_on_B - waste_B_on_B:,} in 3 months (Rs {(waste_A_on_B - waste_B_on_B)*4:,}/year)!", flush=True)

# 5. RETRAIN ON FULL TRAIN DATA AND GENERATE PREDICTIONS
print("\n=== 5. RETRAINING ON FULL DATA AND GENERATING TEST PREDICTIONS ===", flush=True)

# Retrain Model 3 on full train for TARGET A (Scored Target)
full_pipe_A = models['Model 3 (LinearSVC Composite Tokens)']['make_pipe']()
full_pipe_A.fit(merged['composite_text'], merged['target_A'])
test_preds_A = full_pipe_A.predict(test_df['composite_text'])

# Output predictions.csv
sub_A = pd.DataFrame({
    'request_id': test_df['request_id'],
    'team': test_preds_A
})
sub_A.to_csv('predictions.csv', index=False)
print("Saved predictions.csv (TARGET A - Scored Submission)", flush=True)

# Retrain Model 2 on full train for TARGET B (Operational Target)
full_pipe_B = models['Model 2 (LinearSVC TF-IDF + Categoricals)']['make_pipe']()
full_pipe_B.fit(merged, merged['target_B'])
test_preds_B = full_pipe_B.predict(test_df)

# Output operational_predictions.csv
sub_B = pd.DataFrame({
    'request_id': test_df['request_id'],
    'team': test_preds_B
})
sub_B.to_csv('operational_predictions.csv', index=False)
print("Saved operational_predictions.csv (TARGET B - Operational Solution)", flush=True)

# 6. VERIFICATION OF DELIVERABLES
print("\n=== 6. VERIFYING SUBMISSION INTEGRITY ===", flush=True)
for fname in ['predictions.csv', 'operational_predictions.csv']:
    df = pd.read_csv(fname)
    print(f"Checking {fname}:")
    print(f"  Row count: {len(df)} (Expected: 2178) -> {'PASS' if len(df) == 2178 else 'FAIL'}")
    print(f"  Columns: {list(df.columns)} (Expected: ['request_id', 'team']) -> {'PASS' if list(df.columns) == ['request_id', 'team'] else 'FAIL'}")
    print(f"  Exact ID match with test_unlabelled.csv: {(df['request_id'] == test_df['request_id']).all()} -> {'PASS' if (df['request_id'] == test_df['request_id']).all() else 'FAIL'}")
    unique_teams = sorted(df['team'].unique())
    invalid_teams = set(unique_teams) - set(CANONICAL_TEAMS)
    print(f"  Predicted Teams: {unique_teams}")
    print(f"  Only 7 Canonical Teams present: {len(invalid_teams) == 0} -> {'PASS' if len(invalid_teams) == 0 else 'FAIL'}")
    print(f"  Null values: {df.isnull().sum().to_dict()}")
    print()

# Save all results to json for reporting
save_payload = {
    'results': results,
    'cross_eval': {
        'acc_A_on_B': round(acc_A_on_B * 100, 2),
        'f1_A_on_B': round(f1_A_on_B * 100, 2),
        'misroutes_A_on_B': int(misroutes_A_on_B),
        'waste_A_on_B': int(waste_A_on_B),
        'acc_B_on_B': round(accuracy_score(y_val_B, pred_B_best) * 100, 2),
        'f1_B_on_B': round(f1_score(y_val_B, pred_B_best, average='macro') * 100, 2),
        'misroutes_B_on_B': int(misroutes_B_on_B),
        'waste_B_on_B': int(waste_B_on_B),
        'acc_B_on_A': round(acc_B_on_A * 100, 2),
        'quarterly_savings': int(waste_A_on_B - waste_B_on_B),
        'annual_savings': int((waste_A_on_B - waste_B_on_B) * 4)
    },
    'detailed_reports': detailed_reports,
    'detailed_cms': detailed_cms,
    'canonical_teams': CANONICAL_TEAMS
}

with open('validation_results.json', 'w') as f:
    json.dump(save_payload, f, indent=2)
print("Saved validation_results.json successfully.", flush=True)
