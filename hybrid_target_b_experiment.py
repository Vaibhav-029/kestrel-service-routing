import pandas as pd
import numpy as np
import unicodedata, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

print("=== POLICY-GUIDED HYBRID EXPERIMENT FOR TARGET B ===", flush=True)

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
CANONICAL_TEAMS = sorted(list(set(CANONICAL_MAP.values())))

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
merged['month'] = merged['created_at_ist'].str[:7]

train_mask = merged['created_at_ist'] < '2026-04-01'
val_mask = merged['created_at_ist'] >= '2026-04-01'

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

# Baseline ML model (Model 2)
baseline_pipe = Pipeline([
    ('prep', preprocessor_m2),
    ('clf', LinearSVC(C=1.0, random_state=42))
])

baseline_pipe.fit(merged.loc[train_mask], merged.loc[train_mask, 'target_B'])
val_df = merged.loc[val_mask].copy()
raw_preds = baseline_pipe.predict(val_df)
y_val = val_df['target_B'].values

base_acc = accuracy_score(y_val, raw_preds)
base_f1 = f1_score(y_val, raw_preds, average='macro')

print(f"1. BASELINE TARGET B (Model 2):", flush=True)
print(f"   Validation Accuracy: {base_acc*100:.2f}%", flush=True)
print(f"   Macro F1-Score:      {base_f1*100:.2f}%", flush=True)

# Define Policy-Guided Hybrid Function
# Derived strictly from ops-policy.pdf and teams.csv
def apply_policy_rules(preds, df):
    new_preds = list(preds)
    
    # Specific regex patterns from policy & teams
    p_billing_issue = r'\b(invoice not received|gst invoice|double charge|charged twice|refund not credited|emi conversion not done|coupon discount not applied|payment deducted but order .* not placed|need corrected bill)\b'
    p_payment_mention = r'\b(paid on upi|paid via card|paid by emi|payment done|already paid in full|i paid extra for this)\b'
    p_fault = r'\b(not working|stopped working|not turning on|tripping the mcb|leaking water|burnt smell|loud noise|error code|display .* gone blank)\b'
    p_spares = r'\b(order candle and membrane|spare blade set|where to buy spare jar|brush roll spare|replacement filter|amc filter kit|consumable pack .* price)\b'
    p_delivery = r'\b(arrived damaged|box was open|scratched|wrong model .* delivered|missing parts in box|received used|want exchange|return pickup|cancel and return|want to return)\b'
    p_install = r'\b(demo and installation|installation slot|need installation|wall mounting|technician come to install|delivered but not installed|installer did not turn up|reschedule installation)\b'
    p_advice = r'\b(which .* is right for|how to clean|recipe booklet|safe for kids|can .* run on inverter|difference between lite and pro|best settings|power consumption)\b'
    p_warranty = r'\b(claim status .* under warranty|warranty card .* not registered|warranty rejected|shield plan renewal|need warranty certificate|covered under shield plan|extend warranty)\b'
    
    corrections = 0
    harmful_overrides = 0
    neutral_overrides = 0
    
    for i in range(len(df)):
        text = df['cleaned_text'].iloc[i].lower()
        curr_p = preds[i]
        true_y = y_val[i]
        new_p = curr_p
        
        # Rule 1 (Ops Policy §3): Billing rule
        # A request belongs to Billing only when problem IS the payment.
        # Customer mentioning they paid does NOT make it Billing.
        # If predicted Billing, but text mentions fault and NO billing issue -> Repairs
        if curr_p == 'Billing':
            if re.search(p_fault, text) and not re.search(p_billing_issue, text):
                new_p = 'Repairs'
            elif re.search(p_install, text) and not re.search(p_billing_issue, text):
                new_p = 'Installs & Demo'
            elif re.search(p_delivery, text) and not re.search(p_billing_issue, text):
                new_p = 'Returns & Replacement'
                
        # Rule 2 (Teams.csv): Consumables is selling/fitting spares, NOT faults.
        # If predicted Filters & Consumables, but text has clear fault and NO spares order -> Repairs
        elif curr_p == 'Filters & Consumables':
            if re.search(p_fault, text) and not re.search(p_spares, text):
                new_p = 'Repairs'
                
        # Rule 3 (Teams.csv): Returns & Replacement handles delivery damage / returns.
        # If predicted Repairs, but text has explicit delivery damage/return and NO fault after usage -> Returns & Replacement
        elif curr_p == 'Repairs':
            if re.search(p_delivery, text) and not re.search(p_fault, text):
                new_p = 'Returns & Replacement'
            elif re.search(p_advice, text) and not re.search(p_fault, text):
                new_p = 'Product Advice'
            elif re.search(p_warranty, text) and not re.search(p_fault, text):
                new_p = 'Warranty Claims'
                
        if new_p != curr_p:
            new_preds[i] = new_p
            if new_p == true_y and curr_p != true_y:
                corrections += 1
            elif new_p != true_y and curr_p == true_y:
                harmful_overrides += 1
            else:
                neutral_overrides += 1
                
    return np.array(new_preds), corrections, harmful_overrides, neutral_overrides

hybrid_preds, corr, harm, neut = apply_policy_rules(raw_preds, val_df)
hyb_acc = accuracy_score(y_val, hybrid_preds)
hyb_f1 = f1_score(y_val, hybrid_preds, average='macro')

print(f"\n2. HYBRID TARGET B (Model 2 + Policy Rules):", flush=True)
print(f"   Validation Accuracy: {hyb_acc*100:.2f}% (Delta: {(hyb_acc - base_acc)*100:+.2f}%)", flush=True)
print(f"   Macro F1-Score:      {hyb_f1*100:.2f}% (Delta: {(hyb_f1 - base_f1)*100:+.2f}%)", flush=True)
print(f"   Rule Overrides: Total={corr+harm+neut}, Positive Corrections={corr}, Harmful Overrides={harm}, Neutral={neut}", flush=True)
print(f"   Net Accuracy Gain: {corr - harm} correct predictions out of {len(val_df)}", flush=True)

# 3. ROLLING MONTHLY EVALUATION JAN-JUN 2026 FOR BOTH BASELINE AND HYBRID
print("\n3. ROLLING MONTHLY STABILITY (Jan - Jun 2026):", flush=True)
months = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06']
base_monthly = {}
hyb_monthly = {}

for m in months:
    m_tr = merged['month'] < m
    m_te = merged['month'] == m
    
    pipe = Pipeline([
        ('prep', preprocessor_m2),
        ('clf', LinearSVC(C=1.0, random_state=42))
    ])
    pipe.fit(merged.loc[m_tr], merged.loc[m_tr, 'target_B'])
    
    test_m_df = merged.loc[m_te].copy()
    m_y_true = test_m_df['target_B'].values
    m_raw_preds = pipe.predict(test_m_df)
    
    # Apply rules
    m_hyb_preds, _, _, _ = apply_policy_rules(m_raw_preds, test_m_df)
    
    base_monthly[m] = accuracy_score(m_y_true, m_raw_preds)
    hyb_monthly[m] = accuracy_score(m_y_true, m_hyb_preds)

print(f"{'Month':<10} | {'Baseline Acc':<15} | {'Hybrid Acc':<15} | {'Delta':<10}")
print("-" * 55)
for m in months:
    b_a = base_monthly[m] * 100
    h_a = hyb_monthly[m] * 100
    d_a = h_a - b_a
    print(f"{m:<10} | {b_a:.2f}%{'':<9} | {h_a:.2f}%{'':<9} | {d_a:+.2f}%")

print("-" * 55)
print(f"{'Min Month':<10} | {min(base_monthly.values())*100:.2f}%{'':<9} | {min(hyb_monthly.values())*100:.2f}%{'':<9}")

# 4. CONFUSION MATRICES
print("\n4. CONFUSION MATRIX (Hybrid Target B):", flush=True)
cm_hyb = pd.DataFrame(confusion_matrix(y_val, hybrid_preds, labels=CANONICAL_TEAMS),
                      index=CANONICAL_TEAMS, columns=CANONICAL_TEAMS)
print(cm_hyb.to_string())

print("\n5. CLASSIFICATION REPORT (Hybrid Target B):", flush=True)
print(classification_report(y_val, hybrid_preds, target_names=CANONICAL_TEAMS, zero_division=0))
