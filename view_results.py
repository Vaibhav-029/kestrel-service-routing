import json
import pandas as pd

with open('validation_results.json') as f:
    d = json.load(f)

print("=== CANDIDATE MODEL SUMMARY ===")
rows = []
for r in d['results']:
    rows.append({
        'Model': r['Model'],
        'Target': r['Target_Name'],
        'Val Accuracy (%)': r['Validation Accuracy'],
        'Macro F1 (%)': r['Macro F1'],
        'Min Monthly (%)': r['Min Monthly Accuracy'],
        'Misroutes': r['Misroutes'],
        'Waste (INR)': f"Rs {r['Waste Cost (INR)']:,}",
        'Jan 26': r['Monthly_Accs']['2026-01'],
        'Feb 26': r['Monthly_Accs']['2026-02'],
        'Mar 26': r['Monthly_Accs']['2026-03'],
        'Apr 26': r['Monthly_Accs']['2026-04'],
        'May 26': r['Monthly_Accs']['2026-05'],
        'Jun 26': r['Monthly_Accs']['2026-06'],
    })

df_res = pd.DataFrame(rows)
print(df_res.to_string(index=False))

print("\n=== CROSS-TARGET OPERATIONAL COMPARISON ===")
for k, v in d['cross_eval'].items():
    print(f"  {k}: {v}")

print("\n=== TARGET A BEST MODEL PER-CLASS METRICS (Model 3) ===")
rep_A = d['detailed_reports']['Model 3 (LinearSVC Composite Tokens)__target_A']
df_cls_A = pd.DataFrame(rep_A).transpose()
print(df_cls_A.round(4).to_string())

print("\n=== TARGET B BEST MODEL PER-CLASS METRICS (Model 2) ===")
rep_B = d['detailed_reports']['Model 2 (LinearSVC TF-IDF + Categoricals)__target_B']
df_cls_B = pd.DataFrame(rep_B).transpose()
print(df_cls_B.round(4).to_string())

print("\n=== TARGET A CONFUSION MATRIX (Model 3) ===")
cm_A = pd.DataFrame(d['detailed_cms']['Model 3 (LinearSVC Composite Tokens)__target_A'],
                    index=d['canonical_teams'], columns=d['canonical_teams'])
print(cm_A.to_string())

print("\n=== TARGET B CONFUSION MATRIX (Model 2) ===")
cm_B = pd.DataFrame(d['detailed_cms']['Model 2 (LinearSVC TF-IDF + Categoricals)__target_B'],
                    index=d['canonical_teams'], columns=d['canonical_teams'])
print(cm_B.to_string())
