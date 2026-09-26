# Kestrel Service-Request Routing — Validation Report

**Author:** Machine Learning Engineering Team  
**Dataset:** Kestrel Home Task 2 V2 (Variant B)  
**Date:** September 2026  
**Status:** Complete  

---

## 1. Executive Summary

This validation report evaluates candidate routing models for Kestrel Home's customer service desk across two distinct objectives:
1. **Target A (Assessment / Scored Target — `team_label`)**: Replicating the legacy vendor routing bot's intake classification queue to satisfy management's historical benchmark threshold ($90\%+$ match).
2. **Target B (Operational Ground Truth — `final_team`)**: Routing incoming service requests directly to the team that actually resolves the issue, eliminating unnecessary transfers and operational waste as specified in **Kestrel Operations Policy v4.1**.

### Key Findings
- **Target A Benchmark**: **Model 3 (LinearSVC with Composite Structured Tokens)** achieves **96.96% Primary Validation Accuracy** and **96.82% Macro F1-score**, exceeding the client's $90\%$ benchmark by nearly $7$ percentage points. Across all 6 rolling evaluation months from January to June 2026, the model maintains a minimum monthly accuracy of **96.22%**.
- **Target B Operational Truth**: **Model 2 (LinearSVC with TF-IDF and One-Hot Categoricals)** achieves **85.20% Primary Validation Accuracy** and **84.94% Macro F1-score** on the true resolving queue.
- **Financial & Operational Impact**: The legacy vendor bot misrouted **24.91%** of tickets, causing **3,902 transfers** and costing Kestrel **Rs 18.91 lakh** in operational waste over 15 months ($\approx$ **Rs 15.13 lakh/year**), alongside the existing **Rs 3.20 lakh/year** bot licence. On the Apr–Jun 2026 historical holdout, the operational model produced 171 fewer estimated misroutes than the benchmark model. Annualized savings (~Rs 386,460/year) are an estimate assuming the observed validation pattern repeats; they are not observed production savings. Deployment creates the option to retire the existing Rs 3.2 lakh/year bot licence, subject to operational approval. Local inference has no paid API dependency, and measured local inference cost is effectively zero per request; production hosting cost is not estimated because it was not measured/priced.

---

## 2. Validation Methodology

To guard against temporal data leakage, **no random cross-validation was used**. Inbound support data inherently drifts over time due to system migrations, policy updates, and team renames.

### Validation Structure
1. **Primary Time-Based Split (3-Month Holdout)**:
   - **Training Set**: Requests created between **2025-04-01** and **2026-03-31** (12 months, **8,687 requests**).
   - **Validation Set**: Requests created between **2026-04-01** and **2026-06-30** (3 months, **2,135 requests**).
   - *Rationale*: The validation duration (3 months) exactly mirrors the test set duration (**2026-07-01 to 2026-09-30**, 2,178 requests) and takes place entirely within the unified modern CRM era post-15 Jan 2026.
2. **Rolling Monthly Validation (Jan–Jun 2026)**:
   - For each month $m \in \{\text{2026-01}, \text{2026-02}, \text{2026-03}, \text{2026-04}, \text{2026-05}, \text{2026-06}\}$, models were trained strictly on data prior to month $m$ and evaluated on month $m$.
   - This measures temporal stability and verifies that performance does not degrade over time.
3. **Canonical Class Mapping**:
   Historical team renames (effective 15 Jan 2026) were mapped into the 7 active business queues:
   - `Installations` $\rightarrow$ `Installs & Demo`
   - `Consumables` $\rightarrow$ `Filters & Consumables`
   - `Repairs`, `Billing`, `Returns & Replacement`, `Warranty Claims`, `Product Advice` (unchanged).

---

## 3. Model Benchmark Summary

| Model | Target | Primary Val Acc (%) | Macro F1 (%) | Min Monthly Acc (%) | Val Misroutes | Est. Val Waste (INR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (LinearSVC Pure TF-IDF)** | **Target A (`team_label`)** | 95.78% | 95.82% | 94.62% | 90 / 2,135 | Rs 50,850 |
| **Model 2 (LinearSVC TF-IDF + Cat)** | **Target A (`team_label`)** | 96.63% | 96.44% | 96.19% | 72 / 2,135 | Rs 40,680 |
| **Model 2B (LogReg TF-IDF + Cat)** | **Target A (`team_label`)** | 95.69% | 95.51% | 95.37% | 92 / 2,135 | Rs 51,980 |
| **Model 3 (LinearSVC Composite)** | **Target A (`team_label`)** | **96.96%** | **96.82%** | **96.22%** | **65 / 2,135** | **Rs 36,725** |
| Model 1 (LinearSVC Pure TF-IDF) | Target B (`final_team`) | 84.87% | 84.56% | 82.53% | 323 / 2,135 | Rs 182,495 |
| **Model 2 (LinearSVC TF-IDF + Cat)** | **Target B (`final_team`)** | **85.20%** | **84.94%** | **82.53%** | **316 / 2,135** | **Rs 178,540** |
| Model 2B (LogReg TF-IDF + Cat) | Target B (`final_team`) | 84.12% | 84.09% | 82.53% | 339 / 2,135 | Rs 191,535 |
| Model 3 (LinearSVC Composite) | Target B (`final_team`) | 84.78% | 84.69% | 82.67% | 325 / 2,135 | Rs 183,625 |

*Waste calculation per Ops Policy §4: Rs 305 handling per transfer + Rs 260 extra customer contact per misroute = Rs 565 per misrouted request.*

---

## 4. Comprehensive Evaluation: Target A (Client Benchmark)

Target A represents agreement with the historical vendor bot's initial queue. The best performing model is **Model 3 (LinearSVC with Composite Tokens)**.

### A. Overall Performance
- **Validation Accuracy**: **96.96%** (2,070 / 2,135 correct)
- **Macro F1-Score**: **96.82%**
- **Weighted F1-Score**: **96.96%**

### B. Per-Class Precision, Recall, and F1-Score
```
─────────────────────────────────────────────────────────────────────────────
Canonical Queue           Precision    Recall    F1-Score    Validation Support
─────────────────────────────────────────────────────────────────────────────
Billing                      0.9731    0.9849      0.9790           331
Filters & Consumables        0.9450    0.9649      0.9549           285
Installs & Demo              0.9615    0.9783      0.9698           230
Product Advice               0.9685    0.9729      0.9707           221
Repairs                      0.9851    0.9627      0.9738           617
Returns & Replacement        0.9798    0.9643      0.9720           252
Warranty Claims              0.9505    0.9648      0.9576           199
─────────────────────────────────────────────────────────────────────────────
Macro Average                0.9662    0.9704      0.9682          2135
Weighted Average             0.9698    0.9696      0.9696          2135
─────────────────────────────────────────────────────────────────────────────
```

### C. Confusion Matrix (Target A — Model 3)
```
Actual \ Predicted       BIL   F&C   I&D   ADV   REP   R&R   WAR    Total
Billing                  326     2     1     0     0     1     1      331
Filters & Consumables      1   275     2     1     4     0     2      285
Installs & Demo            1     0   225     1     2     1     0      230
Product Advice             2     0     1   215     2     0     1      221
Repairs                    5    10     1     2   594     1     4      617
Returns & Replacement      0     2     2     2     1   243     2      252
Warranty Claims            0     2     2     1     0     2   192      199
Total Predicted          335   291   234   222   603   248   202     2135
```

### D. Rolling Monthly Accuracy (Target A — Model 3)
- **2026-01**: 97.01%
- **2026-02**: 96.22%
- **2026-03**: 97.30%
- **2026-04**: 97.42%
- **2026-05**: 96.59%
- **2026-06**: 96.45%
- **Minimum Monthly Accuracy**: **96.22%**

---

## 5. Comprehensive Evaluation: Target B (Operational Ground Truth)

Target B represents routing directly to the team that ultimately closes the ticket. The best performing model is **Model 2 (LinearSVC with TF-IDF + One-Hot Categoricals)**.

### A. Overall Performance
- **Validation Accuracy**: **85.20%** (1,819 / 2,135 correct)
- **Macro F1-Score**: **84.94%**
- **Weighted F1-Score**: **85.18%**

### B. Per-Class Precision, Recall, and F1-Score
```
─────────────────────────────────────────────────────────────────────────────
Canonical Queue           Precision    Recall    F1-Score    Validation Support
─────────────────────────────────────────────────────────────────────────────
Billing                      0.8996    0.8784      0.8889           255
Filters & Consumables        0.8398    0.8357      0.8378           207
Installs & Demo              0.8476    0.8318      0.8396           321
Product Advice               0.8494    0.8178      0.8333           269
Repairs                      0.8488    0.9016      0.8744           498
Returns & Replacement        0.8594    0.8252      0.8419           326
Warranty Claims              0.8220    0.8378      0.8298           259
─────────────────────────────────────────────────────────────────────────────
Macro Average                0.8524    0.8469      0.8494          2135
Weighted Average             0.8523    0.8520      0.8518          2135
─────────────────────────────────────────────────────────────────────────────
```

### C. Confusion Matrix (Target B — Model 2)
```
Actual \ Predicted       BIL   F&C   I&D   ADV   REP   R&R   WAR    Total
Billing                  224     6     7     4     8     3     3      255
Filters & Consumables      2   173     8     6     8     3     7      207
Installs & Demo            3     8   267     9    14    11     9      321
Product Advice             2     6     6   220    19     5    11      269
Repairs                    4     7    12     7   449    12     7      498
Returns & Replacement      7     4     6     7    23   269    10      326
Warranty Claims            7     2     9     6     8    10   217      259
Total Predicted          249   206   315   259   529   313   264     2135
```

### D. Rolling Monthly Accuracy (Target B — Model 2)
- **2026-01**: 83.78%
- **2026-02**: 83.28%
- **2026-03**: 87.13%
- **2026-04**: 87.37%
- **2026-05**: 85.15%
- **2026-06**: 82.53%
- **Minimum Monthly Accuracy**: **82.53%**

---

## 6. Cross-Target Analysis & Operational Trade-Off

A critical insight uncovered during our audit is the divergence between what the vendor bot did (`team_label`) and what human agents were forced to fix (`final_team`).

### Head-to-Head Comparison on Apr–Jun 2026 Validation Set (2,135 Requests)

| Metric | Target A Model (Bot Replica) | Target B Model (Resolution Specialist) | Operational Advantage of Target B |
| :--- | :---: | :---: | :--- |
| **Accuracy on True Resolution (`final_team`)** | **77.19%** | **85.20%** | **+8.01 percentage points** |
| **True Misrouted Tickets** | **487 tickets** | **316 tickets** | **171 fewer customer transfers** |
| **Estimated Misroute Reduction** | 487 misroutes | 316 misroutes | **171 fewer estimated misroutes** |
| **Quarterly Waste Cost (INR)** | Rs 275,155 | Rs 178,540 | **Rs 96,615 saved per quarter (validation holdout)** |
| **Annualized Waste Cost (INR)** | Rs 1,100,620 | Rs 714,160 | **Rs 386,460 estimated annual savings*** |

*Note: The Rs 386,460 annualized figure is strictly an estimate derived from the Apr–Jun 2026 holdout validation comparison assuming error distributions hold constant; it should not be treated as guaranteed production savings.

### Why Target A Has High Apparent Accuracy but Worse Operational Outcomes
Target A models easily achieve $\approx 97\%$ accuracy because the legacy vendor bot followed rigid keyword patterns (e.g. any message mentioning "paid" $\rightarrow$ `Billing`; any message mentioning "purifier" $\rightarrow$ `Consumables`). 

When deployed:
- The **Target A model** faithfully reproduces those same mistakes, misrouting 22.8% of tickets and burdening agents with 487 transfers every quarter.
- The **Target B model** routes according to true operational resolution, cutting misroutes to 14.8% and eliminating 171 transfers every quarter.

---

## 7. Concrete Case Studies from the Validation & Test Data

### Case 1: Breakdown with Mention of Payment
- **Request `SR510825`**: `"hi, installer did not turn up for cooktop i paid extra for this thanks"`
  - **Target A Prediction (Bot Replica)**: `Billing` (Tricked by the words *"i paid extra"*).
  - **Target B Prediction (Operational Target)**: `Installs & Demo` (Correctly recognizes installer missed appointment).
  - **Policy Reference**: Ops Policy §3 explicitly states: *"A customer mentioning that they have paid does not make it a billing request."*

### Case 2: Water Purifier Breakdown
- **Request `SR510824`**: `"sir water purifier leaking water from bottom purifier not working"`
  - **Target A Prediction (Bot Replica)**: `Filters & Consumables` (Tricked by *"purifier"*).
  - **Target B Prediction (Operational Target)**: `Repairs` (Correctly identifies leakage and non-functional state as technician breakdown).
  - **Policy Reference**: Teams.csv specifies: *Consumables: selling and fitting spares. Not faults.*

---

## 8. Recommendations & Next Steps

1. **For the Assessment Submission**:
   - Deliver **`predictions.csv`** generated by **Model 3 trained on Target A (`team_label`)**. This guarantees compliance with Ritu Deshpande's explicit requirement of matching historical routing labels above the 90% bar (achieving **96.96%** validation accuracy).
2. **For the Executive Business Memo**:
   - Present **`operational_predictions.csv`** generated by **Model 2 trained on Target B (`final_team`)**.
   - Show Farhan Sheikh that switching to the in-house Target B classifier creates the option to retire the existing **Rs 3.20 lakh/year** bot licence, subject to operational approval. On the Apr–Jun 2026 historical holdout, the operational model produced 171 fewer estimated misroutes than the benchmark model (yielding an estimated waste reduction of ~Rs 3.86 lakh/year, assuming the observed validation pattern repeats; not observed production savings). Local inference has no paid API dependency, and measured local inference cost is effectively zero per request; production hosting cost is not estimated because it was not measured/priced.
3. **Artifact Integrity**:
   - Both `predictions.csv` and `operational_predictions.csv` are strictly validated: exactly 2,178 rows, matching `request_id`s, no nulls, and only the 7 canonical team names.
