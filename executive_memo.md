# Executive Memo: Kestrel Service-Request Routing Engine

**To:** Ritu Deshpande (Head of D2C Operations), Farhan Sheikh (Finance Controller), Meenal Joshi (Service Desk Manager)  
**From:** Kabir Nanda (Account Lead) & Machine Learning Engineering Team  
**Date:** 27 September 2026  
**Subject:** Service-Request Routing Engine — Transition Decision, Performance & Operational Roadmap  

---

### 1. The Decision: Two-Model Deployment Architecture

We recommend immediately retiring the legacy third-party vendor routing bot and deploying an in-house, zero-cloud-cost routing solution with a clear **two-model architecture**:

1. **Benchmark Model (Model 3 — Composite Token TF-IDF + LinearSVC)**:
   - **Role**: Replicates historical routing bot queue assignments (`team_label`) to establish compliance with leadership's historical evaluation bar.
   - **Validation Result**: Achieves **96.96% accuracy** and **96.82% Macro F1** on the 3-month holdout (Apr–Jun 2026), decisively surpassing Ritu's 90% threshold across every single evaluation month (minimum 96.22%).
   - **Deliverable**: Powers the official benchmark assessment submission file (`predictions.csv`, 2,178 rows).

2. **Operational Model (Model 2 — Multi-Modal TF-IDF + One-Hot Categoricals + LinearSVC)**:
   - **Role**: Solves the actual operational crisis by predicting the resolving team (`final_team`) directly at intake, eliminating customer transfer churn.
   - **Validation Result**: Achieves **85.20% accuracy** and **84.94% Macro F1** on true ticket resolution, outperforming the bot replica's resolution accuracy (77.19%) by **+8.01 percentage points**.
   - **Deliverable**: Deployed live in the FastAPI microservice (`POST /predict`), the browser UI, and `operational_predictions.csv`.

---

### 2. Validated Numbers & Performance Comparison

All metrics are validated on the 3-month chronological holdout (Apr 1 – Jun 30, 2026, 2,135 tickets) and evaluated across 6 rolling monthly splits (Jan–Jun 2026). *Note: These represent rigorous validation on historical data, not hidden-test scores.*

| Metric | Target A Benchmark (Model 3) | Target B Operational (Model 2) | Discarded Hybrid Rule Filter |
| :--- | :---: | :---: | :---: |
| **Supervision Target** | Bot Queue (`team_label`) | Resolving Queue (`final_team`) | Resolving Queue (`final_team`) |
| **Holdout Validation Accuracy** | **96.96%** | **85.20%** | 85.11% (-0.09% drop) |
| **Macro F1-Score** | **96.82%** | **84.94%** | 84.87% |
| **Minimum Monthly Stability** | 96.22% | 82.53% | 82.53% |
| **Holdout Misroutes (out of 2,135)** | 65 (vs bot label) | **316 (vs true resolution)** | 318 (vs true resolution) |
| **Resolution Accuracy** | 77.19% (on final_team) | **85.20% (on final_team)** | 85.11% |

*Key Technical Learning*: A policy-guided hybrid rule filter was evaluated to test hardcoded business heuristics. It triggered only 4 times across 2,135 tickets, yielded 0 positive corrections, and caused 2 harmful misclassifications. It was decisively discarded in favor of pure Model 2.

---

### 3. Rupees & Operational Economics

Per Kestrel Operations Policy v4.1, each misrouted ticket incurs a **₹305 internal handling transfer cost** plus a **₹260 customer recontact cost** (total **₹565 per misroute**).

1. **Software Licence Option**:
   - Creates the option to retire the existing ₹3.2 lakh/year bot licence, subject to operational approval.
   - Local inference has no paid API dependency, and measured local inference cost is effectively zero per request. Production hosting cost is not estimated because it was not measured/priced.

2. **Estimated Operational Waste Reduction**:
   - On the Apr-Jun 2026 historical holdout, the operational model produced 171 fewer estimated misroutes than the benchmark model (316 vs 487).
   - Annualized savings (~₹3,86,460/year) are an estimate assuming the observed validation pattern repeats; they are not observed production savings.

3. **Total Economic Context**:
   - Provides leadership the option to retire the existing ₹3.2 lakh/year bot licence (subject to operational approval), coupled with an estimated ~₹3,86,460/year reduction in transfer waste based on the historical holdout comparison.

---

### 4. Operational Reality & Known Failure Modes

1. **Root Cause of Historical Transfer Churn**:
   - *Billing Misroutes*: The legacy bot sent requests to Billing whenever the text mentioned payment (e.g., *"paid for installation"*). Ops Policy §3 explicitly clarifies: *Billing is only when the payment itself is the problem.* Operational Model 2 successfully routes these to `Installs & Demo` or `Repairs`.
   - *Spares vs. Faults*: The legacy bot routed water purifier breakdown tickets to `Consumables`. Model 2 correctly routes leaks and power faults to `Repairs`.
2. **Ambiguous & Generic Requests**:
   - Extremely brief messages (e.g., *"please call me about purifier"*, *"not happy with service"*) contain insufficient intake signal for any statistical model to distinguish between a breakdown, an install visit, or a warranty inquiry. These represent the primary source of remaining operational misroutes.
3. **Safety Fallback**:
   - The service calculates a geometric decision margin for every request. Requests with **Low confidence (margin < 0.20)** are flagged with the mandatory notice: *"Routing suggestion based on historical resolution patterns; human review may be required for ambiguous requests."*

---

### 5. What to Do Next Week (Action Plan)

1. **Monday — Production Container & CRM Integration**:
   - Deploy `app.py` in a lightweight internal Docker container on existing CPU virtual infrastructure (zero new hardware acquisition).
   - Hook Kestrel CRM intake webhooks (IVR transcript post-processing, Web Chat, WhatsApp, Email) into `POST /predict`.
2. **Tuesday — Service Desk Agent Pilot & Training**:
   - Provide Meenal Joshi's team access to the lightweight browser intake screen and CRM integration.
   - Train agents on the 3 confidence tiers and how to utilize the human-readable reasons for triage.
3. **Wednesday — Low-Confidence Routing Queue**:
   - Configure CRM routing rules: High/Medium confidence tickets route automatically to the predicted queue; Low confidence tickets route to Meenal's senior triage queue for quick human confirmation.
4. **Thursday–Friday — Telemetry & Retraining Pipeline**:
   - Establish weekly logging comparing intake prediction against agent closure team (`final_team`) to track live operational transfer rates.
   - Schedule monthly automated retraining using `train_and_save.py` as new resolved CRM tickets accumulate.
