# Presentation & Recording Script (<= 3 Minutes)

**Project:** Kestrel Home — Service-Request Routing Engine (Variant B)  
**Target Duration:** 2 minutes 45 seconds  

---

### [0:00 – 0:30] 1. The Problem & What Was Tried

- **The Challenge**: Kestrel Home receives thousands of support requests across IVR transcripts, Web Chat, WhatsApp, and Email. The legacy vendor bot cost ₹3.2 lakh/year and had a 24.9% misroute rate, causing 3,900+ transfers and huge customer frustration.
- **What Was Explored**: We explored multiple local, zero-cost machine learning approaches across TF-IDF word/character n-grams, categorical one-hot encodings, composite structured tokens, Logistic Regression, LinearSVC, and a policy-guided rule hybrid.
- **Zero Paid API**: The entire solution runs locally on standard CPU without external paid APIs, avoiding variable per-ticket costs.

---

### [0:30 – 1:05] 2. Why the Two-Model Architecture Was Chosen

- **The Core Conflict**: Operations leadership had two conflicting requirements:
  1. *Ritu Deshpande (Head of Operations)*: Replicate 18 months of historical routing labels with a 90%+ match bar.
  2. *Farhan Sheikh (Finance) & Meenal Joshi (Support Desk)*: Eliminate misroutes and transfers where the vendor bot failed (e.g. routing repairs to Billing just because the customer said "I paid").
- **The Solution**: An explicit **Two-Model Architecture**:
  - **Benchmark Model (Model 3)**: LinearSVC with Composite Tokens trained on historical intake labels (`team_label`). Achieves **96.96% accuracy** and **96.82% Macro F1** on the 3-month holdout (Apr–Jun 2026), decisively beating the 90% benchmark bar. This generates our official benchmark submission (`predictions.csv`).
  - **Operational Model (Model 2)**: LinearSVC with TF-IDF text features and One-Hot Categorical metadata trained on resolving teams (`final_team`). Achieves **85.20% accuracy** on true ticket resolution, beating the legacy bot replica by **+8.01 percentage points**. This powers our live routing API and `operational_predictions.csv`.

---

### [1:05 – 1:30] 3. What Did Not Work / Was Discarded (The Policy Hybrid)

- We evaluated a hybrid architecture combining statistical Model 2 with deterministic business rules from Kestrel Operations Policy v4.1 (specifically overriding Billing when payments were mentioned alongside repairs or installs).
- **Outcome**: Across 2,135 validation tickets, the hybrid triggered only 4 times, produced zero positive corrections, and caused 2 harmful overrides, dropping accuracy from 85.20% to 85.11%.
- **Decision**: The hybrid was decisively discarded. The pure statistical Model 2 was retained for operational routing.

---

### [1:30 – 2:15] 4. Live Demo: API & Browser UI

- **FastAPI Microservice**: Running locally on port 8000. Serves `POST /predict`.
- **Intake-Only Schema**: Uses strictly intake-time features (`request_id`, `channel`, `product_family`, `warranty_status`, `request_text`, `source`). No post-routing leakage.
- **Browser Screen Demo**:
  - Show the interface at `http://127.0.0.1:8000/`.
  - Click preset: *"Water Purifier leaking and not turning on. Error code E02."*
  - Click **Route Request**.
  - Show instantaneous prediction: **Repairs**, **High Confidence**, with 3 transparent reasons (identifying fault phrases, product context, and official policy responsibility).
  - Highlight the mandatory disclaimer: *"Routing suggestion based on historical resolution patterns; human review may be required for ambiguous requests."*

---

### [2:15 – 2:45] 5. Latency, Economics & Operational Limitations

- **Measured Latency**: Tested over sequential requests — average latency is **~30–40 ms**, well below Kestrel's 500 ms SLA.
- **Economics**:
  - Creates the option to retire the existing ₹3.2 lakh/year bot licence, subject to operational approval.
  - On the Apr-Jun 2026 historical holdout, the operational model produced 171 fewer estimated misroutes than the benchmark model. Annualized savings (~₹3,86,460/year) are an estimate assuming the observed validation pattern repeats; they are not observed production savings.
  - Local inference has no paid API dependency; measured local inference cost is effectively zero per request; production hosting cost is not estimated because it was not measured/priced.
- **Limitations & Next Steps**:
  - Ambiguous requests (e.g. *"please call me about purifier"*) represent the main remaining source of error and receive `Low` confidence scores for human desk review.
  - Ready for immediate production pilot next week.
