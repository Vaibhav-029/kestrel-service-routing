# Kestrel Home — Service-Request Routing System (Variant B)

An automated, local machine-learning service-request routing system designed for **Kestrel Home Customer Operations**.

This system replaces the legacy vendor bot with a high-accuracy, zero-cloud-cost routing engine that creates the option to retire the existing ₹3.2 lakh/year bot licence (subject to operational approval), prevents customer transfer churn, and respects Kestrel Operations Policy v4.1.

---

## Table of Contents

1. [Project Purpose & Architecture](#1-project-purpose--architecture)
2. [Directory Structure](#2-directory-structure)
3. [Benchmark vs. Operational Model Distinction](#3-benchmark-vs-operational-model-distinction)
4. [Expected Local Data Files & Privacy](#4-expected-local-data-files--privacy)
5. [Installation & Clean Setup](#5-installation--clean-setup)
6. [Trained Model Artifacts](#6-trained-model-artifacts)
7. [Running the Live Service & Browser UI](#7-running-the-live-service--browser-ui)
8. [API Specification (`POST /predict`)](#8-api-specification-post-predict)
9. [Confidence Score Methodology](#9-confidence-score-methodology)
10. [Validation Performance & Evidence](#10-validation-performance--evidence)
11. [Local Latency & Zero-Cost Profile](#11-local-latency--zero-cost-profile)
12. [How to Regenerate Predictions](#12-how-to-regenerate-predictions)
13. [Operational Limitations](#13-operational-limitations)
14. [Executive Memo & Video Presentation Notes](#14-executive-memo--video-presentation-notes)

---

## 1. Project Purpose & Architecture

Kestrel Home customer support receives service requests across four communication channels: IVR transcripts, Web Chat, WhatsApp, and Email. Previously, a third-party vendor bot routed requests into queues with high error rates (24.9% misroute rate), incurring ₹3.2 lakh/year in bot software licensing plus substantial operational transfer costs (₹305 transfer cost + ₹260 customer recontact fee = ₹565 per misroute).

This project implements a **two-model architecture**:
1. **Benchmark Model (Model 3)**: Replicates the historical vendor bot queue assignments (`team_label`) to meet leadership's benchmark requirement (90%+ historical match bar).
2. **Operational Routing Model (Model 2)**: Predicts the true operational destination (`final_team`), directing requests to the resolving queue on the first pass to minimize customer transfers.

```
                              Incoming Service Request
                   (ID, Channel, Product, Warranty, Text)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       [Benchmark Model 3]                    [Operational Model 2]
     Target: Historical team_label             Target: Resolving final_team
                 │                                       │
                 ▼                                       ▼
          predictions.csv                        POST /predict & UI
       (2,178 Benchmark Rows)                 (Live Queue Routing + Reasons)
```

---

## 2. Directory Structure

```
kestrel-service-routing/
├── app.py                         # FastAPI service (POST /predict, GET / UI screen)
├── static/
│   └── index.html                 # Lightweight interactive browser UI
├── models/
│   ├── benchmark_model.joblib     # Serialized Model 3 (Composite-Token LinearSVC on team_label)
│   ├── operational_model.joblib   # Serialized Model 2 (TF-IDF + Categoricals on final_team)
│   └── operational_classes.joblib # Serialized class label array for decision margin
├── train_and_save.py              # Script to train and persist both model artifacts
├── evaluate_models.py             # Validation benchmark & prediction generator script
├── hybrid_target_b_experiment.py  # Rejected policy-guided hybrid experiment script
├── inspect_target_b_errors.py     # Script analyzing error distributions & failure modes
├── test_service.py                # Functional (7 cases) & latency (200 requests) test script
├── verify_all.py                  # End-to-end verification suite
├── predictions.csv                # Official benchmark submission (Target A, 2,178 rows)
├── operational_predictions.csv    # Operational queue predictions (Target B, 2,178 rows)
├── validation_results.json        # Machine-readable validation metrics & matrices
├── validation_report.md           # Comprehensive technical validation & audit report
├── executive_memo.md              # 1-page executive memo for leadership (decision, rupees, plan)
├── recording_notes.md             # Timed script & presentation notes for <= 3-minute video
├── submission-form.md             # Standardized assessment submission form
├── model_card.md                  # Standardized ML model card
├── requirements.txt               # Minimal Python dependencies (no paid APIs)
├── .gitignore                     # Data privacy protection (ignores client CSVs/PDFs)
└── README.md                      # Complete system documentation
```

---

## 3. Benchmark vs. Operational Model Distinction

| Dimension | Benchmark Submission (`predictions.csv`) | Live Operational Service (`/predict` & UI) |
| :--- | :--- | :--- |
| **Model** | **Model 3**: Composite-Token TF-IDF + LinearSVC | **Model 2**: Multi-Modal TF-IDF + One-Hot Categoricals + LinearSVC |
| **Supervision Target** | **Target A (`team_label`)**: Canonical intake queue assigned historically by the vendor bot. | **Target B (`final_team`)**: Canonical team that actually resolved and closed the service ticket. |
| **Business Objective** | Verifies model capability against historical benchmark standards (Ritu Deshpande's 90%+ match bar). | Directly routes tickets to resolving teams, cutting misroutes and transfer fees (Farhan Sheikh's operational goal). |
| **Validation Accuracy** | **96.96%** (Apr–Jun 2026 Holdout) | **85.20%** (Apr–Jun 2026 Holdout) |
| **Macro F1-Score** | **96.82%** | **84.94%** |
| **Artifact / Output** | `predictions.csv` (2,178 rows) | FastAPI `POST /predict` & `operational_predictions.csv` |

> **Important Note on Validation Scores**: The reported validation metrics (96.96% and 85.20%) are measured on the 3-month holdout set (April 1 to June 30, 2026). They reflect rigorous chronological validation on past data and are **not** hidden-test set performance.

---

## 4. Expected Local Data Files & Privacy

### Expected Local Files
The following files are expected locally in the root workspace directory for training and evaluation:

- `train.csv`: 10,822 historical intake requests spanning 2025-04-01 to 2026-06-30.
- `resolution_log.csv`: 10,822 corresponding outcome logs (`first_team`, `final_team`, `transfers`, `resolved_at`).
- `test_unlabelled.csv`: 2,178 unlabelled intake requests spanning 2026-07-01 to 2026-09-30.
- `teams.csv`: Canonical mapping and scope of responsibilities for all seven customer teams.
- `ops-policy.pdf`: Kestrel Operations Policy v4.1.

### Canonical Seven Teams (Post-15 Jan 2026)
Historical labels contain legacy names that are normalized to canonical form:
1. `Installs & Demo` (formerly `Installations`)
2. `Repairs`
3. `Filters & Consumables` (formerly `Consumables`)
4. `Billing`
5. `Returns & Replacement`
6. `Warranty Claims`
7. `Product Advice`

### Client Data Privacy Requirement
> **CRITICAL**: The supplied Kestrel client data (`train.csv`, `test_unlabelled.csv`, `resolution_log.csv`, `ops-policy.pdf`, `email-thread.txt`, `README.txt`) contains confidential customer communication and operational metrics. **Do NOT publish, upload, or push these data files to any public repository.** A `.gitignore` file is configured in the repository root to prevent accidental commits of private datasets.

---

## 5. Installation & Clean Setup

The system runs entirely on standard Python 3.10+ without GPU or external cloud dependencies.

### Step 1: Clone / Navigate to Workspace
```bash
cd kestrel-service-routing
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Trained Model Artifacts

Pre-trained model artifacts are stored in `models/`:
- `models/benchmark_model.joblib`: Model 3 pipeline trained on full `train.csv` against canonical `team_label`.
- `models/operational_model.joblib`: Model 2 pipeline trained on full `train.csv` + `resolution_log.csv` against canonical `final_team`.
- `models/operational_classes.joblib`: Serialized list of class label order for decision-margin calculation.

Both models can be retrained from scratch at any time using:
```bash
python train_and_save.py
```

---

## 7. Running the Live Service & Browser UI

### Start the FastAPI Server
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Console output will confirm:
```
INFO:     Started server process
Operational model loaded successfully.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Access the Browser UI
Open any modern web browser and navigate to:
```
http://127.0.0.1:8000/
```
The browser interface provides:
- Quick one-click test preset chips for common customer issue patterns.
- An interactive request intake form.
- Live predictions displaying:
  1. Predicted operational resolving team
  2. Relative confidence badge (`High`, `Medium`, `Low`)
  3. 2–3 plain-English reasons based strictly on intake features
- The mandatory operational disclaimer:
  *"Routing suggestion based on historical resolution patterns; human review may be required for ambiguous requests."*

### Run Functional & Latency Tests
While the server is running, execute:
```bash
python test_service.py
```
This script tests 7 distinct functional cases across canonical teams and executes a 200-request sequential latency benchmark.

---

## 8. API Specification (`POST /predict`)

### Endpoint
`POST http://127.0.0.1:8000/predict`

### Request Headers
`Content-Type: application/json`

### Input JSON Schema
```json
{
  "request_id": "string",
  "created_at_ist": "string (YYYY-MM-DDTHH:MM or YYYY-MM-DD HH:MM)",
  "channel": "string (ivr | chat | whatsapp | email)",
  "product_family": "string (Air Fryer | Ceiling Fan | Induction Cooktop | Mixer Grinder | Robot Vacuum | Room Heater | Water Purifier)",
  "warranty_status": "string (in_warranty | shield | out_of_warranty)",
  "request_text": "string (customer message or transcript)",
  "source": "string (crm | legacy_zoho)"
}
```

### Response JSON Schema
```json
{
  "predicted_team": "string (one of the 7 canonical teams)",
  "confidence": "string (High | Medium | Low)",
  "reasons": [
    "string",
    "string",
    "string"
  ]
}
```

### Example Request
```json
{
  "request_id": "SR-BIL-102",
  "created_at_ist": "2026-06-15T11:00",
  "channel": "email",
  "product_family": "Air Fryer",
  "warranty_status": "in_warranty",
  "request_text": "I was charged twice on my credit card for order KO291823. Please verify the double deduction and initiate a refund for the extra charge.",
  "source": "crm"
}
```

### Example Response
```json
{
  "predicted_team": "Billing",
  "confidence": "High",
  "reasons": [
    "Request text contains key indicator terms: \"charged twice\", \"refund\", \"charged\".",
    "Product: Air Fryer (in warranty), received via email channel.",
    "Billing responsibility: Invoices, GST, double charges, refunds of payments, EMI conversion, coupons. Only when the problem IS the payment."
  ]
}
```

### Reason Generation Principles
Reasons are strictly derived from **intake-time information** and policy mandates:
1. Identified indicator terms in customer request text.
2. Product family and warranty status context.
3. Official team scope defined in Kestrel Operations Policy v4.1.

> **Privacy & Integrity**: The service never exposes post-routing fields (such as `final_team`, `transfers`, resolution time, or agent notes).

---

## 9. Confidence Score Methodology

Linear Support Vector Classifiers (`LinearSVC`) optimize decision hyperplanes and do not produce calibrated probabilities.

To provide operational agents with a clear, relative reliability indicator:
1. For each incoming request, the model computes the signed distance to each team hyperplane using `decision_function`.
2. The distances are sorted in descending order.
3. The **top-class margin** is computed:
   $$\text{Margin} = \text{Score}_{\text{rank 1}} - \text{Score}_{\text{rank 2}}$$
4. The margin is categorized into documented relative levels:
   - **High**: Margin $\ge 0.60$ (Represents ~84.6% of requests; clear separation)
   - **Medium**: $0.20 \le \text{Margin} < 0.60$ (Represents ~6.7% of requests; moderate separation)
   - **Low**: Margin $< 0.20$ (Represents ~8.7% of requests; boundary case requiring human review)

> **Important**: These labels represent relative geometric confidence margins, **not** calibrated statistical probabilities.

---

## 10. Validation Performance & Evidence

Evaluation was conducted using a strict 3-month chronological holdout split:
- **Training Set**: 2025-04-01 to 2026-03-31 (12 months, 8,687 tickets)
- **Validation Holdout**: 2026-04-01 to 2026-06-30 (3 months, 2,135 tickets)

### Model Comparison Summary

| Model Architecture | Target | Validation Accuracy | Macro F1-Score | Min. Monthly Accuracy (Jan–Jun 2026) | Holdout Misroutes |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Model 1 (LinearSVC Word+Char TF-IDF) | Target A (`team_label`) | 95.78% | 95.82% | 94.62% | 90 / 2,135 |
| Model 2 (LinearSVC TF-IDF + OneHot Cat) | Target A (`team_label`) | 96.63% | 96.44% | 96.19% | 72 / 2,135 |
| Model 2B (Logistic Regression TF-IDF + Cat) | Target A (`team_label`) | 95.69% | 95.51% | 95.37% | 92 / 2,135 |
| **Model 3 (Composite-Token LinearSVC)** | **Target A (`team_label`)** | **96.96%** | **96.82%** | **96.22%** | **65 / 2,135** |
| Model 1 (LinearSVC Word+Char TF-IDF) | Target B (`final_team`) | 84.87% | 84.56% | 82.53% | 323 / 2,135 |
| **Model 2 (LinearSVC TF-IDF + OneHot Cat)** | **Target B (`final_team`)** | **85.20%** | **84.94%** | **82.53%** | **316 / 2,135** |
| Model 2B (Logistic Regression TF-IDF + Cat) | Target B (`final_team`) | 84.12% | 84.09% | 82.53% | 339 / 2,135 |
| Model 3 (Composite-Token LinearSVC) | Target B (`final_team`) | 84.78% | 84.69% | 82.67% | 325 / 2,135 |

### Policy-Guided Hybrid Rule Experiment
A rule-based hybrid post-processing filter was evaluated against Target B to test whether business heuristics could correct edge cases:
- Baseline Model 2 Accuracy: **85.20%**
- Hybrid Filter Accuracy: **85.11%** ($\Delta = -0.09\%$)
- Result: Only 4 rule activations occurred, with no net positive corrections. Therefore, the pure statistical Model 2 was retained for production.

### Operational Cost Impact Analysis
Per Ops Policy §4, each customer transfer incurs ₹305 in handling fees and ₹260 in customer recontact overhead (₹565 per misroute).

On the Apr–Jun 2026 holdout set (2,135 tickets):
- Target A Bot Replica misroutes: 487 tickets (₹2,75,155 holdout waste)
- Target B Operational Model misroutes: 316 tickets (₹1,78,540 holdout waste)
- Holdout Difference: On the Apr-Jun 2026 historical holdout, the operational model produced 171 fewer estimated misroutes than the benchmark model.
- **Annualized Waste Reduction Estimate**: **₹3,86,460**

> **Reporting Note**: On the Apr-Jun 2026 historical holdout, the operational model produced 171 fewer estimated misroutes than the benchmark model. Annualized savings (~₹3,86,460/year) are an estimate assuming the observed validation pattern repeats; they are not observed production savings.

---

## 11. Local Latency & Zero-Cost Profile

### Measured Local Latency
Measured across 200 sequential requests to `POST /predict` on standard local CPU:
- **Mean Latency**: **41.4 ms**
- **Median (p50)**: **43.3 ms**
- **95th Percentile (p95)**: **54.1 ms**
- **99th Percentile (p99)**: **99.2 ms**
- **Min / Max**: **27.0 ms / 124.3 ms**

This easily outperforms Kestrel's operational latency target of $< 500\text{ ms}$.

### Zero-Cost Profile
- **No External API Dependencies**: Zero reliance on paid cloud LLMs (OpenAI, Anthropic, Gemini).
- **Zero Per-Request Charges**: Completely immune to bill shocks as ticket volume grows.
- **Resource Footprint**: The entire service consumes $< 120\text{ MB}$ of RAM and executes lightweight local CPU inference. Local inference has no paid API dependency, and measured local inference cost is effectively zero per request. Production hosting cost is not estimated because it was not measured/priced.

---

## 12. How to Regenerate Predictions

To regenerate all project prediction files and verify their integrity from clean data:

```bash
# 1. Train and save model artifacts to models/
python train_and_save.py

# 2. Run full evaluation and generate both submission files:
#    - predictions.csv (Target A - Benchmark submission)
#    - operational_predictions.csv (Target B - Operational destination)
python evaluate_models.py
```

### Verification Checks
Both generated CSV files conform to all submission constraints:
- Exactly 2,178 rows matching `test_unlabelled.csv`.
- Columns: `request_id`, `team`.
- Exactly in test row order.
- Predicted teams are strictly from the 7 canonical names.
- Zero null values.

---

## 13. Operational Limitations

1. **Generic Customer Requests**: Short or vague requests (e.g. *"please call me about purifier"*, *"not happy with service"*) contain insufficient intake signal to distinguish between a product breakdown (`Repairs`), an installation inquiry (`Installs & Demo`), or a warranty dispute (`Warranty Claims`). Analysis indicates that generic requests are a major observed source of operational-model errors.
2. **Ambiguous Mentions of Payment**: While the operational model accurately identifies breakdown issues where the customer incidentally mentions paying, ambiguous messages without fault details may require agent clarification.
3. **New Unseen Product Lines**: The model encodes the 7 known Kestrel product families. New product introductions should be added to the categorical vocabulary during scheduled retraining cycles.
4. **Human Review for Low Confidence**: Requests flagged with `Low` confidence (margin $< 0.20$) should be flagged for intake agent review in the CRM queue rather than automated assignment.

---

## 14. Executive Memo & Video Presentation Notes

- **Executive Memo (`executive_memo.md`)**: A one-page executive memo for Ritu Deshpande, Farhan Sheikh, and Meenal Joshi covering the deployment decision, validated numbers, rupees and economics, operational reality, and immediate next-week action plan.
- **Recording Script & Notes (`recording_notes.md`)**: A structured script supporting a <= 3-minute video walkthrough detailing what was tried, why the two-model architecture was selected, the rejected hybrid rule filter, live API/UI demonstration, measured latency, and operational constraints.
