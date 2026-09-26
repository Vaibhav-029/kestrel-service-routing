# Model Card — Kestrel Service-Request Routing Classifier

## 1. Model Details

- **Model Name**: Kestrel-Router-LinearSVC
- **Model Version**: v2.0 (Post-Audit Release)
- **Model Type**: Multi-class Linear Support Vector Classifier with Word & Character N-Gram TF-IDF and Categorical Encodings.
- **Framework**: Python 3.13, `scikit-learn` v1.6+, `pandas`, `numpy`.
- **Primary Scored Architecture (Target A)**: `Model 3` — LinearSVC ($C=1.0$) trained on composite structured text representations (`channel_<c> product_<p> warranty_<w> <cleaned_text>`) with sublinear TF-IDF word (1–2) and char_wb (3–5) n-grams.
- **Primary Operational Architecture (Target B)**: `Model 2` — LinearSVC ($C=1.0$) trained on multi-modal ColumnTransformer combining text TF-IDF with One-Hot Encoded categorical attributes.
- **License**: Proprietary / Client Internal Use Only.

---

## 2. Intended Use

### Primary Intended Uses
- Automated real-time queue classification for incoming customer support requests across 4 channels (IVR speech-to-text transcripts, Web Chat, WhatsApp, and Email).
- Replacement of the legacy vendor routing bot at intake to eliminate per-ticket licensing fees.
- Routing directly to one of the **7 active canonical teams** defined in Kestrel Operations Policy v4.1:
  1. `Installs & Demo`
  2. `Repairs`
  3. `Filters & Consumables`
  4. `Billing`
  5. `Returns & Replacement`
  6. `Warranty Claims`
  7. `Product Advice`

### Out-of-Scope & Prohibited Uses
- Predicting ticket resolution duration, agent handling time, or escalations.
- Processing requests using post-routing operational variables (`transfers`, `resolved_at`, agent notes).
- Generating autonomous customer-facing conversational replies (the model routes queues; it does not draft messages).

---

## 3. Data & Preprocessing Pipeline

### A. Training & Evaluation Data
- **Training Corpus**: 10,822 historical tickets spanning **2025-04-01** to **2026-06-30** (15 months).
  - 4,320 migrated from Legacy Zoho Desk (prior to 1 Oct 2025).
  - 6,502 from Kestrel CRM (1 Oct 2025 onwards).
- **Test Corpus**: 2,178 unlabelled tickets spanning **2026-07-01** to **2026-09-30** (100% Kestrel CRM).
- **Input Features at Intake**:
  - `channel`: `ivr`, `chat`, `whatsapp`, `email`
  - `product_family`: 7 categories (`Water Purifier`, `Air Fryer`, `Mixer Grinder`, `Induction Cooktop`, `Room Heater`, `Ceiling Fan`, `Robot Vacuum`)
  - `warranty_status`: `in_warranty`, `out_of_warranty`, `shield`
  - `request_text`: Customer message or IVR transcript
  - `created_at_ist`: Intake timestamp in Indian Standard Time

### B. Text Sanitization & Normalization
1. **Mojibake & Character Restoration**:
   - Replaced Windows-1252 artifact `â€¦` with standard ellipsis `...`.
   - Stripped UTF-8 replacement character `\ufffd` (``) present in legacy Zoho migrations.
   - Applied NFKD unicode decomposition to normalize accented characters (e.g. `urgént` $\rightarrow$ `urgent`, `thé` $\rightarrow$ `the`, `sméll` $\rightarrow$ `smell`).
2. **Entity Masking / Canonical Tokenization**:
   - Customer Order IDs: Normalized regex pattern `\bko\d+\b` $\rightarrow$ `<ORDER_ID>`.
   - Registration Numbers: Normalized regex pattern `\bsr\d+\b` $\rightarrow$ `<REG_ID>`.
   - *Rationale*: Eliminates 537 out-of-vocabulary numerical tokens in the test set while preserving syntactic intent signals.

### C. Feature Engineering
- **Word TF-IDF**: N-gram range $(1, 2)$, minimum document frequency $\text{min\_df}=2$, sublinear term frequency scaling enabled ($1 + \log(\text{tf})$).
- **Character N-Grams**: Analyzer `char_wb` (word-boundary character n-grams) over range $(3, 5)$, $\text{min\_df}=5$, sublinear TF enabled. Captures spelling variations, colloquial abbreviations, and IVR transcription errors without ballooning feature space.
- **Categorical Feature Union**: Categorical metadata injected into feature space either as explicit one-hot vectors (Model 2) or prepended semantic tokens (Model 3).

---

## 4. Performance & Validation Metrics

### Performance Summary (Time-Based Split: Apr–Jun 2026 Holdout, 2,135 Requests)

| Metric | Target A (team_label / Bot Benchmark) | Target B (final_team / Operational Truth) |
| :--- | :---: | :---: |
| **Model Architecture** | Model 3 (Composite LinearSVC) | Model 2 (Multi-Modal LinearSVC) |
| **Validation Accuracy** | **96.96%** | **85.20%** |
| **Macro F1-Score** | **96.82%** | **84.94%** |
| **Minimum Monthly Accuracy (Jan–Jun 2026)** | **96.22%** | **82.53%** |
| **Quarterly Misroutes** | 65 / 2,135 (3.04%) | 316 / 2,135 (14.80%) |
| **Estimated Quarterly Misroute Waste** | Rs 36,725 | Rs 178,540 |

### Per-Class F1-Scores (Target A — Model 3)
- `Billing`: **0.9790**
- `Repairs`: **0.9738**
- `Returns & Replacement`: **0.9720**
- `Product Advice`: **0.9707**
- `Installs & Demo`: **0.9698**
- `Warranty Claims`: **0.9576**
- `Filters & Consumables`: **0.9549**

---

## 5. Operational & Systems Constraints

- **Cost & Inference Profile**:
  - The model trains in under 15 seconds on a standard 4-core CPU.
  - Memory consumption during inference is $< \text{30 MB}$.
  - Inference latency is **$< \text{20 ms per request}$** on standard local CPU.
  - Local inference has no paid API dependency, and measured local inference cost is effectively zero per request. Production hosting cost is not estimated because it was not measured/priced.
  - Creates the option to retire the existing Rs 3.2 lakh/year bot licence, subject to operational approval.

### Business Policy Compliance (Ops Policy v4.1)
- **Billing Integrity (§3)**: Trained on natural payment dispute semantics rather than superficial occurrences of "paid".
- **Spares vs Faults (§3)**: Clear division between spare part inquiries (`Filters & Consumables`) and breakdown symptoms (`Repairs`).
- **Team Renaming (§5)**: Strictly emits the 7 active post-15 Jan 2026 queue identifiers.

---

## 6. Artifact Verification

- **`predictions.csv`**: Target A submission file. Exactly 2,178 rows matching `test_unlabelled.csv` request IDs, containing only canonical names.
- **`operational_predictions.csv`**: Target B resolution predictions for operations and headcount planning.
- **Verification Status**: Passed all automated schema, row count, null-check, and label-validity assertions.

---

## 7. Live Service & Browser Screen

### Architecture

The live routing service uses the **operational model (Model 2 / Target B)** to predict the team that will actually resolve each request — not the historical benchmark label.

```
  Browser Screen (index.html)
        |
        v
  POST /predict  (FastAPI / uvicorn)
        |
        v
  Operational Model (TF-IDF + OneHot + LinearSVC)
        |
        v
  JSON Response: { predicted_team, confidence, reasons[] }
```

### Endpoint: `POST /predict`

**Input JSON:**
| Field | Type | Required |
|---|---|---|
| `request_id` | string | Yes |
| `created_at_ist` | string | Yes |
| `channel` | string (ivr/chat/whatsapp/email) | Yes |
| `product_family` | string | Yes |
| `warranty_status` | string (in_warranty/shield/out_of_warranty) | Yes |
| `request_text` | string | Yes |
| `source` | string (crm/legacy_zoho) | Yes |

**Output JSON:**
| Field | Type | Description |
|---|---|---|
| `predicted_team` | string | One of the 7 canonical team names |
| `confidence` | string | `High`, `Medium`, or `Low` (margin-based, not calibrated probability) |
| `reasons` | string[] | 2-3 human-readable explanations using only intake-time features |

### Confidence Levels

Confidence is derived from the LinearSVC `decision_function` margin (winning class score minus runner-up). These are **relative margin-based scores, not calibrated probabilities**.

| Level | Margin Threshold |
|---|---|
| High | > 0.6 |
| Medium | 0.2 - 0.6 |
| Low | < 0.2 |

### Testing Results

| Test | Result |
|---|---|
| Functional tests (5 cases) | 5/5 passed |
| Canonical label validation | All predictions in canonical set |
| JSON schema validation | All responses contain predicted_team, confidence, reasons[] |
| Reason count validation | All responses have 2-3 reasons |
| Latency (200 sequential requests) | Mean 33.5 ms, p50 33.9 ms, p95 47.3 ms |

### Usage

```bash
# 1. Train and save model artifacts
python train_and_save.py

# 2. Start the service
python -m uvicorn app:app --port 8000

# 3. Open browser screen
http://127.0.0.1:8000/

# 4. Run tests (while server is running)
python test_service.py
```

### Zero-Cost Design

- No paid API keys required
- No external cloud calls during inference
- CPU-only (no GPU required)
- Model artifacts are local joblib files (< 10 MB total)
- Total infrastructure: a single Python process

### Important Disclaimer

The prediction is a **suggested** routing from a machine-learning model trained on historical service-request data. It is **not** a guaranteed assignment. Agents must review the prediction and re-route if the suggestion does not match the actual issue. Confidence levels are relative margin-based scores, not calibrated probabilities.
