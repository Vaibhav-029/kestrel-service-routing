"""
Kestrel Service-Request Routing — FastAPI Service

Loads the operational model (Model 2: TF-IDF + Categoricals + LinearSVC,
trained on canonical final_team) and serves routing predictions.

POST /predict  → operational routing suggestion
GET  /          → browser screen (static HTML)
"""

import os
import re
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ── Constants ────────────────────────────────────────────────────────
CANONICAL_TEAMS = [
    'Billing',
    'Filters & Consumables',
    'Installs & Demo',
    'Product Advice',
    'Repairs',
    'Returns & Replacement',
    'Warranty Claims',
]

TEAM_RESPONSIBILITIES = {
    'Installs & Demo': 'New-product installation, demo, and wall-mounting visits.',
    'Repairs': 'Product faults, breakdowns, error codes, noise, leaks — anything needing a technician.',
    'Filters & Consumables': 'Filters, candles, membranes, jars, brushes, blades, AMC kits — spares. Not faults.',
    'Billing': 'Invoices, GST, double charges, refunds of payments, EMI conversion, coupons. Only when the problem IS the payment.',
    'Returns & Replacement': 'Damaged, wrong, or incomplete deliveries; returns and exchanges within the return window.',
    'Warranty Claims': 'Warranty and Kestrel Shield registration, coverage questions, claim status.',
    'Product Advice': 'Pre- and post-purchase usage questions. No fault reported.',
}

from contextlib import asynccontextmanager

# Keyword groups for human-readable reason generation
TEAM_KEYWORDS = {
    'Repairs': [
        'not working', 'stopped working', 'not turning on', 'tripping', 'tripped',
        'leaking', 'leak', 'burnt smell', 'burnt', 'loud noise', 'noise',
        'vibrating', 'vibration', 'error code', 'gone blank', 'spark', 'sparks',
        'smoke', 'dead', 'broken', 'breakdown', 'fault', 'faulty', 'not heating',
        'not cooling', 'overheating', 'heating', 'cooling'
    ],
    'Billing': [
        'invoice', 'gst', 'double charge', 'charged twice', 'refund', 'emi',
        'coupon', 'payment deducted', 'deducted', 'bill', 'receipt', 'overcharged',
        'discount', 'payment', 'charged'
    ],
    'Returns & Replacement': [
        'damaged', 'broken', 'wrong model', 'wrong item', 'wrong product',
        'return', 'exchange', 'missing parts', 'incomplete', 'box was open',
        'dent', 'scratch', 'scratched', 'replacement'
    ],
    'Installs & Demo': [
        'installation', 'install', 'installer', 'wall mounting', 'wall mount',
        'demo', 'demonstration', 'delivered but not installed', 'slot', 'setup',
        'fitting', 'visit'
    ],
    'Filters & Consumables': [
        'filter', 'candle', 'membrane', 'brush roll', 'brush', 'amc',
        'blade set', 'blade', 'jar', 'cartridge', 'consumable', 'spare', 'spares'
    ],
    'Warranty Claims': [
        'warranty', 'shield', 'claim status', 'claim', 'coverage', 'certificate',
        'registration', 'registered', 'rma', 'extended warranty'
    ],
    'Product Advice': [
        'how to', 'best settings', 'settings', 'recipe', 'safe for kids',
        'power consumption', 'difference between', 'run on inverter', 'inverter',
        'specification', 'manual', 'guide', 'usage', 'features'
    ],
}

# Confidence thresholds (documented)
# LinearSVC decision_function returns a signed distance to each class hyperplane.
# We use the margin of the winning class over the runner-up as a relative confidence.
# These thresholds were chosen based on the distribution of margins in the
# Apr–Jun 2026 validation set; they are NOT calibrated probabilities.
# Margin distribution on validation set: High (>0.6): ~84.6%, Medium (0.2-0.6): ~6.7%, Low (<0.2): ~8.7%
CONFIDENCE_HIGH_THRESHOLD = 0.6   # top-class margin >= 0.6 → High
CONFIDENCE_LOW_THRESHOLD = 0.2    # top-class margin < 0.2 → Low
                                  # otherwise → Medium


# ── Text cleaning (same as training pipeline) ────────────────────────
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


# ── Reason generation ────────────────────────────────────────────────
def generate_reasons(
    predicted_team: str,
    cleaned_text: str,
    channel: str,
    product_family: str,
    warranty_status: str,
) -> list[str]:
    """Produce 2–3 concise, human-readable reasons using only intake-time info."""
    reasons = []
    text_lower = cleaned_text.lower()

    # 1. Match key indicator phrases specifically for the predicted team
    team_kws = TEAM_KEYWORDS.get(predicted_team, [])
    matched_kws = [kw for kw in team_kws if kw in text_lower]

    if matched_kws:
        quoted = ', '.join(f'"{kw}"' for kw in matched_kws[:3])
        reasons.append(f'Request text contains key indicator terms: {quoted}.')
    else:
        # Check any cross-team indicators or salient tokens
        all_hits = []
        for team, kws in TEAM_KEYWORDS.items():
            for kw in kws:
                if kw in text_lower and kw not in all_hits:
                    all_hits.append(kw)
        if all_hits:
            quoted = ', '.join(f'"{kw}"' for kw in all_hits[:2])
            reasons.append(f'Request mentions {quoted}, with overall language matching {predicted_team} resolution.')
        else:
            tokens = [w for w in re.findall(r'[a-zA-Z]{3,}', text_lower)
                      if w not in {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'please', 'reg', 'order', 'hello', 'dear', 'team', 'urgent'}]
            if tokens:
                sample_tokens = ', '.join(f'"{t}"' for t in tokens[:3])
                reasons.append(f'Intake text phrasing ({sample_tokens}) matches {predicted_team} patterns.')
            else:
                reasons.append(f'Intake message patterns and characteristics associate with {predicted_team}.')

    # 2. Product and warranty context
    warranty_label = warranty_status.replace('_', ' ')
    reasons.append(
        f'Product: {product_family} ({warranty_label}), received via {channel} channel.'
    )

    # 3. Policy responsibility from Ops Policy v4.1 / teams.csv
    responsibility = TEAM_RESPONSIBILITIES.get(predicted_team, '')
    if responsibility:
        reasons.append(f'{predicted_team} responsibility: {responsibility}')

    return reasons[:3]


# ── Load model at module level ───────────────────────────────────────
MODEL_DIR = Path(__file__).parent / 'models'

_operational_model = None
_operational_classes = None


def get_model():
    global _operational_model, _operational_classes
    if _operational_model is None:
        model_path = MODEL_DIR / 'operational_model.joblib'
        classes_path = MODEL_DIR / 'operational_classes.joblib'
        if not model_path.exists():
            raise RuntimeError(
                f'Model artifact not found at {model_path}. '
                'Run `python train_and_save.py` first.'
            )
        _operational_model = joblib.load(model_path)
        _operational_classes = joblib.load(classes_path)
    return _operational_model, _operational_classes


# ── Pydantic schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    request_id: str = Field(..., description='Service request number')
    created_at_ist: str = Field(..., description='Creation time (IST)')
    channel: str = Field(..., description='ivr | chat | whatsapp | email')
    product_family: str = Field(..., description='Product the request is about')
    warranty_status: str = Field(..., description='in_warranty | shield | out_of_warranty')
    request_text: str = Field(..., description="Customer's opening message or IVR transcript")
    source: str = Field('crm', description='crm | legacy_zoho')


class PredictResponse(BaseModel):
    predicted_team: str
    confidence: str
    reasons: list[str]


# ── FastAPI app with modern lifespan ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    print('Operational model loaded successfully.', flush=True)
    yield


app = FastAPI(
    title='Kestrel Service-Request Router',
    description='Predicts the operational routing queue for incoming service requests.',
    version='1.0.0',
    lifespan=lifespan,
)


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    model, classes = get_model()

    cleaned = clean_text(req.request_text)

    row = pd.DataFrame([{
        'cleaned_text': cleaned,
        'channel': req.channel,
        'product_family': req.product_family,
        'warranty_status': req.warranty_status,
    }])

    start = time.perf_counter()
    predicted_team = model.predict(row)[0]

    # Decision-function margin for confidence
    scores = model.decision_function(row)[0]  # shape (n_classes,)
    sorted_scores = np.sort(scores)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]

    elapsed_ms = (time.perf_counter() - start) * 1000

    if margin >= CONFIDENCE_HIGH_THRESHOLD:
        confidence = 'High'
    elif margin >= CONFIDENCE_LOW_THRESHOLD:
        confidence = 'Medium'
    else:
        confidence = 'Low'

    if predicted_team not in CANONICAL_TEAMS:
        raise HTTPException(status_code=500, detail=f'Invalid prediction: {predicted_team}')

    reasons = generate_reasons(
        predicted_team, cleaned, req.channel, req.product_family, req.warranty_status,
    )

    return PredictResponse(
        predicted_team=predicted_team,
        confidence=confidence,
        reasons=reasons,
    )


# ── Serve static HTML screen ────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / 'static'


@app.get('/', response_class=HTMLResponse)
def serve_screen():
    index_path = STATIC_DIR / 'index.html'
    if not index_path.exists():
        return HTMLResponse('<h1>Static screen not found</h1>', status_code=404)
    return HTMLResponse(index_path.read_text(encoding='utf-8'))
