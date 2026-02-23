# AI Explainability + Copilot Integration Spec

Status: Proposed  
Owner: ML Explainability + AI Product Engineering  
Date: 2026-02-23  
Related:
- `docs/PRODUCT_AVM_DASHBOARD_PRD.md`
- `docs/FRONTEND_ARCHITECTURE_RFC.md`
- `docs/AVM_BUSINESS_LOGIC.md`

## 1) Scope and Outcome

Add production-ready explainability and copilot capabilities to the web dashboard:
1. Property-level SHAP explanation (top positive/negative drivers).
2. Confidence and uncertainty display with caveats.
3. AI copilot answers for:
   - "why this estimate?"
   - "what changed recently?"
   - "what would improve confidence?"
4. Safety guardrails and deterministic fallback behavior.
5. Typed API contracts for frontend integration.

## 2) SHAP Integration Design (Property-Level)

## 2.1 Explainability objective

For each valuation response, return:
- Local feature contributions for that property.
- Ranked top positive and top negative drivers.
- Human-readable driver labels.
- Evidence references (model/run/version/explainer type).

## 2.2 Computation path

Inference service flow:
1. Build feature frame used for prediction.
2. Transform with model preprocessor.
3. Compute contributions:
   - Primary: XGBoost `pred_contribs` (stable for tree models).
   - Secondary: SHAP TreeExplainer.
   - Tertiary fallback: feature-level delta heuristic vs segment baseline.
4. Aggregate contributions to model features and map to UI-safe labels.
5. Return top-N positive and negative drivers (`N=5` default).

Notes:
- Explanations must use inference-time features only.
- Never expose raw internal one-hot feature IDs to end users.
- Include `explainer_type` for transparency.

## 2.3 Output fields

- `drivers_positive`: ordered list by descending positive contribution.
- `drivers_negative`: ordered list by descending negative contribution magnitude.
- `local_accuracy`: optional quality indicator for explainer availability.
- `explanation_status`: `ready | degraded | unavailable`.

## 3) Confidence + Uncertainty UX

## 3.1 Confidence model

Expose three uncertainty layers:
1. Prediction interval (`low`, `high`).
2. Confidence score (`0-1`).
3. Confidence band (`high | medium | low`).

## 3.2 Confidence derivation (v1)

Combine:
- Segment calibration error (historical MdAPE/PPE10 in route).
- Distance from training support (feature-space outlier score).
- Data completeness/quality score for provided inputs.

Band mapping:
- `high`: confidence_score >= 0.75
- `medium`: 0.45 <= score < 0.75
- `low`: score < 0.45

## 3.3 Required caveats in UI

Always show:
1. "Estimate is probabilistic, not an appraisal."
2. "Confidence depends on data quality and comparable coverage."
3. "Recent regime shifts can increase uncertainty."

For low confidence, show contextual caveat:
- "Sparse comparables or atypical feature values reduced confidence."

## 4) AI Copilot Capability

## 4.1 Product behavior

Copilot is retrieval-grounded and evidence-citing.  
It does not invent metrics or policy decisions.

Supported intents:
1. `why_estimate`
   - Explain top drivers and route/model context.
2. `what_changed_recently`
   - Summarize recent monitoring, drift, and proposal changes.
3. `improve_confidence`
   - Suggest data/coverage improvements and operational next steps.

## 4.2 Context sources

Primary context bundle:
- Current valuation response (prediction + interval + drivers).
- Model artifacts (`metrics`, segment scorecard, run card).
- Monitoring artifacts (`drift_latest.json`, `performance_latest.json`).
- Arena artifacts (latest proposal + comparison).

Copilot response must include:
- `answer`
- `citations` (artifact paths)
- `confidence` (assistant confidence in answer)
- `limitations`

## 4.3 UX placement

- Single valuation page: right rail `Explain + Ask` panel.
- Governance page: `Ask Copilot` in proposal detail.
- Monitoring page: alert-focused copilot prompt shortcuts.

Starter prompts:
- "Why this estimate?"
- "What changed in the last 30 days?"
- "How can we increase confidence for this property?"

## 5) Safety, Guardrails, and Fallbacks

## 5.1 Guardrails

1. Grounding-only policy:
   - Respond only from provided artifacts and model outputs.
2. Scope policy:
   - No legal, lending, or compliance determinations.
3. PII policy:
   - Redact sensitive fields in prompts/logs.
4. Prompt safety:
   - Use secure prompt delimiting and injection protection patterns.
5. Cost/token controls:
   - Enforce request budget and model limits.

Implementation should reuse controls from `src/ai_security.py`:
- token budget validation
- secure prompt building
- retry/error handling
- audit logging

## 5.2 Fallback behavior

If copilot unavailable or unsafe:
- Return deterministic template response from structured data.
- Display `"AI copilot unavailable; showing evidence summary."`
- Keep explainability panel functional (non-LLM).

If SHAP unavailable:
- Set `explanation_status=degraded`
- Show baseline feature-importance + caveat.

If uncertainty model unavailable:
- Show interval from residual baseline.
- Force confidence band to `medium` with warning badge.

## 6) API Contracts and Payload Schemas

Base path: `/api/v1`  
Auth: JWT bearer token  
All responses include `request_id`.

## 6.1 Single valuation with explanation

`POST /valuations/single`

Response extension:
```json
{
  "valuation_id": "val_01H...",
  "predicted_price": 1285000,
  "prediction_interval": {
    "low": 1170000,
    "high": 1399000,
    "method": "quantile_residual_v1"
  },
  "confidence": {
    "score": 0.68,
    "band": "medium",
    "factors": {
      "segment_calibration": 0.71,
      "support_coverage": 0.62,
      "input_completeness": 0.74
    },
    "caveats": [
      "Estimate is probabilistic, not an appraisal."
    ]
  },
  "explanation": {
    "status": "ready",
    "explainer_type": "xgboost_pred_contribs",
    "drivers_positive": [
      {"feature": "gross_square_feet", "impact": 142000, "display": "Larger interior size"},
      {"feature": "h3_price_lag", "impact": 89000, "display": "Higher local market baseline"}
    ],
    "drivers_negative": [
      {"feature": "building_age", "impact": -47000, "display": "Older building profile"},
      {"feature": "distance_to_center_km", "impact": -26000, "display": "Farther from central demand zones"}
    ]
  },
  "model": {
    "alias": "champion",
    "run_id": "34e917e198af4e58adb2097b8d9ca229",
    "model_version": "1",
    "route": "SMALL_MULTI"
  },
  "evidence": {
    "run_card_path": "reports/arena/run_card_34e9....md",
    "metrics_path": "models/metrics_v1.json",
    "shap_summary_path": "reports/model/shap_summary_v1.png"
  },
  "request_id": "req_01H..."
}
```

## 6.2 Property explanation endpoint (optional split)

`POST /explanations/property`

Request:
```json
{
  "valuation_id": "val_01H...",
  "top_n": 5
}
```

Response:
```json
{
  "valuation_id": "val_01H...",
  "status": "ready",
  "explainer_type": "xgboost_pred_contribs",
  "drivers_positive": [],
  "drivers_negative": [],
  "request_id": "req_01H..."
}
```

## 6.3 Copilot ask endpoint

`POST /copilot/ask`

Request:
```json
{
  "question": "Why this estimate?",
  "intent": "why_estimate",
  "context": {
    "valuation_id": "val_01H...",
    "proposal_id": "57e6c66f5205",
    "window": "30d"
  }
}
```

Response:
```json
{
  "answer": "The estimate is primarily driven up by interior size and local market baseline, with downward pressure from building age.",
  "citations": [
    "models/metrics_v1.json",
    "reports/arena/run_card_34e917e198af4e58adb2097b8d9ca229.md"
  ],
  "assistant_confidence": 0.83,
  "limitations": [
    "No fresh external listing feed was included in this estimate."
  ],
  "safety": {
    "guardrail_mode": "strict_grounded",
    "fallback_used": false
  },
  "request_id": "req_01H..."
}
```

## 6.4 Copilot error/fallback contract

Standard fallback response:
```json
{
  "answer": "AI copilot unavailable; showing evidence summary from latest artifacts.",
  "citations": [
    "models/metrics_v1.json"
  ],
  "assistant_confidence": 0.4,
  "limitations": [
    "Generated from deterministic fallback template."
  ],
  "safety": {
    "guardrail_mode": "fallback_template",
    "fallback_used": true,
    "reason": "upstream_timeout"
  },
  "request_id": "req_01H..."
}
```

## 7) Frontend Integration Checklist

1. Single valuation page renders:
- top positive/negative drivers
- confidence band + interval + caveats
- copilot panel with preset prompts

2. Governance and monitoring pages include:
- copilot contextual ask entrypoint
- citations drawer linking artifacts

3. Error states:
- clear degraded/unavailable badges
- deterministic fallback copy

4. Telemetry:
- track copilot usage by intent
- capture fallback rate and latency

