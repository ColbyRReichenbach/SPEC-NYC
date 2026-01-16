---
description: Train and evaluate XGBoost models with SHAP explainability
---

# ML Engineer Workflow

**Role**: Deliver high-precision property valuations with quantified uncertainty and SHAP-based explainability.

---

## Prerequisites Check

// turbo
1. Verify data is available:
   ```bash
   docker-compose exec db psql -U spec_user -d spec_nyc -c "SELECT COUNT(*) FROM sales WHERE sale_price >= 10000;"
   ```
   Expected: ≥50,000 records

// turbo
2. Check feature matrix exists:
   ```bash
   ls -la /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/data/processed/
   ```

---

## Task: Model Training (Phase 1.7)

### Model Standards (Non-Negotiable)

| Standard | Requirement |
|----------|-------------|
| Framework | XGBoost 2.0+ |
| Optimization | Optuna, minimum 50 trials |
| Validation | 5-fold cross-validation |
| Target Metric | PPE10 ≥70%, MdAPE ≤8% |
| Explainability | SHAP waterfall charts |

### Required Features

| Feature | Source | Type |
|---------|--------|------|
| `sqft` | PLUTO | Numeric |
| `year_built` | PLUTO | Numeric |
| `units_total` | PLUTO | Numeric |
| `building_class` | Sales | Categorical |
| `borough` | Sales | Categorical |
| `distance_to_center_km` | Computed | Numeric |
| `h3_price_lag` | Computed | Numeric |

### Step 1: Create Model Module

Create `src/model.py`:

```python
import xgboost as xgb
import optuna
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from pathlib import Path
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = [
    'sqft', 'year_built', 'units_total', 
    'distance_to_center_km', 'h3_price_lag',
    'building_class_encoded', 'borough'
]
TARGET = 'sale_price'
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

def compute_ppe10(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of Predictions within ±10% of actual."""
    pct_error = np.abs(y_pred - y_true) / y_true
    return (pct_error <= 0.10).mean() * 100

def compute_mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Percentage Error."""
    return np.median(np.abs(y_pred - y_true) / y_true) * 100

def objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function for hyperparameter tuning."""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kfold)
    
    return compute_ppe10(y.values, y_pred)

def train_model(X: pd.DataFrame, y: pd.Series, n_trials: int = 50):
    """Train XGBoost with Optuna optimization."""
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best PPE10: {study.best_value:.2f}%")
    
    # Train final model with best params
    model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Save model
    model_path = MODEL_DIR / 'xgb_v1.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, study

def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute all evaluation metrics."""
    y_pred = model.predict(X)
    
    metrics = {
        'ppe10': compute_ppe10(y.values, y_pred),
        'mdape': compute_mdape(y.values, y_pred),
        'r2': model.score(X, y),
        'n_samples': len(y)
    }
    
    logger.info(f"PPE10: {metrics['ppe10']:.2f}%")
    logger.info(f"MdAPE: {metrics['mdape']:.2f}%")
    logger.info(f"R²: {metrics['r2']:.4f}")
    
    return metrics

def generate_shap_explanation(model, X: pd.DataFrame, sample_idx: int = 0):
    """Generate SHAP waterfall chart for a sample property."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Save waterfall plot
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    import matplotlib.pyplot as plt
    plt.savefig('models/shap_waterfall_sample.png', bbox_inches='tight', dpi=150)
    plt.close()
    logger.info("SHAP waterfall saved to models/shap_waterfall_sample.png")
    
    return shap_values
```

### Step 2: Prepare Feature Matrix

```python
from src.database import Session
import pandas as pd

# Load data
session = Session()
query = """
    SELECT s.*, p.sqft, p.year_built, p.units_total, p.lat, p.lon
    FROM sales s
    JOIN properties p ON s.bbl = p.bbl
    WHERE s.sale_price >= 10000
"""
df = pd.read_sql(query, session.bind)

# Encode categoricals
df['building_class_encoded'] = df['building_class'].str[0].astype('category').cat.codes
df['borough'] = df['borough'].astype('category').cat.codes

# Prepare X and y
X = df[FEATURES]
y = df[TARGET]
```

### Step 3: Run Training

```bash
cd /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap
python -c "
from src.model import train_model, evaluate_model, generate_shap_explanation
import pandas as pd

# Load your prepared data
# X, y = load_feature_matrix()  # Implement this

model, study = train_model(X, y, n_trials=50)
metrics = evaluate_model(model, X, y)
shap_values = generate_shap_explanation(model, X)

print(f'\\n=== Final Results ===')
print(f'PPE10: {metrics[\"ppe10\"]:.2f}% (Target: ≥70%)')
print(f'MdAPE: {metrics[\"mdape\"]:.2f}% (Target: ≤8%)')
print(f'R²: {metrics[\"r2\"]:.4f} (Target: ≥0.75)')
"
```

### Step 4: Verify Success Criteria

| Metric | Target | Achieved | Pass? |
|--------|--------|----------|-------|
| PPE10 | ≥70% | ___ | [ ] |
| MdAPE | ≤8% | ___ | [ ] |
| R² | ≥0.75 | ___ | [ ] |

### Step 5: Generate Performance Report

After training, automatically generate:

1. **Metrics Summary** (save to `models/metrics_v1.json`):
```json
{
    "model_version": "v1.0",
    "timestamp": "2026-01-16T15:00:00",
    "metrics": {
        "ppe10": 72.5,
        "mdape": 7.2,
        "r2": 0.78
    },
    "hyperparameters": {...},
    "n_samples": 52000
}
```

2. **SHAP Waterfall** (saved to `models/shap_waterfall_sample.png`)

3. **Training Log** (saved to `logs/training.log`)

---

## Task: Quantile Regression (Phase 2.1 - V2.0)

> **Gate**: Only proceed after V1.0 targets are met.

### Quantile Models

Train three separate models for confidence intervals:

```python
def train_quantile_models(X, y):
    """Train models for 10th, 50th, and 90th percentiles."""
    models = {}
    
    for q, name in [(0.10, 'lower'), (0.50, 'point'), (0.90, 'upper')]:
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            **best_params  # from Optuna
        )
        model.fit(X, y)
        models[name] = model
        joblib.dump(model, MODEL_DIR / f'xgb_q{int(q*100)}_v2.joblib')
    
    return models

def predict_with_uncertainty(models: dict, X: pd.DataFrame) -> dict:
    """Generate point estimate with 80% confidence interval."""
    return {
        'point_estimate': models['point'].predict(X),
        'confidence_80': {
            'lower': models['lower'].predict(X),
            'upper': models['upper'].predict(X)
        }
    }
```

---

## Task: Backtesting (Phase 2.3)

### Backtest Protocol

```python
def run_backtest(df: pd.DataFrame):
    """Train on 2018-2022, predict 2023."""
    train = df[df['sale_year'] <= 2022]
    test = df[df['sale_year'] == 2023]
    
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    backtest_ppe10 = compute_ppe10(y_test.values, y_pred)
    
    logger.info(f"Backtest PPE10 (2023): {backtest_ppe10:.2f}%")
    return backtest_ppe10
```

Target: ≥70% PPE10 on 2023 holdout

---

## Troubleshooting

### Model Not Converging
1. Check for outliers in `sale_price` (cap at 99th percentile)
2. Verify features have no NaN values
3. Try log-transforming the target

### PPE10 Below Target
1. Add more spatial features (`subway_distance_m`)
2. Increase Optuna trials to 100
3. Check for data leakage in `h3_price_lag`

### SHAP Errors
1. Ensure XGBoost version ≥2.0
2. Use `shap.TreeExplainer` (not KernelExplainer)
3. Limit explanation to tree-based models only

---

## Handoff

When model training is complete:

1. Update `context.md`:
   - Set Phase 1.7 to `complete`
   - Record final metrics in "Key Metrics" section

2. Commit artifacts:
   ```bash
   git add models/ logs/
   git commit -m "Add trained XGBoost v1.0 - PPE10: XX%"
   ```

3. If metrics pass, proceed to Dashboard (Phase 1.8)

4. Route to: `/project-lead` for next phase orchestration
