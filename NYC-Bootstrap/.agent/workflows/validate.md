---
description: Validate project health, test components, and verify deliverables
---

# Validation Workflow

**Role**: Quality assurance, testing, and verification of all project components.

---

## Quick Health Check

// turbo-all

Run this sequence to verify overall project health:

### 1. Docker Services
```bash
cd /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap && docker-compose ps
```
Expected: Both `db` and `app` services running

### 2. Database Connection
```bash
docker-compose exec db psql -U spec_user -d spec_nyc -c "SELECT 1;"
```
Expected: Returns `1`

### 3. Python Environment
```bash
cd /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap && python -c "import xgboost; import shap; print('Dependencies OK')"
```
Expected: "Dependencies OK"

### 4. Project Structure
```bash
ls -la /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/{data,models,logs,src}/
```
Expected: All directories exist

---

## Data Validation

### Record Count Check
```bash
docker-compose exec db psql -U spec_user -d spec_nyc -c "
SELECT 
    'sales' as table_name, COUNT(*) as count FROM sales
UNION ALL
SELECT 
    'properties', COUNT(*) FROM properties
UNION ALL
SELECT 
    'predictions', COUNT(*) FROM predictions;
"
```
Expected: sales ≥50,000, properties ≥50,000

### Data Quality Check
```bash
docker-compose exec db psql -U spec_user -d spec_nyc -c "
SELECT
    COUNT(*) as total,
    COUNT(CASE WHEN sale_price < 10000 THEN 1 END) as below_threshold,
    COUNT(CASE WHEN sqft IS NULL THEN 1 END) as missing_sqft,
    COUNT(CASE WHEN bbl IS NULL OR LENGTH(bbl) != 10 THEN 1 END) as invalid_bbl
FROM sales s
LEFT JOIN properties p ON s.bbl = p.bbl;
"
```
Expected: below_threshold=0, missing_sqft=0, invalid_bbl=0

### Feature Coverage
```bash
docker-compose exec db psql -U spec_user -d spec_nyc -c "
SELECT
    COUNT(*) as total,
    COUNT(distance_to_center_km) as has_distance,
    COUNT(h3_index) as has_h3,
    COUNT(h3_price_lag) as has_lag
FROM features;
"
```
Expected: All counts equal

---

## Model Validation

### Model Exists
```bash
ls -la /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/models/*.joblib
```
Expected: At least `xgb_v1.joblib`

### Model Loads Successfully
```python
import joblib
model = joblib.load('models/xgb_v1.joblib')
print(f"Model type: {type(model).__name__}")
print(f"Feature count: {model.n_features_in_}")
```

### Metrics File Check
```bash
cat /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/models/metrics_v1.json
```
Expected: JSON with ppe10, mdape, r2 keys

### SHAP Explanation Check
```bash
ls -la /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/models/shap_waterfall_sample.png
```
Expected: File exists, size >0

---

## API Validation (V3.0+)

### FastAPI Health
```bash
curl -s http://localhost:8000/health | jq .
```
Expected: `{"status": "healthy"}`

### Valuation Endpoint
```bash
curl -s -X POST http://localhost:8000/valuation \
  -H "Content-Type: application/json" \
  -d '{"bbl": "1000010001"}' | jq .
```
Expected: JSON with `point_estimate` key

### Response Time
```bash
for i in {1..5}; do
  time curl -s -X POST http://localhost:8000/valuation \
    -H "Content-Type: application/json" \
    -d '{"bbl": "1000010001"}' > /dev/null
done
```
Expected: Each request <500ms

---

## Security Validation

### Environment Variables
```bash
grep -v '^#' /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/.env | grep -E '^[A-Z]' | cut -d= -f1
```
Expected: OPENAI_API_KEY, DATABASE_URL, etc. (keys only, not values)

### No Secrets in Code
```bash
grep -r "sk-" /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap/src/ || echo "No API keys found in code"
```
Expected: "No API keys found in code"

### Audit Log Active
```bash
docker-compose exec db psql -U spec_user -d spec_nyc -c "SELECT COUNT(*) FROM ai_audit_log;"
```
Expected: Table exists (count may be 0 initially)

---

## Phase Deliverables Checklist

### V1.0 Deliverables

| Item | Check Command | Status |
|------|---------------|--------|
| Docker works | `docker-compose up -d` | [ ] |
| PostgreSQL has data | `SELECT COUNT(*) FROM sales` ≥50k | [ ] |
| Model achieves target | PPE10 ≥70% in metrics.json | [ ] |
| SHAP works | shap_waterfall_sample.png exists | [ ] |
| Map shows NYC | Manual dashboard check | [ ] |
| README complete | Has data sources, metrics | [ ] |
| Git tagged | `git tag -l 'v1.0'` | [ ] |

### V2.0 Deliverables

| Item | Check | Status |
|------|-------|--------|
| Confidence intervals | Quantile models exist | [ ] |
| Subway distance | Feature in model | [ ] |
| Backtesting | 2023 PPE10 ≥70% | [ ] |
| Comparable sales | Comps table works | [ ] |
| 5 boroughs | All borough data loaded | [ ] |
| Git tagged | `git tag -l 'v2.0'` | [ ] |

---

## Troubleshooting

### Docker Won't Start
```bash
# Check for port conflicts
lsof -i :5432
lsof -i :8501

# Reset containers
docker-compose down -v
docker-compose up -d
```

### Database Connection Failed
```bash
# Check container logs
docker-compose logs db

# Verify credentials
cat .env | grep DATABASE
```

### Model Training Failed
```bash
# Check training logs
tail -100 logs/training.log

# Verify data availability
python -c "
from src.database import Session
import pandas as pd
df = pd.read_sql('SELECT COUNT(*) FROM sales', Session().bind)
print(df)
"
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

---

## Automated Test Suite

### Run All Tests
```bash
cd /Users/colbyreichenbach/SPEC-NYC/NYC-Bootstrap
pytest tests/ -v --tb=short
```

### Run Specific Test Categories
```bash
# Data tests
pytest tests/test_etl.py -v

# Model tests  
pytest tests/test_model.py -v

# API tests
pytest tests/test_api.py -v

# Security tests
pytest tests/test_security.py -v
```

---

## Handoff

After validation:

1. If all checks pass:
   - Update `context.md` with validation timestamp
   - Proceed to next phase

2. If checks fail:
   - Document failures in `context.md` Blocking Issues
   - Route to appropriate workflow for fixes
   - Re-run validation after fixes

3. Route to: `/project-lead` with validation report
