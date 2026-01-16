---
description: Start the S.P.E.C. NYC project from scratch
---

# NYC Project Bootstrap

This workflow initializes the S.P.E.C.-NYC project. Run this in a fresh directory.

## Prerequisites
- Docker installed
- Python 3.9+
- Git configured

## Steps

1. Create repository structure
```bash
mkdir -p S.P.E.C-Valuation-NYC/{config,data/{raw,processed},docs,src,api,frontend,models,tests}
cd S.P.E.C-Valuation-NYC
git init
```

2. Create .gitignore
```
__pycache__/
*.pyc
.env
data/raw/
data/processed/
models/*.joblib
.DS_Store
node_modules/
.next/
```

3. Create requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.43.0
optuna>=3.4.0
mlflow>=2.8.0
streamlit>=1.28.0
plotly>=5.18.0
folium>=0.15.0
streamlit-folium>=0.15.0
h3>=3.7.0
pandera>=0.17.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
geopandas>=0.14.0
python-dotenv>=1.0.0
```

4. Create docker-compose.yml
```yaml
version: '3.8'
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: spec
      POSTGRES_PASSWORD: spec_password
      POSTGRES_DB: spec_nyc
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  app:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - db
    environment:
      DATABASE_URL: postgresql://spec:spec_password@db:5432/spec_nyc
    volumes:
      - .:/app

volumes:
  postgres_data:
```

5. Create .env.example
```
DATABASE_URL=postgresql://spec:spec_password@localhost:5432/spec_nyc
OPENAI_API_KEY=your_key_here
```

6. Download NYC data
- Go to: https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page
- Download Manhattan and Brooklyn rolling sales (Excel files)
- Go to: https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page
- Download PLUTO for Manhattan and Brooklyn
- Save to `data/raw/`

7. Start development
```bash
docker-compose up -d db
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

8. Reference the full implementation plan
See `docs/NYC_IMPLEMENTATION_PLAN.md` for detailed phase-by-phase instructions.
