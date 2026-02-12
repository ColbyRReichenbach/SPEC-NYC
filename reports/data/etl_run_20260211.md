# ETL Run Report - 20260211

## Run Metadata
- Started (UTC): 2026-02-11T22:48:29.041084
- Input: `data/raw/annualized_sales_2019_2025.csv`
- Dry run: `False`

## Stage Summary
```text
             stage   rows  unique_properties latest_sale_date
       extract_raw 498666                NaN       2024-12-31
           cleaned 295917                NaN       2024-12-31
feature_engineered 295457           256430.0       2024-12-31
   loaded_postgres 295457           256430.0       2024-12-31
```

## Data Contract Results
- **post-clean**: PASS
- **post-feature-engineering**: PASS