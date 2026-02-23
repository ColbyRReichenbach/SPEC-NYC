# Workflow: Data Source Onboarding
1. Sample source data and profile schema.
2. Author mapping file under `src/datasources/mappings/`.
3. Run canonicalization + canonical contracts.
4. Run ETL with `--contract-profile canonical`.
5. Run release validator smoke mode.
