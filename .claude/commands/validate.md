Run a full validation check on the project.

Use the @validator agent to:

1. Check Docker services are running
2. Verify database connection and record counts
3. Validate data quality (no bad BBLs, no low-price sales)
4. Check model artifacts exist and metrics meet targets
5. Verify no secrets in code
6. Report overall project health

Return a summary with ✓ for passing checks and ✗ for failures.
