from src.autonomy.loop_runner import _failed_checks


def test_failed_checks_extracts_failed_and_missing():
    payload = {
        "gates": {
            "Gate A (Data)": {"failed_checks": ["etl_smoke"], "missing_checks": []},
            "Gate B (Model)": {"failed_checks": [], "missing_checks": ["model_smoke"]},
            "Gate C (Product)": {"failed_checks": [], "missing_checks": []},
        }
    }
    failed = _failed_checks(payload)
    assert failed["Gate A (Data)"] == ["etl_smoke"]
    assert failed["Gate B (Model)"] == ["model_smoke"]
    assert "Gate C (Product)" not in failed
