from baselines.runner import run_smoke
from baselines.utils import read_json


def test_two_parameter_sets_reach_the_shared_training_core(baseline_datastore, tmp_path):
    first = run_smoke(
        "rf",
        baseline_datastore,
        tmp_path / "rf-eight",
        overrides={"n_estimators": 8},
    )
    second = run_smoke(
        "rf",
        baseline_datastore,
        tmp_path / "rf-twelve",
        overrides={"n_estimators": 12},
    )
    assert first["status"] == second["status"] == "PASS"
    first_config = read_json(tmp_path / "rf-eight" / "resolved_config.json")
    second_config = read_json(tmp_path / "rf-twelve" / "resolved_config.json")
    assert first_config["training"]["n_estimators"] == 8
    assert second_config["training"]["n_estimators"] == 12
    assert "three fitted forests each contain 8 estimators" in read_json(tmp_path / "rf-eight" / "model_audit.json")["training_update"]["evidence"]
    assert "three fitted forests each contain 12 estimators" in read_json(tmp_path / "rf-twelve" / "model_audit.json")["training_update"]["evidence"]
