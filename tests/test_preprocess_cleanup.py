from unittest.mock import patch

import pandas as pd

from preprocess_data import preprocess_and_save


def test_preprocess_removes_stale_task_files(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "smiles": ["CC", "CCC", "CCO", "CCN"],
            "task": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(raw_csv, index=False)
    task_dir = tmp_path / "processed" / "task"
    task_dir.mkdir(parents=True)
    stale_file = task_dir / "data_999.pt"
    stale_file.write_bytes(b"stale")

    with patch("preprocess_data.get_graph_data_from_smiles", return_value=object()):
        preprocess_and_save(
            raw_csv,
            "task",
            tmp_path / "processed",
            splitting="random",
            valid_size=0.25,
            calibration_size=0.25,
            test_size=0.25,
            seed=7,
        )

    assert not stale_file.exists()
    assert sorted(path.name for path in task_dir.glob("data_*.pt")) == [
        "data_0.pt",
        "data_1.pt",
        "data_2.pt",
        "data_3.pt",
    ]
