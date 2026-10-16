import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="session")
def c01_cases():
    return json.loads((Path(__file__).parent / "fixtures/c01_structures.json").read_text(encoding="utf-8"))["cases"]
