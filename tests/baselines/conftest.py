from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("baseline integration assets")
    group.addoption("--baseline-datastore", action="store", default=None)
    group.addoption("--chemprop-source", action="store", default=None)
    group.addoption("--grover-source", action="store", default=None)
    group.addoption("--grover-pretrained", action="store", default=None)
    group.addoption("--toxacol-source", action="store", default=None)


def _asset(request, option: str, environment: str, label: str) -> Path:
    value = request.config.getoption(option) or os.environ.get(environment)
    if not value:
        pytest.skip(f"{label} integration asset not supplied via {option} or {environment}")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"Configured {label} integration asset does not exist: {path}")
    return path


@pytest.fixture
def baseline_datastore(request):
    return _asset(request, "--baseline-datastore", "BASELINE_DATASTORE", "DataStore")


@pytest.fixture
def chemprop_source(request):
    return _asset(request, "--chemprop-source", "CHEMPROP_SOURCE", "Chemprop source")


@pytest.fixture
def grover_source(request):
    return _asset(request, "--grover-source", "GROVER_SOURCE", "GROVER source")


@pytest.fixture
def grover_pretrained(request):
    return _asset(request, "--grover-pretrained", "GROVER_PRETRAINED", "GROVER_base weights")


@pytest.fixture
def toxacol_source(request):
    return _asset(request, "--toxacol-source", "TOXACOL_SOURCE", "TOXACol source")
