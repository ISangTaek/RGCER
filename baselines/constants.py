"""Immutable identities and ordering for the protocol-locked baselines."""

from __future__ import annotations

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS, TOXACUTE_TASKS as _TASKS

TOXACUTE_TASKS = tuple(_TASKS)
HUMAN3_TASKS = tuple(HUMAN_TARGET_TASKS)
SPLITS = ("train", "validation", "calibration", "test")
SMOKE_ALLOWED_SPLITS = ("train", "validation")
METHODS = ("rf", "afp", "dmpnn", "grover", "toxacol")

PROTOCOL_SHA256 = "a9be83025bd58ed3a3ce2e2b4e6759c14b2f632a1dcab812683737c7901b4265"
DATASTORE_BUILD_ID = "toxacute-v2-7b7bd62a6457"
DATASTORE_FINGERPRINT = "7b7bd62a6457501c8010f12b9bae851ec9c8b265539c93ddca583cf4b952be5c"
SPLIT_MANIFEST_HASH = "61a2e494469a4035447f237532272487fd897adb3fadae7879e0dc75d0b01085"

CHEMPROP_VERSION = "1.6.1"
CHEMPROP_COMMIT = "f3d1bff19a6e1b03d28e9cfabdf4c80dd8c67382"
GROVER_COMMIT = "40b6d97098e4508687912f3c05eca369fc2c6213"
GROVER_BASE_SHA256 = "47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de"
TOXACOL_COMMIT = "704edcb951ab397e31561bc130154d01e0103b3c"

SMOKE_TRAIN_PER_TASK = 32
SMOKE_VALIDATION_PER_TASK = 8
DEFAULT_SEED = 42

if len(TOXACUTE_TASKS) != 59 or HUMAN3_TASKS != (
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
):
    raise RuntimeError("Canonical ToxAcute task registry changed; protocol review is required")
