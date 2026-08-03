"""Named experiment defaults.

The processed directory remains configurable because the current workspace
already contains the three-task smoke data there.  The CLI's formal task
scope, however, defaults to all 59 endpoints.
"""

TOXACUTE_RAW_CSV = "data/toxacute.csv"
TOXACUTE_PREPROCESSED_DIR = r"D:\PROJECT\MTL\MultiTask Toxic Prediction Reborn\processed_graph_data"
TOXACUTE_SMOKE_RUN_DIR = "artifacts/runs/toxacute_smoke_human3"
TOXACUTE_MAIN_RUN_DIR = "artifacts/runs/toxacute_all59_rgcer"

# Compatibility aliases used by preprocessing and older experiment scripts.
TOXACUTE_PHASE0_TASKS = (
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
)
TOXACUTE_PHASE0_RAW_CSV = TOXACUTE_RAW_CSV
TOXACUTE_PHASE0_PREPROCESSED_DIR = TOXACUTE_PREPROCESSED_DIR
TOXACUTE_PHASE0_RUN_DIR = TOXACUTE_SMOKE_RUN_DIR
TOXACUTE_PHASE0_CHECKPOINT_NAME = "toxacute_phase0_graphormer_prompt"
