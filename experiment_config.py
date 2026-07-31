"""Fixed Phase 0 experiment configuration."""

TOXACUTE_PHASE0_TASKS = (
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
)

TOXACUTE_PHASE0_RAW_CSV = "data/toxacute.csv"
TOXACUTE_PHASE0_PREPROCESSED_DIR = "artifacts/processed/toxacute_phase0"
TOXACUTE_PHASE0_RUN_DIR = "artifacts/runs/toxacute_phase0_3task"
TOXACUTE_PHASE0_CHECKPOINT_NAME = "toxacute_phase0_graphormer_prompt"
