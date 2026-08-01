"""Fixed Phase 0 experiment configuration."""

TOXACUTE_PHASE0_TASKS = (
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
)

TOXACUTE_PHASE0_RAW_CSV = "data/toxacute.csv"
TOXACUTE_PHASE0_PREPROCESSED_DIR = r"D:\PROJECT\MTL\MultiTask Toxic Prediction Reborn\processed_graph_data"
TOXACUTE_PHASE0_RUN_DIR = "artifacts/runs/toxacute_phase0_3task"
TOXACUTE_PHASE0_CHECKPOINT_NAME = "toxacute_phase0_graphormer_prompt"
