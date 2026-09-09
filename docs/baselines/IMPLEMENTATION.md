# Five baseline implementation

The five Stage 3A2 baselines share one configuration-driven train/validation core. Formal runs read the complete frozen train and validation partitions; local smoke runs call the same core with an explicit `smoke_override` and the frozen Human3 sample selector. Calibration and test are rejected by the prediction command in this delivery.

- RF fits three independent `RandomForestRegressor` models using Morgan radius 2, 2048 bits, and chirality. The formal trial grid is exactly 54 combinations and `RandomState(42)` selects 50 without replacement.
- AttentiveFP uses the real PyG model and the existing nine atom/four bond categorical fields as concatenated one-hot features.
- D-MPNN uses the locked Chemprop 1.6.1 `MPNEncoder` plus Chemprop's native two-layer FFN (`300 -> 300 -> 3`).
- GROVER uses the locked official source, native molecular graph, complete 106-key encoder load, and an exact approved GROVER_base SHA-256 gate.
- TOXACol preserves 59 outputs, Avalon1024, the 26-column endpoint matrix, train-only task adjacency, and the locked upstream forward/initialization order. Formal training uses every train row with at least one of the 59 labels; selection remains Human3 validation macro-RMSE.

Every checkpoint stores and independently validates the model type, exact task order, feature schema, resolved config, scaler, data identity, seed, epoch, and best validation score. `baseline_predict.py` creates a new model, restores these objects from the checkpoint, loads an explicitly supplied DataStore validation split, and computes new predictions. It does not print a prior artifact.

TOXACol's upstream `epoch_size=100` is preserved in metadata. Inspection of the locked loader and training loop shows that this value only affects reported loader length; the actual iterator uses seeded shuffle, no replacement, complete dataset traversal, and `drop_last=False`. The implementation records both facts and does not reinterpret 100 as a sampling count.

Missing official source trees, a wrong GROVER SHA, an incomplete encoder, task/schema drift, or a missing scaler fail closed. Unit tests may skip external integration checks when paths are not supplied; `baseline_ready.py` never treats such skips as READY.

Formal AttentiveFP keeps graph records lazy. Its train and validation views share one read-only DataStore owner, so switching between partitions reuses the same lazily opened LMDB shard handles. `BaselineData.close()` and its context-manager exit release that owner once; a later load creates a fresh owner. This avoids duplicate in-process LMDB opens without materializing all graphs, changing LMDB flags, or changing any split.
