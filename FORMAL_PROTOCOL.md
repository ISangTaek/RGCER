# 正式实验协议（冻结版）

> 状态：**GO — 代码与数据冻结，自本文件提交起不得更改以下任何条目。**
> 冻结依据：正式门禁复审（checkpoint v6 + resume selection contract）全部通过。
> 违反本协议的运行结果不得进入论文主表。

---

## 1. 环境与代码

| 项 | 值 |
|---|---|
| Python | 3.10.19（conda 环境 `icl`，`requirements.txt` 固定版本） |
| PyTorch | 2.10.0+cu126 |
| algos 扩展 | cp310，`BOUNDED_EDGE_INPUT=1`，编译于本机 |
| 代码版本 | 本仓库 v6 checkpoint 契约提交（见 §7） |

## 2. 数据（不得重建）

| 项 | 值 |
|---|---|
| Manifest | 当前 approved Manifest V3（`split_algorithm=constrained_scaffold_v3`） |
| Manifest SHA256 | `61a2e494469a4035447f237532272487fd897adb3fadae7879e0dc75d0b01085` |
| split_seed | **42** |
| DataStore | 当前正式 V2 build：`toxacute-v2-7b7bd62a6457` |
| DataStore fingerprint | `7b7bd62a6457501c8010f12b9bae851ec9c8b265539c93ddca583cf4b952be5c` |
| raw CSV SHA256 | `47b406217dfbe916b0644a11ca071783791dd8a73f2f93685876560b0ab97eae` |
| feature schema | `atom_v2_bond_v1_pathavg_v1`（`max_path_distance=8`） |

每次正式运行前执行：

```bash
conda run -n icl python scripts/validate_datastore.py \
  --root data/toxacute_datastore_v2 --strict
```

必须 `status=OK` 才允许开跑。

## 3. 训练协议

| 项 | 值 |
|---|---|
| num_loader_workers | **0** |
| persistent_workers | 始终禁用（`disabled_for_strict_resume`） |
| task_sampling | **proportional** |
| selection_scope | **human3** |
| conformal_scope | **human3** |
| conformal_alpha | **0.10** |
| RGCER source policy | **animal56_only** |
| HPS warmup | **10 epochs**（warmup 期间 routing 关闭，之后启用） |
| checkpoint 契约 | strict **v6**：强制 `reproducibility` + `selection_state` |
| seed policy | **v2**（`reproducibility.py`） |

## 4. 随机种子（整篇论文固定，不得增删）

```text
训练 seeds：42, 43, 44, 45, 46
split_seed：42（已冻结于 Manifest / DataStore）
seed policy version：2
```

- 每个 (模型 × seed) 组合独立 fresh process 运行，禁止 mid-run 改代码。
- 断点续训只允许通过 v6 `*_last.pt` 的 epoch-boundary resume，且必须满足
  continuous == interrupted（best_epoch / best_val_score / 最终 best 模型哈希一致）。
- `initial_model_sha256` 必须写入每次运行的 `run_metadata.json`；
  同 seed 必须同哈希，异 seed 必须异哈希。

## 5. HPS 与 RGCER 公平比较契约

两者唯一结构差异是 RGCER 的 response-guided cross-endpoint routing；
以下各项**必须完全一致**，任何一项不一致的结果作废：

| 共享项 | 说明 |
|---|---|
| Manifest | 同一份 approved Manifest V3，同一 split |
| DataStore | 同一 V2 build（fingerprint 校验） |
| Graphormer backbone | 同一 encoder 结构与超参 |
| atom/bond features | 同一 `atom_v2_bond_v1_pathavg_v1` schema |
| quantile head | 同一 `TaskPredictionHead`（0.05/0.95，log-scale） |
| optimizer | 同 `adamw`、同 lr / weight_decay / 调度 |
| batch size | 同 `--bs` |
| task sampling | 同 `proportional`（同 epoch 任务批计划协议） |
| seed list | 同 §4 的 42–46 |
| task/epoch loader order | 同 `stable_seed(base_seed, task, "train", epoch)` 派生的 shuffle |
| human3 checkpoint selection | 同 `selection_scope=human3` 最佳验证选择 |
| human3 CQR protocol | 同 `conformal_scope=human3`、`alpha=0.10` |

## 6. 正式运行命令模板

```bash
cd ~/PROJECT/RGCER && conda activate icl

# HPS（基线）
python main.py --mode train --dataset toxacute --arch Graphormer \
  --toxacute_task_scope all59 --data_store_dir data/toxacute_datastore_v2 \
  --save_path artifacts/runs/formal_hps --experiment_tag formal \
  --bs 64 --epochs 100 --hps_warmup_epochs 10 \
  --task_sampling proportional --selection_scope human3 \
  --conformal_scope human3 --conformal_alpha 0.10 \
  --num_loader_workers 0 --gpu_id 0 --seed <SEED>

# RGCER（主方法）
python main.py --mode train --dataset toxacute --arch Graphormer_rgcer \
  --toxacute_task_scope all59 --data_store_dir data/toxacute_datastore_v2 \
  --save_path artifacts/runs/formal_rgcer --experiment_tag formal \
  --bs 64 --epochs 100 --hps_warmup_epochs 10 \
  --rgcer_source_policy animal56_only \
  --task_sampling proportional --selection_scope human3 \
  --conformal_scope human3 --conformal_alpha 0.10 \
  --num_loader_workers 0 --gpu_id 0 --seed <SEED>
```

## 7. 开跑前门禁（全部 PASS 才能开始正式实验）

- `python -m compileall -q .` → 0 错误
- `python -m pytest -q` → 0 failed / 0 errors
- HPS 1 epoch smoke：loss finite、human3 selection score finite、
  checkpoint v6 / seed policy v2 / selection_state 存在
- RGCER warmup1 + routed1 smoke：loss finite、NULL finite、
  joint source mass finite、joint + NULL ≈ 1、animal56_only 生效、
  source decoder grad = 0、router grad ≠ 0、checkpoint v6
- DataStore strict validate OK（§2）

**最近一次门禁执行结果**：2026-08-28 全部 PASS
（pytest 217 passed；HPS/RGCER smoke、resume 一致性、DataStore strict 均通过，
详见 git 提交记录与本文件同期的门禁日志）。
