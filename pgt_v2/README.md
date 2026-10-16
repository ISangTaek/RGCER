# PGT V2 实验接口

本目录实现 2026-10-16 修订方案的 C02/C03。真实毒性数据的训练须使用后续已发布的具体 run JSON；G01 只准备数据与检查前向，不调用优化器。主模型为 M6，不能根据外层 test 结果更换主模型。

## 运行环境

使用独立环境安装 `requirements-pgt-v2.txt`，保留已有兼容的 PyTorch/PyG 安装。已在 Python 3.10.19、PyTorch 2.10.0、PyG 2.8.0.post1、RDKit 2025.09.6 验证。Chemprop 1.6.1 使用 NumPy 1.26.4；不要降级其他现有环境。第三方版本及作者文件哈希见 `configs/pgt_v2/dependency_lock.json`。M1 从固定 commit 的原始 Chemprop 目录导入，拒绝加载路径错误或源文件被改动的版本。

仓库根目录必须能导入编译的 `algos`。保留 Windows `.pyd`；Linux/WSL 使用同源、匹配 Python/NumPy ABI 的 `.so`，必要时在当前环境执行 `python setup.py build_ext --inplace`。生成库、`.c` 与构建目录均不提交，不改变 `.pyx`。该库只用于图距离，本框架不使用旧单条最短路径的 edge-input 编码。

## 数据与模型

`GraphCache.read` 返回同一登记图与 Mol；RF、D-MPNN、Avalon 均直接使用该 Mol，不重解析显示 SMILES。缓存没有标签；训练视图将 `train.npz`、`validation.npz`、`test.npz` 和对应的纯输入文件分开，来源 SHA、原始行、任务顺序、化学身份及统计值均有内容身份。默认四套候选是 `configs/pgt_v2/C0.json` 至 `C3.json`。

模型工厂 `build_model` 支持 M0–M6、A1–A3。M2/M4/M5/M6/A1 使用相同的独立点预测头；M5/M6 只在共享查询与逐任务查询数量上有差别。M4 复用已有 Router。A1 使用 C01 元数据并固定残差系数 1。A2 为有符号、任务条件值投影；A3 是原子均值线性控制。ToxACoL 保留作者 15 个主体通道与网络计算路径，统一优化、标准化、划分等差异列在 `method_registry.json`。

## 可执行入口

所有命令从仓库根目录执行；`--threads` 是放在子命令之前的可选全局参数，默认 1。

```text
python -B -m pgt_v2 environment --output ENVIRONMENT.json
python -B -m pgt_v2 inventory --source SOURCE.csv --output INVENTORY.json
python -B -m pgt_v2 prepare --source SOURCE.csv --output NEW_PREPARATION_DIRECTORY
python -B -m pgt_v2 smoke --view VIEW_DIRECTORY --cache CACHE_DIRECTORY --chemprop-root CHEMPROP_ROOT --toxacol-root TOXACOL_ROOT --device cpu --output FORWARD.json
python -B -m pgt_v2 resolve --row RELEASED_ROW.json --view VIEW_DIRECTORY --output RESOLVED_RUN.json
python -B -m pgt_v2 train --run RESOLVED_RUN.json --view VIEW_DIRECTORY --cache CACHE_DIRECTORY --output NEW_RUN_DIRECTORY
python -B -m pgt_v2 select --runs RUN_C0 RUN_C1 RUN_C2 RUN_C3 --output SELECTION.json
python -B -m pgt_v2 predict --checkpoint RUN/best.pt --view VIEW_DIRECTORY --cache CACHE_DIRECTORY --output NEW_PREDICTION_DIRECTORY
python -B -m pgt_v2 score --predictions PREDICTION_DIRECTORY --view VIEW_DIRECTORY --expected-checkpoint-sha256 LOCKED_CHECKPOINT_SHA256 --output SCORE.json
python -B -m pgt_v2 explain --checkpoint RUN/best.pt --view VIEW_DIRECTORY --cache CACHE_DIRECTORY --reference FOLD/reference.json --record-id RECORD_ID --task TASK_NAME --output NEW_EXPLANATION_DIRECTORY
python -B -m pgt_v2 package --files EXPLICIT_FILE_MAP.json --output NEW_PACKAGE.zip
```

这是参数说明，不是已发布的训练执行卡。`resolve` 不会将 PLANNED 改为 READY。`SELECTED/INHERIT` 必须提供 `--selection`，解析为具体候选；A1–A3 继承同折 M6 配置，P5 继承本方法全标签配置。run 中必须包含确切 commit、fold、config、view/source/cache/FeatureSpec/partition SHA。DEV 只映射到 F0 的 train/validation。`train` 和 `select` 不加载 `test.npz`；`predict` 只读相应纯输入文件，封存预测和 checkpoint 身份后，`score` 独立打开真值。

## 训练与恢复

端点等权损失使用实际 batch 行数以及整个可见训练视图的端点计数，均匀采样至少有一个可见标签的行。梯度累积按各微批实际行数加权，缺失标签梯度为零。有效 batch 默认 64，微批默认 16。资源探针后须冻结微批选择；代码不捕捉 OOM 并临时跳样本、改 microbatch 或继续训练。

BatchNorm 的单行尾批合并到前一批，保留所有记录；M3 最小物理微批为 2，尾批合并可能得到 3。BatchNorm 统计与 Dropout 的随机采样会受物理微批影响，因此“损失加权的梯度等价”不意味着这些模型在任意微批划分下轨迹相同。恢复必须保留原微批大小及运行环境。

AdamW、恒定学习率、weight decay=1e-4、clip=5，默认最多 200 epoch，至少 40 epoch 后才允许按 patience=30 早停。最佳 epoch 可以早于 40；相同验证分数保留较早 epoch。候选验证 macro-RMSE 差不超过 1e-12 时选较小配置 ID。seed42 复用被选中的 best，不合并 train/validation 重拟合。

`best.pt` 与 `last.pt` 都有 SHA。`--resume` 从最后完成的 epoch 恢复模型、优化器、随机数、历史与早停状态，未完成 epoch 从头重放；身份不一致或 best 文件改变即停止。CPU 恢复不初始化 GPU 随机状态，CUDA 恢复只读取所选设备。仅加载本流程产生并经外部 SHA 确认的可信检查点。

## 解释与统计边界

A2/A3 在固定训练参照上中心化，原尺度包含训练均值、标准差和截距，双终点差异包含截距之差。中心化是等价的模型账目分解，不是因果机制证据。解释输出保存原子映射、对称类、参考集合身份和重建误差。

M2/M5/M6/A1 的 IG 对完整模型求导，固定拓扑与 graph token，分别使用零嵌入、训练参照原子均值嵌入。固定 64 个 Gauss–Legendre 梯度点；归一化完整性残差超过 1e-3 就标记 `unreliable`，不自动追加积分点。随机初始化的 LayerNorm 模型在零基线附近可能明显不满足该阈值，失败仍计入后续分母。`representation_intervention` 会重跑完整主干与读出。G06 的面板、合法结构编辑、匹配对照及调用预算须在阶段卡另行冻结。

`statistics.pooled_oof` 先在每个 seed 内合并所有外层折观察，再计算端点 RMSE、宏平均，最后平均 seed 指标。`paired_group_bootstrap` 固定任务分母，在每个外层折内按结构组配对重采样，所有终点/seed 共用抽样；默认 10,000 次尝试、seed=20261016，同时返回 95% 与 98.3333% 区间、全部不利端点与缺任务失效比例。失效样本不重抽，超过 5% 标记不稳定。后续 OOF 全矩阵封存和解释面板仍由相应阶段卡决定。

## 定向验证

```text
python -B -m pytest tests/pgt_v2 tests/test_features.py tests/test_graphormer_bias.py tests/test_graphormer_edge_bias.py tests/test_collator.py tests/test_collator_boundary.py -q --basetemp=NEW_UNIQUE_TEST_DIRECTORY --junitxml=RESULT.xml -o junit_family=xunit1
```

测试只用合成标签和已登记的结构 fixture，不读取真实毒性表或历史缓存。每个进程必须使用不同的新临时目录，避免仓库默认 pytest 目录冲突。资产、真实缓存、覆盖率、服务器 GPU 以及正式效果均需另行验证；测试通过不代表科学结论成立。
