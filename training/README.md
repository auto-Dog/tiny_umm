# 训练入口

基础阶段继续使用 `training.train_understanding`、`training.train_flow` 和
`training.train_connector`；序列基线使用 `training.train_base`。

共享 Query 的 FM / CE 训练与条件 IB 使用 `training.train_shared`。
从仓库根目录查看参数：

```powershell
python -m training.train_shared --help
```

已有同配置的 Stage C `connected.pt` 时，可启动共享 Query IB：

```powershell
python -m training.train_shared --config configs/default.yaml --steps 1600 --target-prior geometry --posterior-std 0.003 --edit-beta 0.0006 --run-dir artifacts/experiments/shared_ib_example
```

`--run-dir` 必须尚不存在。训练产物为 `shared.pt` 和 `history.json`，
推理与评估继续使用 `inference.py` 和 `evaluation.evaluate`。

## 已移除的历史入口

`training/train_bottleneck.py` 是早期高斯 IB / 条件 IB 后训练脚本，已从当前版本移除。
删除前检查了 Python 导入、脚本调用、打包配置及 notebook 单元，
没有其他运行代码依赖该入口；旧命令仅见于历史实验说明。

复现旧实验应在单独的历史 checkout 中使用分支 `codex/IB` 的提交
`5ffb211a04384f47dc685322d557d8a1270e873c`，避免将旧命令用于当前版本。
该提交保留了旧入口及当时的实现。

`models/bottleneck.py` 和旧 `posttrained.pt` 的加载支持仍被模型、checkpoint
加载器与测试依赖，继续保留。临时 notebook 修改、内核调试等辅助脚本统一放在
被 Git 忽略的 `tmp/`，不作为正式训练入口。
