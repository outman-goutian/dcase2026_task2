# EAT 提交结果复现说明

本文档说明如何复现本次提交的 EAT 结果。当前提交对应的主配置是：

- checkpoint: `checkpoints_lr_sweep_train_scp/eat_base_full/eat_base_full_best_trainscp_lr4e-5_ep8/model_epoch3.pth`
- layers: `8,10,12`
- layer ensemble: `score_mean`
- score method: `knn_domain_local_density`
- local density scale: `sum`
- distance: `l2`
- source K: `16`
- target K: `9`

最终提交输出目录示例：

- `final_submission_eat/team_exp_diffpt_eat_layers8_10_12_score_mean_20260613-074542/`

## 1. 数据准备

确保下面两个文件存在：

- `/kanas/asr/wangjunjie/plans/data/train.scp`
- `/kanas/asr/wangjunjie/plans/data/test_dev.scp`
- `/kanas/asr/wangjunjie/plans/data/test_final.scp`

其中 `train.scp` 用于建立 memory bank，`test_dev.scp` 用于复现验证结果，`test_final.scp` 用于生成最终提交流程文件。

## 2. Docker 环境

使用镜像 `torchds:updated`。脚本默认占用 GPU 2。

```bash
cd /kanas/asr/wangjunjie/plans/exp_diffpt
```

## 3. 先做 dev 一致性检查

这一步会用和最终提交完全一致的配置先跑 `test_dev.scp`，确认结果和之前验证实验一致。

```bash
DEV_SCP=/workspace/data/test_dev.scp SKIP_FINAL=1 bash scripts/run_eat_final_submission.sh
```

如果配置一致，应该看到聚合指标接近并对齐到下面数值：

- AUC: `0.6916`
- pAUC: `0.5696`
- F1: `0.7185`
- source_AUC: `0.6866`
- source_pAUC: `0.6099`
- target_AUC: `0.6704`
- target_pAUC: `0.5951`

这组数值和 `reports/eat_top1_layer_ensemble/summary_vs_layer12.tsv` 里的 `layers8_10_12_score_mean` 一致。

## 4. 生成 final submission

dev 检查通过后，直接跑 final：

```bash
bash scripts/run_eat_final_submission.sh
```

脚本会自动：

1. 加载同一个 EAT checkpoint
2. 对 `test_final.scp` 按机器拆分
3. 提取 layers `8/10/12` 的 embedding
4. 做 `score_mean` ensemble
5. 用 `knn_domain_local_density` 打分
6. 生成每台机器的 `anomaly_score_*.csv` 和 `decision_result_*.csv`
7. 打包成 zip

## 5. 输出文件

最终输出会写到类似下面的目录：

- `final_submission_eat/team_exp_diffpt_eat_layers8_10_12_score_mean_<timestamp>/`

里面包含：

- `anomaly_score_BlowerDustCollector_section_00_test.csv`
- `anomaly_score_Sander_section_00_test.csv`
- `anomaly_score_SewingMachine_section_00_test.csv`
- `anomaly_score_ToothBrush_section_00_test.csv`
- `anomaly_score_ToyDrone_section_00_test.csv`
- `decision_result_*.csv`
- `submission_summary.json`

同时会生成对应的 zip 文件。

## 6. 相关脚本

- [scripts/create_final_submission_eat.py](/kanas/asr/wangjunjie/plans/exp_diffpt/scripts/create_final_submission_eat.py)
- [scripts/run_eat_final_submission.sh](/kanas/asr/wangjunjie/plans/exp_diffpt/scripts/run_eat_final_submission.sh)
- [configs/layer_sweep/eat_top1_ensemble/layers8_10_12_score_mean.yaml](/kanas/asr/wangjunjie/plans/exp_diffpt/configs/layer_sweep/eat_top1_ensemble/layers8_10_12_score_mean.yaml)
- [reports/eat_top1_layer_ensemble/summary_vs_layer12.tsv](/kanas/asr/wangjunjie/plans/exp_diffpt/reports/eat_top1_layer_ensemble/summary_vs_layer12.tsv)

## 7. 备注

- 这份提交不是 `mean` 版 local density，而是 `sum` 版。
- 如果你只想验证配置是否一致，不想重新生成 final 文件，可以只跑 dev：

```bash
DEV_SCP=/workspace/data/test_dev.scp SKIP_FINAL=1 bash scripts/run_eat_final_submission.sh
```
