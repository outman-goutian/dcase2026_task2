# 提交结果复现说明

这份文档记录当前最终提交版本的完整复现方法，便于后续和其他模型做分数融合时保持一致。

## 1. 固定配置

- 模型：BEATs full finetuning
- checkpoint：
  `checkpoints_sweep_mixup_lr_train_scp_ep10/beats_full/beats_full_trainscp_lr1e-4_wd5e-3_mix035_ep10/model_epoch10.pth`
- 抽取层：`encoder_10`
- 通道：`ch1`，脚本参数是 `--channel 0`
- 特征归一化：`none`
- 异常分数：`KNN`
- `top_k`：`1`
- 数据增强：只对目标域做 `SMOTE`
- `sampling_ratio`：`0.25`
- `smote_k_neighbors`：`3`
- 随机种子：`42`

当前确认不要切换到：

- 平均模型
- `domain_local_density_sum`
- 标准化后再跑的那组参数

## 2. 先用 `test_dev.scp` 复现校验

先跑 dev，确认结果和之前一致：

```bash
docker exec torchds bash -lc 'cd /workspace/exp_beats && python scripts/create_final_submission.py \
  --config configs/beats_full.yaml \
  --checkpoint checkpoints_sweep_mixup_lr_train_scp_ep10/beats_full/beats_full_trainscp_lr1e-4_wd5e-3_mix035_ep10/model_epoch10.pth \
  --eval-scp /workspace/data/test_dev.scp \
  --eval-only \
  --layer 10 \
  --channel 0 \
  --score-method knn \
  --top-k 1 \
  --smote-sampling-ratio 0.25 \
  --smote-k-neighbors 3 \
  --batch-size 64 \
  --num-workers 4 \
  --output-dir final_submission_eval_check \
  --team-name eval_check_top1_smote'
```

期望聚合结果：

| AUC | pAUC | F1 | source AUC | target AUC |
|---:|---:|---:|---:|---:|
| 0.694199 | 0.572059 | 0.710661 | 0.712843 | 0.665936 |

校验输出：

`final_submission_eval_check/eval_check_top1_smote_20260613-075832/eval_metrics.json`

## 3. 生成最终提交

确认 dev 结果没偏后，再跑 final：

```bash
docker exec torchds bash -lc 'cd /workspace/exp_beats && python scripts/create_final_submission.py \
  --config configs/beats_full.yaml \
  --checkpoint checkpoints_sweep_mixup_lr_train_scp_ep10/beats_full/beats_full_trainscp_lr1e-4_wd5e-3_mix035_ep10/model_epoch10.pth \
  --test-scp /workspace/data/test_final.scp \
  --layer 10 \
  --channel 0 \
  --score-method knn \
  --top-k 1 \
  --smote-sampling-ratio 0.25 \
  --smote-k-neighbors 3 \
  --batch-size 64 \
  --num-workers 4 \
  --output-dir final_submission \
  --team-name team_exp_beats_top1_smote'
```

最终提交目录：

`final_submission/team_exp_beats_top1_smote_20260613-062359/`

最终提交压缩包：

`final_submission/team_exp_beats_top1_smote_20260613-062359.zip`

## 4. 输出文件格式

每个 final machine 都会生成两份文件：

```text
anomaly_score_<machine>_section_00_test.csv
decision_result_<machine>_section_00_test.csv
```

每个 CSV 都是两列、无表头：

```text
wav_file_name,score_or_decision
```

当前 final test 包含 5 个 machine：

- `BlowerDustCollector`
- `Sander`
- `SewingMachine`
- `ToothBrush`
- `ToyDrone`

每个 machine 200 条测试样本，所以每个 CSV 都是 200 行。

## 5. 后续做融合时怎么用

做 score fusion 时优先使用 `anomaly_score_*.csv`，因为它保留了每个 wav 的原始异常分数。

建议融合前先按 machine 做归一化，再做加权平均，避免不同模型分数尺度不一致。

## 6. 最终版本逐机结果

下面这组是最终提交版本在 `test_dev.scp` 上的校验结果，可直接用于文档里替换旧结果：

```yaml
ToyCarEmu:
  auc_source: 62.48
  auc_target: 93.60
  pauc: 60.00

ToyCar:
  auc_source: 65.20
  auc_target: 87.00
  pauc: 61.84

bearingEmu:
  auc_source: 67.48
  auc_target: 60.52
  pauc: 60.00

fan:
  auc_source: 76.32
  auc_target: 50.40
  pauc: 52.42

gearboxEmu:
  auc_source: 76.88
  auc_target: 63.76
  pauc: 53.21

sliderEmu:
  auc_source: 65.16
  auc_target: 61.96
  pauc: 49.16

valveEmu:
  auc_source: 94.52
  auc_target: 67.80
  pauc: 68.26
```

## 7. 参考文件

- 复现脚本：[create_final_submission.py](/kanas/asr/wangjunjie/plans/exp_beats/scripts/create_final_submission.py)
- 复现文档：[FINAL_SUBMISSION_REPRODUCE.md](/kanas/asr/wangjunjie/plans/exp_beats/FINAL_SUBMISSION_REPRODUCE.md)
- 最终提交 summary：`final_submission/team_exp_beats_top1_smote_20260613-062359/submission_summary.json`
