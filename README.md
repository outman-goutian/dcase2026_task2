# DCASE 2026 Task 2 Final Submission

This repository contains the cleaned final code for the selected BEATs and EAT
systems, plus the score-level fusion used for the best validation result.

## Best Results

All numbers below are from `test_dev.scp` validation. The official selection
score is `hmean(AUC, pAUC)`.

| System | AUC | pAUC | F1 | source AUC | target AUC | official |
|---|---:|---:|---:|---:|---:|---:|
| BEATs best | 0.694199 | 0.572059 | 0.710661 | 0.712843 | 0.665936 | 0.627238 |
| EAT best | 0.691600 | 0.569600 | 0.718500 | 0.686600 | 0.670400 | 0.624699 |
| Fusion best | 0.714181 | 0.586091 | 0.725602 | 0.726954 | 0.687884 | 0.643827 |

Fusion best uses per-machine rank normalization and weights:

```text
beats=0.55,diffpt=0.45
```

## Repository Layout

- `configs/config_beats.yaml`: BEATs training and validation config
- `configs/config_eat.yaml`: EAT training and validation config
- `checkpoints/beats/checkpoint.pth`: final BEATs checkpoint
- `checkpoints/eat/checkpoint.pth`: final EAT checkpoint
- `beats/BEATs_iter3_plus_AS2M.pt`: BEATs pretrained checkpoint
- `eat_base/model_noft.safetensors`: EAT pretrained checkpoint
- `scripts/train_model.sh`: train BEATs or EAT
- `scripts/validate.sh`: run dev validation with final settings
- `scripts/test.sh`: generate final CSVs and zip
- `scripts/fuse_submissions.py`: fuse generated BEATs/EAT submissions
- `docs/`: original reproduce notes

## Environment

The tested runtime is the existing Docker image `torchds:updated`:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD":/workspace/dcase2026_task2 \
  -v /kanas/asr/wangjunjie/plans/data:/workspace/data \
  -w /workspace/dcase2026_task2 \
  torchds:updated bash
```

Alternatively build this repository image:

```bash
bash docker/build.sh
bash docker/run.sh bash
```

For a non-Docker environment:

```bash
pip install -r requirements.txt
```

Use Docker for GPU runs if the host PyTorch/CUDA build does not match the driver.

## Data

Inside Docker, the code expects:

```text
/workspace/data/train.scp
/workspace/data/test_dev.scp
/workspace/data/test_final.scp
```

Check existing SCP files or regenerate them from extracted DCASE data:

```bash
python3 scripts/prepare_data.py --data-root /workspace/data --output-dir /workspace/data
```

If the audio folders are missing, the script exits with a message asking you to
download and unzip the DCASE data first.

## Train

Train EAT:

```bash
bash scripts/train_model.sh eat
```

Train BEATs:

```bash
bash scripts/train_model.sh beats
```

Explicit equivalents:

```bash
python3 scripts/train.py --config_file configs/config_eat.yaml
python3 scripts/train.py --config_file configs/config_beats.yaml
```

Training writes logs under `runs/` and checkpoints under the `save_dir` defined
in the selected config.

## Validate

Validation uses the same checkpoint, config, layers, channel, and scoring method
as final submission generation.

```bash
bash scripts/validate.sh eat
bash scripts/validate.sh beats
```

Equivalent explicit commands:

```bash
python3 scripts/validate.py --model eat
python3 scripts/validate.py --model beats
```

Important final validation settings:

- BEATs: layer `10`, channel `0` (`ch1`), KNN `top_k=1`, target SMOTE ratio `0.25`, SMOTE `k=3`
- EAT: layers `8,10,12`, score mean ensemble, `knn_domain_local_density`, source K `16`, target K `9`, scale `sum`

## Final Submission

Generate the EAT final submission:

```bash
bash scripts/test.sh eat
```

Generate the BEATs final submission:

```bash
bash scripts/test.sh beats
```

Direct entry:

```bash
python3 scripts/create_submission.py --model eat
python3 scripts/create_submission.py --model beats
```

Outputs are written to:

- `final_submission_eat/team_exp_diffpt_eat_layers8_10_12_score_mean_<timestamp>/`
- `final_submission/team_exp_beats_top1_smote_<timestamp>/`

Each output directory contains:

- `anomaly_score_<machine>_section_00_test.csv`
- `decision_result_<machine>_section_00_test.csv`
- `submission_summary.json`
- matching `.zip` archive next to the directory

## Fusion Submission

After generating BEATs and EAT final submissions, create the best tuned fusion:

```bash
python3 scripts/fuse_submissions.py \
  --weights beats=0.55,diffpt=0.45 \
  --norm rank \
  --team-name team_exp_fusion_top1_rank_b055_d045
```

By default, `fuse_submissions.py` finds the latest local BEATs and EAT final
submission directories:

```text
final_submission/team_exp_beats_top1_smote_*
final_submission_eat/team_exp_diffpt_eat_layers8_10_12_score_mean_*
```

To pass inputs explicitly:

```bash
python3 scripts/fuse_submissions.py \
  --input beats=final_submission/team_exp_beats_top1_smote_YYYYMMDD-HHMMSS \
  --input diffpt=final_submission_eat/team_exp_diffpt_eat_layers8_10_12_score_mean_YYYYMMDD-HHMMSS \
  --weights beats=0.55,diffpt=0.45 \
  --norm rank
```

To retune fusion weights on dev scores, first run both validation commands, then:

```bash
bash scripts/select_fusion_from_test_scp.sh
```

## Smoke Test

A quick code-path check can be run with tiny SCP files, but it is not a score
reproduction. Full validation should use `/workspace/data/test_dev.scp`.

The code was smoke-tested in `torchds:updated` with CUDA enabled, loading both
final checkpoints and producing dev-score JSON outputs.
