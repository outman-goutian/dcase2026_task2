#!/usr/bin/env bash
set -euo pipefail

TEST_SCP="${TEST_SCP:-/workspace/data/test_dev.scp}"
WEIGHT_STEP="${WEIGHT_STEP:-0.05}"
SELECT_METRIC="${SELECT_METRIC:-official}"
TUNE_NORMS="${TUNE_NORMS:-rank,minmax,zscore}"

latest_dir_with_scores() {
    local pattern="$1"
    local dir
    while IFS= read -r dir; do
        if find "${dir}" -maxdepth 1 -type f -name 'anomaly_score_*_section_00_test.csv' -print -quit | grep -q .; then
            printf '%s\n' "${dir}"
            return 0
        fi
    done < <(ls -dt ${pattern} 2>/dev/null || true)
    return 1
}

BEATS_TUNE_DIR="${BEATS_TUNE_DIR:-$(latest_dir_with_scores final_submission_eval_check/eval_check_top1_smote_* || true)}"
DIFFPT_TUNE_DIR="${DIFFPT_TUNE_DIR:-$(latest_dir_with_scores final_submission_eat/team_exp_diffpt_eat_layers8_10_12_score_mean_* || true)}"

if [ -z "${BEATS_TUNE_DIR}" ] || [ ! -d "${BEATS_TUNE_DIR}" ]; then
    echo "Missing BEATs tune score dir. Generate it first or set BEATS_TUNE_DIR." >&2
    exit 1
fi

if [ -z "${DIFFPT_TUNE_DIR}" ] || [ ! -d "${DIFFPT_TUNE_DIR}" ]; then
    echo "Missing DiffPT/EAT tune score dir. Generate it first or set DIFFPT_TUNE_DIR." >&2
    exit 1
fi

python3 scripts/fuse_submissions.py \
    --tune-scp "${TEST_SCP}" \
    --tune-input "beats=${BEATS_TUNE_DIR}" \
    --tune-input "diffpt=${DIFFPT_TUNE_DIR}" \
    --tune-norms "${TUNE_NORMS}" \
    --weight-step "${WEIGHT_STEP}" \
    --select-metric "${SELECT_METRIC}" \
    --team-name "team_exp_fusion_tuned_${SELECT_METRIC}"
