#!/bin/bash
# 列出所有实验的脚本

# 获取脚本所在目录的父目录（ds26根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS26_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Available Experiments ==="
echo ""

RUNS_DIR="${DS26_DIR}/runs"
CHECKPOINTS_DIRS=("${DS26_DIR}/checkpoints_beats_lora" "${DS26_DIR}/checkpoints_beats_full" "${DS26_DIR}/checkpoints_ast_lora" "${DS26_DIR}/checkpoints_ast_full" "${DS26_DIR}/checkpoints")

# 列出runs目录下的所有tag
if [ -d "$RUNS_DIR" ]; then
    echo "📊 Runs (logs and metrics):"
    for tag_dir in ${RUNS_DIR}/*/; do
        if [ -d "$tag_dir" ]; then
            tag=$(basename "$tag_dir")
            num_logs=$(ls -1 ${tag_dir}training_*.log 2>/dev/null | wc -l)
            num_metrics=$(ls -1 ${tag_dir}metrics_*.csv 2>/dev/null | wc -l)
            latest_log=$(ls -t ${tag_dir}training_*.log 2>/dev/null | head -1)
            
            if [ -n "$latest_log" ]; then
                log_date=$(basename "$latest_log" | cut -d'_' -f2 | cut -d'-' -f1-2)
                echo "  - $tag (${num_logs} logs, ${num_metrics} metrics, latest: ${log_date})"
            else
                echo "  - $tag (${num_logs} logs, ${num_metrics} metrics)"
            fi
        fi
    done
    echo ""
else
    echo "  No runs directory found"
    echo ""
fi

# 列出checkpoints目录下的所有tag
echo "💾 Checkpoints:"
found_checkpoints=false
for checkpoint_dir in "${CHECKPOINTS_DIRS[@]}"; do
    if [ -d "$checkpoint_dir" ]; then
        for tag_dir in ${checkpoint_dir}/*/; do
            if [ -d "$tag_dir" ]; then
                found_checkpoints=true
                tag=$(basename "$tag_dir")
                base_dir=$(basename "$checkpoint_dir")
                num_checkpoints=$(ls -1 ${tag_dir}model_epoch*.pth 2>/dev/null | wc -l)
                latest_checkpoint=$(ls -t ${tag_dir}model_epoch*.pth 2>/dev/null | head -1)
                
                if [ -n "$latest_checkpoint" ]; then
                    epoch=$(basename "$latest_checkpoint" | sed 's/model_epoch\([0-9]*\)\.pth/\1/')
                    echo "  - ${base_dir}/${tag} (${num_checkpoints} checkpoints, latest: epoch ${epoch})"
                else
                    echo "  - ${base_dir}/${tag} (${num_checkpoints} checkpoints)"
                fi
            fi
        done
    fi
done

if [ "$found_checkpoints" = false ]; then
    echo "  No checkpoints found"
fi

echo ""
echo "=== Quick Commands ==="
echo ""
echo "View metrics for a specific tag:"
echo "  ./view_metrics.sh <tag>"
echo ""
echo "Compare experiments:"
echo "  tail -1 runs/*/metrics_*.csv"
echo ""
echo "List all metrics files:"
echo "  find runs -name 'metrics_*.csv' -type f"
echo ""
