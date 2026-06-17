#!/bin/bash
# 查看训练指标的辅助脚本

# 获取脚本所在目录的父目录（ds26根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS26_DIR="$(dirname "$SCRIPT_DIR")"

# 使用方法:
# ./view_metrics.sh                           # 查看最新的metrics文件
# ./view_metrics.sh exp1                      # 查看exp1 tag的最新metrics
# ./view_metrics.sh exp1/metrics_xxx.csv      # 查看指定的metrics文件

METRICS_DIR="${DS26_DIR}/runs"

if [ $# -eq 0 ]; then
    # 如果没有参数，找到最新的metrics文件（搜索所有tag目录）
    METRICS_FILE=$(find ${METRICS_DIR} -name "metrics_*.csv" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    if [ -z "$METRICS_FILE" ]; then
        echo "Error: No metrics CSV files found in ${METRICS_DIR}/"
        exit 1
    fi
    echo "Viewing latest metrics file: $METRICS_FILE"
    echo ""
elif [ $# -eq 1 ]; then
    # 检查参数是tag还是文件路径
    if [ -f "$1" ]; then
        # 参数是文件路径
        METRICS_FILE="$1"
    elif [ -d "${METRICS_DIR}/$1" ]; then
        # 参数是tag，找到该tag下最新的metrics文件
        METRICS_FILE=$(ls -t ${METRICS_DIR}/$1/metrics_*.csv 2>/dev/null | head -1)
        if [ -z "$METRICS_FILE" ]; then
            echo "Error: No metrics CSV files found in ${METRICS_DIR}/$1/"
            exit 1
        fi
        echo "Viewing latest metrics file for tag '$1': $METRICS_FILE"
        echo ""
    elif [ -f "${METRICS_DIR}/$1" ]; then
        # 参数是相对于METRICS_DIR的文件路径
        METRICS_FILE="${METRICS_DIR}/$1"
    else
        echo "Error: Tag directory or file not found: $1"
        echo ""
        echo "Available tags:"
        ls -d ${METRICS_DIR}/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null || echo "  (none)"
        exit 1
    fi
fi

# 检查是否安装了column命令（大多数Linux系统都有）
if command -v column &> /dev/null; then
    echo "=== Training Metrics (formatted) ==="
    echo ""
    cat "$METRICS_FILE" | column -t -s,
else
    # 如果没有column，使用cat直接显示
    echo "=== Training Metrics ==="
    echo ""
    cat "$METRICS_FILE"
fi

echo ""
echo "=== Summary ==="
echo "Total epochs: $(tail -n +2 "$METRICS_FILE" | wc -l)"
echo ""
echo "Best validation AUC:"
tail -n +2 "$METRICS_FILE" | awk -F',' '{if ($4 != "") print $1, $4}' | sort -k2 -rn | head -1 | awk '{printf "  Epoch %s: %.4f\n", $1, $2}'
echo ""
echo "Best validation pAUC:"
tail -n +2 "$METRICS_FILE" | awk -F',' '{if ($5 != "") print $1, $5}' | sort -k2 -rn | head -1 | awk '{printf "  Epoch %s: %.4f\n", $1, $2}'
echo ""
echo "Latest metrics:"
tail -1 "$METRICS_FILE" | awk -F',' '{
    printf "  Epoch: %s\n", $1
    printf "  Train Loss: %.4f, Train Acc: %.4f\n", $2, $3
    if ($4 != "") {
        printf "  Val AUC: %.4f, pAUC: %.4f, F1: %.4f\n", $4, $5, $6
        printf "  Source AUC: %.4f, pAUC: %.4f\n", $7, $8
        printf "  Target AUC: %.4f, pAUC: %.4f\n", $9, $10
    }
}'
