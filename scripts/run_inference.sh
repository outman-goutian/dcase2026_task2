#!/bin/bash
# 便捷推理脚本

# 获取脚本所在目录的父目录（ds26根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS26_DIR="$(dirname "$SCRIPT_DIR")"

# 使用方法:
# ./run_inference.sh <tag> <epoch>                    # 使用默认配置推理指定tag和epoch
# ./run_inference.sh <tag> <epoch> <config>           # 使用指定配置推理
# ./run_inference.sh --checkpoint <path>              # 直接指定checkpoint路径

set -e  # 遇到错误立即退出

# 默认值
DEFAULT_CONFIG="${DS26_DIR}/configs/inference_beats_lora.yaml"
DEFAULT_CHECKPOINT_DIR="${DS26_DIR}/checkpoints_beats_lora"

# 显示帮助信息
show_help() {
    echo "Usage:"
    echo "  ./run_inference.sh <tag> <epoch>                    # 使用默认配置"
    echo "  ./run_inference.sh <tag> <epoch> <config>           # 使用指定配置"
    echo "  ./run_inference.sh --checkpoint <checkpoint_path>   # 直接指定checkpoint"
    echo ""
    echo "Examples:"
    echo "  ./run_inference.sh exp1 10"
    echo "  ./run_inference.sh exp1 10 configs/inference_beats_full.yaml"
    echo "  ./run_inference.sh --checkpoint checkpoints_beats_lora/exp1/model_epoch10.pth"
    echo ""
    echo "Available experiments:"
    "${SCRIPT_DIR}/list_experiments.sh" 2>/dev/null || echo "  (run ./list_experiments.sh to see available experiments)"
}

# 检查参数
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# 处理 --checkpoint 参数
if [ "$1" = "--checkpoint" ]; then
    if [ $# -lt 2 ]; then
        echo "Error: --checkpoint requires a path argument"
        exit 1
    fi
    CHECKPOINT_PATH="$2"
    CONFIG="${3:-$DEFAULT_CONFIG}"
    
    echo "Running inference with:"
    echo "  Checkpoint: $CHECKPOINT_PATH"
    echo "  Config: $CONFIG"
    echo ""
    
    python "${SCRIPT_DIR}/inference.py" --config "$CONFIG" --checkpoint "$CHECKPOINT_PATH"
    exit $?
fi

# 处理 tag + epoch 参数
if [ $# -lt 2 ]; then
    echo "Error: Missing arguments"
    show_help
    exit 1
fi

TAG="$1"
EPOCH="$2"
CONFIG="${3:-$DEFAULT_CONFIG}"

# 从配置文件中提取checkpoint目录
# 简单方法：根据配置文件名推断
if [[ "$CONFIG" == *"beats_lora"* ]]; then
    CHECKPOINT_DIR="${DS26_DIR}/checkpoints_beats_lora"
elif [[ "$CONFIG" == *"beats_full"* ]]; then
    CHECKPOINT_DIR="${DS26_DIR}/checkpoints_beats_full"
elif [[ "$CONFIG" == *"ast_lora"* ]]; then
    CHECKPOINT_DIR="${DS26_DIR}/checkpoints_ast_lora"
elif [[ "$CONFIG" == *"ast_full"* ]]; then
    CHECKPOINT_DIR="${DS26_DIR}/checkpoints_ast_full"
else
    CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
fi

# 构建checkpoint路径
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${TAG}/model_epoch${EPOCH}.pth"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints for tag '$TAG':"
    if [ -d "${CHECKPOINT_DIR}/${TAG}" ]; then
        ls -1 "${CHECKPOINT_DIR}/${TAG}"/model_epoch*.pth 2>/dev/null || echo "  (none)"
    else
        echo "  Tag directory not found: ${CHECKPOINT_DIR}/${TAG}"
    fi
    exit 1
fi

echo "Running inference with:"
echo "  Tag: $TAG"
echo "  Epoch: $EPOCH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Config: $CONFIG"
echo ""

# 运行推理
python "${SCRIPT_DIR}/inference.py" --config "$CONFIG" --checkpoint "$CHECKPOINT_PATH"

echo ""
echo "Inference completed!"
