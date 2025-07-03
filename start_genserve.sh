#!/bin/bash
# start_genserve.sh - GenServe启动脚本，支持CPU Offload

# 设置错误处理
set -e

echo "🚀 GenServe 启动脚本"
echo "====================="

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU状态检查:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  未检测到nvidia-smi，将使用CPU模式"
fi

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.8,roundup_power2_divisions:16"
export FLUX_GPUS="0,1,2"
# CPU Offload配置（针对24GB显存不足）
export ENABLE_CPU_OFFLOAD="true"
export CPU_OFFLOAD_AGGRESSIVE="true"
export FLUX_USE_CPU_OFFLOAD="true"
export FLUX_OFFLOAD_TEXT_ENCODER="true"
export FLUX_OFFLOAD_VAE="true"

# 数据类型设置（bfloat16为Flux推荐）
export TORCH_DTYPE="bfloat16"

# 设置GPU配置（如果有多GPU）
if [ -z "$FLUX_GPUS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        if [ $GPU_COUNT -gt 1 ]; then
            export FLUX_GPUS="0,1,2,3,4,5,6,7"  # 8卡配置
            echo "🎯 自动检测到${GPU_COUNT}张GPU，设置FLUX_GPUS=${FLUX_GPUS}"
        else
            export FLUX_GPUS="0"
            echo "🎯 检测到单GPU，设置FLUX_GPUS=${FLUX_GPUS}"
        fi
    else
        echo "🎯 使用CPU模式"
    fi
else
    echo "🎯 使用配置的GPU: ${FLUX_GPUS}"
fi

# 模型路径配置
if [ -z "$FLUX_MODEL_PATH" ]; then
    export FLUX_MODEL_PATH="/home/shuzuan/prj/models/flux1-dev"
    echo "🎯 使用默认模型路径: ${FLUX_MODEL_PATH}"
else
    echo "🎯 使用配置的模型路径: ${FLUX_MODEL_PATH}"
fi

# 验证模型路径
if [ ! -d "$FLUX_MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: ${FLUX_MODEL_PATH}"
    echo "请确保模型已正确下载并设置正确的路径"
    exit 1
fi

# 并发配置
export MAX_GLOBAL_QUEUE_SIZE="50"
export MAX_GPU_QUEUE_SIZE="5"  # 减少GPU队列大小
export TASK_TIMEOUT="300"

# 服务配置
export HOST="0.0.0.0"
export PORT="12411"

echo ""
echo "🔧 配置摘要:"
echo "  服务地址: ${HOST}:${PORT}"
echo "  GPU配置: ${FLUX_GPUS}"
echo "  CPU Offload: ${ENABLE_CPU_OFFLOAD}"
echo "  激进模式: ${CPU_OFFLOAD_AGGRESSIVE}"
echo "  数据类型: ${TORCH_DTYPE}"
echo "  模型路径: ${FLUX_MODEL_PATH}"
echo "  最大全局队列: ${MAX_GLOBAL_QUEUE_SIZE}"
echo "  最大GPU队列: ${MAX_GPU_QUEUE_SIZE}"
echo ""

# 启动服务
echo "🚀 启动GenServe服务..."
python main.py

# 如果脚本被中断，清理环境
trap 'echo "🛑 正在关闭服务..."; kill $!; exit' INT TERM