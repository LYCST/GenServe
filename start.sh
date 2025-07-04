#!/bin/bash

# GenServe 多GPU图片生成服务启动脚本
# 作者：GenServe Team
# 版本：2.0.0

echo "🚀 启动 GenServe 多GPU图片生成服务..."

# =============================================================================
# 基础服务配置
# =============================================================================

# 服务监听地址和端口
export HOST="0.0.0.0"
export PORT="12411"

# 日志级别 (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL="INFO"

# =============================================================================
# GPU和设备配置
# =============================================================================

# 是否使用GPU (true/false)
export USE_GPU="true"

# PyTorch数据类型 (bfloat16推荐用于Flux模型)
export TORCH_DTYPE="bfloat16"

# PyTorch CUDA内存分配配置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6"

# =============================================================================
# 模型路径配置 (请根据您的实际模型路径修改)
# =============================================================================

# 基础Flux模型路径 (必须配置)
export FLUX_MODEL_PATH="/home/shuzuan/prj/models/flux1-dev"

# Depth控制模型路径 (可选，如果不配置则跳过该模型)
# export FLUX_DEPTH_MODEL_PATH="/home/shuzuan/prj/models/FLUX.1-Depth-dev"

# Fill填充模型路径 (可选，如果不配置则跳过该模型)
# export FLUX_FILL_MODEL_PATH="/home/shuzuan/prj/models/FLUX.1-Fill-dev"

# Canny边缘控制模型路径 (可选，如果不配置则跳过该模型)
# export FLUX_CANNY_MODEL_PATH="/home/shuzuan/prj/models/flux1-canny-dev"

# OpenPose姿态控制模型路径 (可选，如果不配置则跳过该模型)
# export FLUX_OPENPOSE_MODEL_PATH="/home/shuzuan/prj/models/flux1-openpose-dev"

# =============================================================================
# GPU分配配置
# =============================================================================

# 基础模型GPU分配 (使用GPU 0和1)
export FLUX_GPUS="0,1"

# Depth模型GPU分配 (使用GPU 2，如果不配置则使用基础模型的GPU)
# export FLUX_DEPTH_GPUS="2"

# Fill模型GPU分配 (使用GPU 3，如果不配置则使用基础模型的GPU)
# export FLUX_FILL_GPUS="3"

# Canny模型GPU分配 (使用GPU 0和2，如果不配置则使用基础模型的GPU)
# export FLUX_CANNY_GPUS="0,2"

# OpenPose模型GPU分配 (使用GPU 1和3，如果不配置则使用基础模型的GPU)
# export FLUX_OPENPOSE_GPUS="1,3"

# =============================================================================
# 性能和并发配置
# =============================================================================

# 最大并发请求数
export MAX_CONCURRENT_REQUESTS="10"

# 任务超时时间 (秒)
export TASK_TIMEOUT="180"

# 全局任务队列最大大小
export MAX_GLOBAL_QUEUE_SIZE="100"

# 每个GPU队列最大大小
export MAX_GPU_QUEUE_SIZE="5"

# GPU内存清理阈值 (MB)
export GPU_MEMORY_THRESHOLD_MB="800"

# 启用激进内存清理
export ENABLE_AGGRESSIVE_CLEANUP="true"

# =============================================================================
# 安全配置
# =============================================================================

# 最大提示词长度
export MAX_PROMPT_LENGTH="1000"

# 最大图片尺寸
export MAX_IMAGE_SIZE="2048"

# =============================================================================
# API配置
# =============================================================================

# 启用跨域请求
export ENABLE_CORS="true"

# 允许的跨域来源 (逗号分隔，*表示允许所有)
export CORS_ORIGINS="*"

# =============================================================================
# 优化配置
# =============================================================================

# 启用模型优化
export ENABLE_OPTIMIZATION="true"

# 启用内存高效注意力机制
export MEMORY_EFFICIENT_ATTENTION="true"

# 启用CPU卸载
export ENABLE_CPU_OFFLOAD="true"

# =============================================================================
# 环境检查
# =============================================================================

echo "📋 检查环境配置..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装或不在PATH中"
    exit 1
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  GPU $line"
    done
else
    echo "⚠️  nvidia-smi 未找到，可能没有NVIDIA GPU或驱动未安装"
fi

# 检查基础模型路径
if [ ! -d "$FLUX_MODEL_PATH" ]; then
    echo "❌ 基础模型路径不存在: $FLUX_MODEL_PATH"
    echo "请检查并修改 FLUX_MODEL_PATH 环境变量"
    exit 1
fi

echo "✅ 基础模型路径: $FLUX_MODEL_PATH"

# 检查可选模型路径
if [ -n "$FLUX_DEPTH_MODEL_PATH" ]; then
    if [ -d "$FLUX_DEPTH_MODEL_PATH" ]; then
        echo "✅ Depth模型路径: $FLUX_DEPTH_MODEL_PATH"
    else
        echo "⚠️  Depth模型路径不存在: $FLUX_DEPTH_MODEL_PATH (将跳过该模型)"
        unset FLUX_DEPTH_MODEL_PATH
    fi
fi

if [ -n "$FLUX_FILL_MODEL_PATH" ]; then
    if [ -d "$FLUX_FILL_MODEL_PATH" ]; then
        echo "✅ Fill模型路径: $FLUX_FILL_MODEL_PATH"
    else
        echo "⚠️  Fill模型路径不存在: $FLUX_FILL_MODEL_PATH (将跳过该模型)"
        unset FLUX_FILL_MODEL_PATH
    fi
fi

if [ -n "$FLUX_CANNY_MODEL_PATH" ]; then
    if [ -d "$FLUX_CANNY_MODEL_PATH" ]; then
        echo "✅ Canny模型路径: $FLUX_CANNY_MODEL_PATH"
    else
        echo "⚠️  Canny模型路径不存在: $FLUX_CANNY_MODEL_PATH (将跳过该模型)"
        unset FLUX_CANNY_MODEL_PATH
    fi
fi

if [ -n "$FLUX_OPENPOSE_MODEL_PATH" ]; then
    if [ -d "$FLUX_OPENPOSE_MODEL_PATH" ]; then
        echo "✅ OpenPose模型路径: $FLUX_OPENPOSE_MODEL_PATH"
    else
        echo "⚠️  OpenPose模型路径不存在: $FLUX_OPENPOSE_MODEL_PATH (将跳过该模型)"
        unset FLUX_OPENPOSE_MODEL_PATH
    fi
fi

# =============================================================================
# 依赖检查
# =============================================================================

echo "📦 检查Python依赖..."

# 检查requirements.txt
if [ -f "requirements.txt" ]; then
    echo "检查依赖包..."
    python3 -m pip check > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ 依赖包检查通过"
    else
        echo "⚠️  依赖包可能有问题，建议运行: pip install -r requirements.txt"
    fi
else
    echo "⚠️  requirements.txt 文件不存在"
fi

# =============================================================================
# 启动服务
# =============================================================================

echo "🚀 启动GenServe服务..."
echo "服务地址: http://$HOST:$PORT"
echo "API文档: http://$HOST:$PORT/docs"
echo "按 Ctrl+C 停止服务"
echo "==============================================="

# 启动主程序
python3 main.py

# =============================================================================
# 清理和退出
# =============================================================================

echo ""
echo "🛑 GenServe服务已停止"
echo "感谢使用GenServe!" 