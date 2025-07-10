#!/bin/bash
# 优化的GenServe启动脚本
# 基于性能测试结果自动生成

# =============================================================================
# 环境变量设置
# =============================================================================

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# =============================================================================
# 模型路径配置
# =============================================================================

# 基础模型路径
export FLUX_MODEL_PATH="/home/shuzuan/prj/models/flux1-dev"

# =============================================================================
# GPU分配配置 - 基于您的测试结果优化
# =============================================================================

# 基础模型GPU分配 (使用GPU 0,1,3,4 - 根据您的测试结果)
export FLUX_GPUS="0,1,3,4"

# =============================================================================
# 性能和并发配置 - 基于测试结果优化
# =============================================================================

# 调度器睡眠时间 - 减少以提高响应性
export SCHEDULER_SLEEP_TIME="0.05"

# 全局任务队列最大大小 - 增加缓冲
export MAX_GLOBAL_QUEUE_SIZE="150"

# 每个GPU队列最大大小 - 减少单个GPU过载
export MAX_GPU_QUEUE_SIZE="3"

# 任务超时时间 - 基于您的测试结果调整
export TASK_TIMEOUT="240"

# GPU内存清理阈值 (MB)
export GPU_MEMORY_THRESHOLD_MB="800"

# GPU内存清理间隔 - 增加频率
export GPU_MEMORY_CLEANUP_INTERVAL="3"

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

# 启用CORS
export ENABLE_CORS="true"

# CORS来源
export CORS_ORIGINS="*"

# =============================================================================
# 性能优化配置
# =============================================================================

# PyTorch内存管理配置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6"

# 启用优化
export ENABLE_OPTIMIZATION="true"

# 启用内存高效注意力
export MEMORY_EFFICIENT_ATTENTION="true"

# 启用CPU卸载
export ENABLE_CPU_OFFLOAD="true"

# 使用bfloat16
export TORCH_DTYPE="bfloat16"

# =============================================================================
# 日志配置
# =============================================================================

# 日志级别
export LOG_LEVEL="INFO"

# =============================================================================
# API认证配置
# =============================================================================

# 默认API密钥（如果未配置其他密钥，将使用此密钥）
export DEFAULT_API_KEY="genserve-default-key-2024"

# API密钥文件加密密钥（可选，用于加密api_keys.json文件）
# 生成命令: python generate_encryption_key.py
# export API_KEYS_ENCRYPTION_KEY="your-44-character-encryption-key-here"

# API密钥配置（格式：key:name:permissions）
# 支持最多10个API密钥
export API_KEY_1="your-api-key-1:user1:generation,readonly"
export API_KEY_2="your-api-key-2:user2:generation"
export API_KEY_3="admin-key-2024:admin:all"

# 示例密钥配置（请根据实际需要修改）
# export API_KEY_1="abc123def456:developer:generation,readonly"
# export API_KEY_2="xyz789ghi012:user:generation"
# export API_KEY_3="admin123admin456:admin:all"

# =============================================================================
# 启动服务
# =============================================================================

echo "🚀 启动优化的GenServe服务..."
echo "📊 优化配置:"
echo "  - 调度器睡眠时间: ${SCHEDULER_SLEEP_TIME}s"
echo "  - 全局队列大小: ${MAX_GLOBAL_QUEUE_SIZE}"
echo "  - GPU队列大小: ${MAX_GPU_QUEUE_SIZE}"
echo "  - 任务超时: ${TASK_TIMEOUT}s"
echo "  - GPU内存清理间隔: ${GPU_MEMORY_CLEANUP_INTERVAL}s"
echo "  - 使用的GPU: ${FLUX_GPUS}"

# 启动服务
python main.py
