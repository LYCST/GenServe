#!/usr/bin/env python3
"""
配置优化脚本
根据性能测试结果自动调整并行参数
"""

import json
import os
import shutil
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigOptimizer:
    """配置优化器"""
    
    def __init__(self, config_file: str = "config.py"):
        self.config_file = config_file
        self.backup_file = f"{config_file}.backup"
    
    def backup_config(self):
        """备份当前配置"""
        if os.path.exists(self.config_file):
            shutil.copy2(self.config_file, self.backup_file)
            logger.info(f"✅ 配置已备份到: {self.backup_file}")
    
    def restore_config(self):
        """恢复配置"""
        if os.path.exists(self.backup_file):
            shutil.copy2(self.backup_file, self.config_file)
            logger.info(f"✅ 配置已从备份恢复: {self.backup_file}")
    
    def optimize_for_parallel_performance(self, test_results: Dict[str, Any]):
        """根据测试结果优化配置"""
        logger.info("🔧 开始优化并行配置...")
        
        # 分析测试结果
        avg_response_time = test_results.get("avg_response_time", 50)
        max_response_time = test_results.get("max_response_time", 90)
        min_response_time = test_results.get("min_response_time", 25)
        balance_score = test_results.get("balance_score", 0.67)
        concurrent_efficiency = test_results.get("concurrent_efficiency", 0.8)
        
        # 计算优化建议
        optimizations = self._calculate_optimizations(
            avg_response_time, max_response_time, min_response_time,
            balance_score, concurrent_efficiency
        )
        
        # 应用优化
        self._apply_optimizations(optimizations)
        
        logger.info("✅ 配置优化完成")
        return optimizations
    
    def _calculate_optimizations(self, avg_response_time, max_response_time, 
                               min_response_time, balance_score, concurrent_efficiency):
        """计算优化建议"""
        optimizations = {
            "scheduler_sleep_time": 0.1,
            "max_global_queue_size": 100,
            "max_gpu_queue_size": 5,
            "task_timeout": 180,
            "gpu_memory_cleanup_interval": 5,
            "enable_aggressive_cleanup": True
        }
        
        # 根据响应时间差异调整调度器睡眠时间
        time_variance = max_response_time - min_response_time
        if time_variance > avg_response_time * 0.5:
            # 响应时间差异大，减少调度器睡眠时间以提高响应性
            optimizations["scheduler_sleep_time"] = 0.05
            logger.info("  📊 检测到响应时间差异较大，减少调度器睡眠时间")
        
        # 根据负载均衡评分调整队列大小
        if balance_score < 0.7:
            # 负载不均衡，增加全局队列大小以提供更多缓冲
            optimizations["max_global_queue_size"] = 150
            optimizations["max_gpu_queue_size"] = 3  # 减少单个GPU队列大小
            logger.info("  ⚖️ 检测到负载不均衡，调整队列配置")
        
        # 根据并发效率调整超时设置
        if concurrent_efficiency < 0.8:
            # 并发效率低，增加超时时间
            optimizations["task_timeout"] = 240
            logger.info("  🚀 检测到并发效率较低，增加任务超时时间")
        
        # 根据平均响应时间调整内存清理间隔
        if avg_response_time > 60:
            # 响应时间长，增加内存清理频率
            optimizations["gpu_memory_cleanup_interval"] = 3
            optimizations["enable_aggressive_cleanup"] = True
            logger.info("  🧹 检测到响应时间较长，增加内存清理频率")
        
        return optimizations
    
    def _apply_optimizations(self, optimizations: Dict[str, Any]):
        """应用优化配置"""
        logger.info("📝 应用优化配置:")
        
        for key, value in optimizations.items():
            logger.info(f"  {key}: {value}")
        
        # 这里可以添加实际的配置文件修改逻辑
        # 由于配置文件是Python代码，需要谨慎修改
        logger.info("💡 请手动更新配置文件中的以下参数:")
        
        config_updates = {
            "SCHEDULER_SLEEP_TIME": optimizations["scheduler_sleep_time"],
            "MAX_GLOBAL_QUEUE_SIZE": optimizations["max_global_queue_size"],
            "MAX_GPU_QUEUE_SIZE": optimizations["max_gpu_queue_size"],
            "TASK_TIMEOUT": optimizations["task_timeout"],
            "GPU_MEMORY_CLEANUP_INTERVAL": optimizations["gpu_memory_cleanup_interval"],
            "ENABLE_AGGRESSIVE_CLEANUP": optimizations["enable_aggressive_cleanup"]
        }
        
        for key, value in config_updates.items():
            logger.info(f"  {key} = {value}")
    
    def generate_optimized_start_script(self, optimizations: Dict[str, Any]):
        """生成优化的启动脚本"""
        script_content = f"""#!/bin/bash
# 优化的GenServe启动脚本
# 基于性能测试结果自动生成

# 并发配置优化
export SCHEDULER_SLEEP_TIME="{optimizations['scheduler_sleep_time']}"
export MAX_GLOBAL_QUEUE_SIZE="{optimizations['max_global_queue_size']}"
export MAX_GPU_QUEUE_SIZE="{optimizations['max_gpu_queue_size']}"
export TASK_TIMEOUT="{optimizations['task_timeout']}"
export GPU_MEMORY_CLEANUP_INTERVAL="{optimizations['gpu_memory_cleanup_interval']}"
export ENABLE_AGGRESSIVE_CLEANUP="{str(optimizations['enable_aggressive_cleanup']).lower()}"

# 性能优化配置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6"
export ENABLE_OPTIMIZATION="true"
export MEMORY_EFFICIENT_ATTENTION="true"
export ENABLE_CPU_OFFLOAD="true"

# 启动服务
python main.py
"""
        
        with open("start_optimized.sh", "w") as f:
            f.write(script_content)
        
        os.chmod("start_optimized.sh", 0o755)
        logger.info("✅ 优化的启动脚本已生成: start_optimized.sh")

def main():
    """主函数"""
    optimizer = ConfigOptimizer()
    
    # 模拟测试结果（您可以从实际测试中获取）
    test_results = {
        "avg_response_time": 49.75,
        "max_response_time": 87.14,
        "min_response_time": 25.49,
        "balance_score": 0.67,
        "concurrent_efficiency": 0.8,
        "total_requests": 10,
        "success_rate": 1.0
    }
    
    logger.info("🔧 开始配置优化...")
    
    # 备份当前配置
    optimizer.backup_config()
    
    # 计算并应用优化
    optimizations = optimizer.optimize_for_parallel_performance(test_results)
    
    # 生成优化的启动脚本
    optimizer.generate_optimized_start_script(optimizations)
    
    logger.info("🎉 配置优化完成！")
    logger.info("📋 建议:")
    logger.info("  1. 使用 start_optimized.sh 启动服务")
    logger.info("  2. 运行性能测试验证优化效果")
    logger.info("  3. 根据实际效果进一步调整参数")

if __name__ == "__main__":
    main() 