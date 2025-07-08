#!/usr/bin/env python3
"""
并行性能优化脚本
分析当前并行效果并提供优化建议
"""

import asyncio
import aiohttp
import time
import json
import random
import statistics
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelPerformanceOptimizer:
    """并行性能优化器"""
    
    def __init__(self, base_url: str = "http://localhost:12411"):
        self.base_url = base_url
        self.results = []
    
    async def test_parallel_performance(self, num_requests: int = 20, batch_size: int = 5):
        """测试并行性能"""
        logger.info(f"🚀 开始并行性能测试: {num_requests}个请求，批次大小: {batch_size}")
        
        # 生成测试提示词
        prompts = [
            "A beautiful sunset over mountains, digital art",
            "A futuristic city with flying cars, sci-fi style",
            "A peaceful forest with ancient trees, fantasy art",
            "A steampunk mechanical robot, detailed illustration",
            "A magical crystal cave with glowing crystals",
            "A cyberpunk street scene with neon lights",
            "A medieval castle on a hill, fantasy landscape",
            "A space station orbiting Earth, sci-fi scene",
            "A underwater city with mermaids, fantasy art",
            "A desert oasis with palm trees, realistic style"
        ]
        
        # 分批发送请求
        all_tasks = []
        start_time = time.time()
        
        for i in range(0, num_requests, batch_size):
            batch_end = min(i + batch_size, num_requests)
            logger.info(f"📤 发送批次 {i//batch_size + 1}: 请求 {i+1}-{batch_end}")
            
            # 创建当前批次的任务
            batch_tasks = []
            for j in range(i, batch_end):
                prompt = prompts[j % len(prompts)]
                task = self._create_single_request(j + 1, prompt)
                batch_tasks.append(task)
            
            # 并发执行当前批次
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_tasks.extend(batch_results)
            
            # 短暂等待，避免过载
            if batch_end < num_requests:
                await asyncio.sleep(0.5)
        
        total_time = time.time() - start_time
        
        # 分析结果
        self._analyze_performance(all_tasks, total_time)
    
    async def _create_single_request(self, request_id: int, prompt: str) -> Dict[str, Any]:
        """创建单个请求"""
        start_time = time.time()
        
        try:
            # 构建请求数据
            data = {
                "prompt": f"{prompt}, request {request_id}",
                "model_id": "flux1-dev",
                "height": 512,
                "width": 512,
                "num_inference_steps": 20,  # 减少步数以加快测试
                "cfg": 3.5,
                "seed": random.randint(1, 1000000),
                "priority": random.randint(0, 2),  # 随机优先级
                "mode": "text2img"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/generate", json=data) as response:
                    response_time = time.time() - start_time
                    
                    result = {
                        "request_id": request_id,
                        "status_code": response.status,
                        "response_time": response_time,
                        "success": response.status == 200,
                        "timestamp": start_time
                    }
                    
                    if response.status == 200:
                        try:
                            response_json = await response.json()
                            result["task_id"] = response_json.get("task_id", "")
                            result["gpu_id"] = response_json.get("gpu_id", "")
                            result["model_id"] = response_json.get("model_id", "")
                            result["elapsed_time"] = response_json.get("elapsed_time", 0)
                            logger.info(f"请求 {request_id}: ✅ 成功，GPU: {result['gpu_id']}, 耗时: {response_time:.2f}s")
                        except:
                            logger.warning(f"请求 {request_id}: 响应解析失败")
                    else:
                        logger.error(f"请求 {request_id}: ❌ 失败，状态码: {response.status}")
                    
                    return result
                    
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"请求 {request_id}: ❌ 异常: {e}")
            return {
                "request_id": request_id,
                "status_code": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "timestamp": start_time
            }
    
    def _analyze_performance(self, results: List[Dict[str, Any]], total_time: float):
        """分析性能结果"""
        logger.info("=" * 80)
        logger.info("📊 并行性能分析报告")
        logger.info("=" * 80)
        
        # 过滤有效结果
        valid_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
        exception_results = [r for r in results if isinstance(r, Exception)]
        
        logger.info(f"总请求数: {len(results)}")
        logger.info(f"成功请求: {len(valid_results)} ({len(valid_results)/len(results)*100:.1f}%)")
        logger.info(f"失败请求: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
        logger.info(f"异常请求: {len(exception_results)} ({len(exception_results)/len(results)*100:.1f}%)")
        logger.info(f"总耗时: {total_time:.2f}s")
        
        if valid_results:
            # 响应时间分析
            response_times = [r["response_time"] for r in valid_results]
            elapsed_times = [r.get("elapsed_time", 0) for r in valid_results if r.get("elapsed_time")]
            
            logger.info(f"\n⏱️ 响应时间分析:")
            logger.info(f"  平均响应时间: {statistics.mean(response_times):.2f}s")
            logger.info(f"  中位数响应时间: {statistics.median(response_times):.2f}s")
            logger.info(f"  最快响应时间: {min(response_times):.2f}s")
            logger.info(f"  最慢响应时间: {max(response_times):.2f}s")
            logger.info(f"  响应时间标准差: {statistics.stdev(response_times):.2f}s")
            
            if elapsed_times:
                logger.info(f"\n⚡ 实际生成时间分析:")
                logger.info(f"  平均生成时间: {statistics.mean(elapsed_times):.2f}s")
                logger.info(f"  中位数生成时间: {statistics.median(elapsed_times):.2f}s")
                logger.info(f"  最快生成时间: {min(elapsed_times):.2f}s")
                logger.info(f"  最慢生成时间: {max(elapsed_times):.2f}s")
            
            # GPU使用分析
            gpu_usage = {}
            for r in valid_results:
                gpu_id = r.get("gpu_id", "unknown")
                gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
            
            logger.info(f"\n🎮 GPU使用情况:")
            for gpu_id, count in sorted(gpu_usage.items()):
                percentage = count / len(valid_results) * 100
                logger.info(f"  GPU {gpu_id}: {count} 个请求 ({percentage:.1f}%)")
            
            # 负载均衡评估
            if len(gpu_usage) > 1:
                counts = list(gpu_usage.values())
                mean_count = statistics.mean(counts)
                variance = statistics.variance(counts)
                balance_score = 1.0 / (1.0 + variance / mean_count) if mean_count > 0 else 0
                
                logger.info(f"\n⚖️ 负载均衡评估:")
                logger.info(f"  均衡度评分: {balance_score:.3f} (1.0为完美均衡)")
                if balance_score > 0.8:
                    logger.info(f"  ✅ 负载均衡效果优秀")
                elif balance_score > 0.6:
                    logger.info(f"  ⚠️ 负载均衡效果良好")
                else:
                    logger.info(f"  ❌ 负载均衡效果需要改进")
            
            # 并发效率评估
            avg_response_time = statistics.mean(response_times)
            concurrent_efficiency = total_time / avg_response_time if avg_response_time > 0 else 0
            
            logger.info(f"\n🚀 并发效率评估:")
            logger.info(f"  并发效率: {concurrent_efficiency:.2f} (理想值接近请求数)")
            logger.info(f"  理论最大并发: {len(valid_results)}")
            logger.info(f"  实际并发度: {concurrent_efficiency:.1f}")
            
            if concurrent_efficiency > len(valid_results) * 0.8:
                logger.info(f"  ✅ 并发效率优秀")
            elif concurrent_efficiency > len(valid_results) * 0.6:
                logger.info(f"  ⚠️ 并发效率良好")
            else:
                logger.info(f"  ❌ 并发效率需要改进")
        
        # 优化建议
        self._provide_optimization_suggestions(valid_results, total_time)
    
    def _provide_optimization_suggestions(self, valid_results: List[Dict[str, Any]], total_time: float):
        """提供优化建议"""
        logger.info(f"\n💡 优化建议:")
        
        if not valid_results:
            logger.info("  ❌ 没有成功请求，请检查服务状态")
            return
        
        # 分析响应时间分布
        response_times = [r["response_time"] for r in valid_results]
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        # 检查响应时间差异
        time_variance = statistics.variance(response_times)
        if time_variance > avg_response_time * 0.5:
            logger.info("  🔧 响应时间差异较大，建议:")
            logger.info("    - 检查GPU负载均衡算法")
            logger.info("    - 优化任务调度策略")
            logger.info("    - 考虑GPU性能差异")
        
        # 检查并发度
        concurrent_efficiency = total_time / avg_response_time if avg_response_time > 0 else 0
        if concurrent_efficiency < len(valid_results) * 0.7:
            logger.info("  🔧 并发度较低，建议:")
            logger.info("    - 增加GPU数量")
            logger.info("    - 优化队列管理")
            logger.info("    - 减少任务处理时间")
        
        # 检查GPU使用情况
        gpu_usage = {}
        for r in valid_results:
            gpu_id = r.get("gpu_id", "unknown")
            gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
        
        if len(gpu_usage) > 1:
            counts = list(gpu_usage.values())
            max_count = max(counts)
            min_count = min(counts)
            if max_count > min_count * 2:
                logger.info("  🔧 GPU使用不均衡，建议:")
                logger.info("    - 改进负载均衡算法")
                logger.info("    - 检查GPU性能差异")
                logger.info("    - 调整任务分配策略")
        
        # 总体建议
        logger.info("  📈 总体优化方向:")
        logger.info("    - 监控GPU内存使用情况")
        logger.info("    - 优化模型加载和推理速度")
        logger.info("    - 调整队列大小和超时设置")
        logger.info("    - 考虑使用更快的存储设备")

async def main():
    """主函数"""
    optimizer = ParallelPerformanceOptimizer()
    
    # 测试不同规模的并发
    test_scenarios = [
        (10, 5),   # 10个请求，批次大小5
        (20, 5),   # 20个请求，批次大小5
        (30, 10),  # 30个请求，批次大小10
    ]
    
    for num_requests, batch_size in test_scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 测试场景: {num_requests}个请求，批次大小{batch_size}")
        logger.info(f"{'='*60}")
        
        await optimizer.test_parallel_performance(num_requests, batch_size)
        
        # 等待一段时间再进行下一个测试
        if num_requests != test_scenarios[-1][0]:
            logger.info("⏳ 等待30秒后进行下一个测试...")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main()) 