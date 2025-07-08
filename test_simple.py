#!/usr/bin/env python3
"""
简化并发测试脚本 - 快速测试GenServe的排队机制
"""

import asyncio
import aiohttp
import time
import json
import random
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_request(session: aiohttp.ClientSession, request_id: int, base_url: str) -> Dict[str, Any]:
    """测试单个请求"""
    start_time = time.time()
    
    try:
        # 构建请求数据
        data = {
            "prompt": f"Beautiful landscape with mountains and trees, request {request_id}",
            "model_id": "flux1-dev",
            "height": 512,
            "width": 512,
            "num_inference_steps": 10,  # 减少步数以加快测试
            "cfg": 3.5,
            "seed": random.randint(1, 1000000),
            "priority": random.randint(0, 3),  # 随机优先级
            "mode": "text2img"
        }
        
        logger.info(f"请求 {request_id}: 开始发送")
        
        async with session.post(f"{base_url}/generate", json=data) as response:
            response_time = time.time() - start_time
            response_text = await response.text()
            
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

async def run_concurrent_test(base_url: str = "http://localhost:12411", total_requests: int = 20, concurrent_limit: int = 5):
    """运行并发测试"""
    logger.info(f"🚀 开始并发测试")
    logger.info(f"   总请求数: {total_requests}")
    logger.info(f"   并发限制: {concurrent_limit}")
    logger.info(f"   服务地址: {base_url}")
    
    # 创建连接器
    connector = aiohttp.TCPConnector(limit=concurrent_limit, limit_per_host=concurrent_limit)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 创建任务列表
        tasks = []
        for i in range(total_requests):
            task = test_single_request(session, i + 1, base_url)
            tasks.append(task)
        
        # 执行所有任务
        logger.info(f"📤 开始发送 {len(tasks)} 个并发请求...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 处理结果
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"任务执行异常: {result}")
            else:
                valid_results.append(result)
        
        logger.info(f"✅ 并发测试完成，总耗时: {total_time:.2f}s")
        
        # 分析结果
        analyze_results(valid_results, total_time)

def analyze_results(results: List[Dict[str, Any]], total_time: float):
    """分析测试结果"""
    if not results:
        logger.warning("没有测试结果可分析")
        return
    
    # 基础统计
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.get("success", False))
    failed_requests = total_requests - successful_requests
    
    # 响应时间统计
    response_times = [r.get("response_time", 0) for r in results if r.get("success", False)]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    max_response_time = max(response_times) if response_times else 0
    min_response_time = min(response_times) if response_times else 0
    
    # GPU使用统计
    gpu_usage = {}
    for result in results:
        if result.get("success") and result.get("gpu_id"):
            gpu_id = result["gpu_id"]
            gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
    
    # 打印分析结果
    logger.info("=" * 50)
    logger.info("📊 测试结果分析")
    logger.info("=" * 50)
    logger.info(f"总请求数: {total_requests}")
    logger.info(f"成功请求: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    logger.info(f"失败请求: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    logger.info(f"总耗时: {total_time:.2f}s")
    logger.info(f"平均响应时间: {avg_response_time:.2f}s")
    logger.info(f"最大响应时间: {max_response_time:.2f}s")
    logger.info(f"最小响应时间: {min_response_time:.2f}s")
    
    logger.info("\n🎮 GPU使用情况:")
    for gpu_id, count in sorted(gpu_usage.items()):
        logger.info(f"  GPU {gpu_id}: {count} 个请求")
    
    # 检查负载均衡
    if len(gpu_usage) > 1:
        gpu_counts = list(gpu_usage.values())
        max_gpu_count = max(gpu_counts)
        min_gpu_count = min(gpu_counts)
        balance_ratio = min_gpu_count / max_gpu_count if max_gpu_count > 0 else 1.0
        logger.info(f"\n⚖️ 负载均衡评估: {balance_ratio:.2f} (1.0为完美均衡)")
        
        if balance_ratio < 0.5:
            logger.warning("⚠️ 负载均衡效果较差")
        elif balance_ratio < 0.8:
            logger.info("📊 负载均衡效果一般")
        else:
            logger.info("✅ 负载均衡效果良好")
    
    logger.info("=" * 50)

async def main():
    """主函数"""
    # 测试配置
    BASE_URL = "http://localhost:12411"  # 服务地址
    TOTAL_REQUESTS = 20  # 总请求数
    CONCURRENT_LIMIT = 5  # 并发限制
    
    try:
        # 运行并发测试
        await run_concurrent_test(BASE_URL, TOTAL_REQUESTS, CONCURRENT_LIMIT)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main()) 