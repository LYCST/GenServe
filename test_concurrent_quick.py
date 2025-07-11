#!/usr/bin/env python3
"""
快速并发测试脚本 - 验证并发机制是否正常工作
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

async def test_concurrent_requests(base_url: str = "http://localhost:12411", total_requests: int = 10):
    """测试并发请求"""
    logger.info(f"开始快速并发测试")
    logger.info(f"   总请求数: {total_requests}")
    logger.info(f"   服务地址: {base_url}")
    
    # 创建连接器
    connector = aiohttp.TCPConnector(limit=total_requests, limit_per_host=total_requests)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 创建任务列表
        tasks = []
        start_times = []
        
        for i in range(total_requests):
            start_time = time.time()
            start_times.append(start_time)
            
            # 构建请求数据
            data = {
                "prompt": f"Beautiful landscape with mountains and trees, request {i+1}",
                "model_id": "flux1-dev",
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,  # 减少步数以加快测试
                "cfg": 3.5,
                "seed": random.randint(1, 1000000),
                "priority": random.randint(0, 3),
                "mode": "text2img"
            }
            
            # 创建异步任务
            task = test_single_request(session, i + 1, base_url, data, start_time)
            tasks.append(task)
        
        # 同时发送所有请求
        logger.info(f"📤 同时发送 {len(tasks)} 个请求...")
        overall_start = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        overall_time = time.time() - overall_start
        
        # 处理结果
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"任务执行异常: {result}")
            else:
                valid_results.append(result)
        
        logger.info(f"并发测试完成，总耗时: {overall_time:.2f}s")
        
        # 分析结果
        analyze_concurrent_results(valid_results, overall_time, start_times)

async def test_single_request(session: aiohttp.ClientSession, request_id: int, base_url: str, data: Dict, start_time: float) -> Dict[str, Any]:
    """测试单个请求 - 同步等待结果"""
    try:
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
                    logger.info(f"请求 {request_id}: 成功，物理GPU: {result['gpu_id']}, 耗时: {response_time:.2f}s")
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

def analyze_concurrent_results(results: List[Dict[str, Any]], overall_time: float, start_times: List[float]):
    """分析并发测试结果"""
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
    
    # 计算并发度
    first_start = min(start_times)
    last_start = max(start_times)
    request_spread = last_start - first_start
    
    # GPU使用统计
    gpu_usage = {}
    for result in results:
        if result.get("success") and result.get("gpu_id"):
            gpu_id = result["gpu_id"]
            gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
    
    # 打印分析结果
    logger.info("=" * 60)
    logger.info("📊 并发测试结果分析")
    logger.info("=" * 60)
    logger.info(f"总请求数: {total_requests}")
    logger.info(f"成功请求: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    logger.info(f"失败请求: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    logger.info(f"总耗时: {overall_time:.2f}s")
    logger.info(f"请求发送时间跨度: {request_spread:.2f}s")
    logger.info(f"平均响应时间: {avg_response_time:.2f}s")
    logger.info(f"最大响应时间: {max_response_time:.2f}s")
    logger.info(f"最小响应时间: {min_response_time:.2f}s")
    
    # 并发度评估
    if request_spread < 1.0:
        logger.info("✅ 请求发送时间跨度很小，并发度良好")
    else:
        logger.warning("⚠️ 请求发送时间跨度较大，可能存在串行化问题")
    
    # 响应时间分析
    if max_response_time - min_response_time < 5.0:
        logger.info("✅ 响应时间差异较小，负载均衡良好")
    else:
        logger.warning("⚠️ 响应时间差异较大，可能存在负载不均衡")
    
    logger.info("\n物理GPU使用情况:")
    for gpu_id, count in sorted(gpu_usage.items()):
        logger.info(f"  物理GPU {gpu_id}: {count} 个请求")
    
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
    
    # 并发性能评估
    if successful_requests == total_requests:
        logger.info("✅ 所有请求都成功，并发处理正常")
    else:
        logger.warning(f"⚠️ 有 {failed_requests} 个请求失败，需要检查")
    
    if overall_time < max_response_time * 1.5:
        logger.info("✅ 总体耗时接近最大响应时间，并发效果良好")
    else:
        logger.warning("⚠️ 总体耗时远大于最大响应时间，可能存在串行化")
    
    logger.info("=" * 60)

async def main():
    """主函数"""
    # 测试配置
    BASE_URL = "http://localhost:12411"  # 服务地址
    TOTAL_REQUESTS = 10  # 总请求数
    
    try:
        # 运行并发测试
        await test_concurrent_requests(BASE_URL, TOTAL_REQUESTS)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main()) 