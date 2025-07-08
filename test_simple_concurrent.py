#!/usr/bin/env python3
"""
简单的并发测试脚本 - 使用curl命令测试真正的并发
"""

import subprocess
import time
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_single_request(request_id: int, base_url: str = "http://localhost:12411") -> dict:
    """发送单个curl请求"""
    try:
        # 构建请求数据
        data = {
            "prompt": f"Beautiful landscape with mountains and trees, request {request_id}",
            "model_id": "flux1-dev",
            "height": 512,
            "width": 512,
            "num_inference_steps": 10,
            "cfg": 3.5,
            "seed": random.randint(1, 1000000),
            "priority": random.randint(0, 3),
            "mode": "text2img"
        }
        
        # 转换为JSON字符串
        json_data = json.dumps(data)
        
        # 记录开始时间
        start_time = time.time()
        logger.info(f"🚀 请求 {request_id}: 开始发送curl请求")
        
        # 执行curl命令
        cmd = [
            "curl", "-X", "POST",
            f"{base_url}/generate",
            "-H", "Content-Type: application/json",
            "-d", json_data,
            "-s"  # 静默模式
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        if result.returncode == 0:
            try:
                response_json = json.loads(result.stdout)
                success = response_json.get("success", False)
                task_id = response_json.get("task_id", "")
                gpu_id = response_json.get("gpu_id", "")
                
                logger.info(f"✅ 请求 {request_id}: 成功，GPU: {gpu_id}, 耗时: {response_time:.2f}s")
                
                return {
                    "request_id": request_id,
                    "success": success,
                    "response_time": response_time,
                    "task_id": task_id,
                    "gpu_id": gpu_id,
                    "status_code": 200
                }
            except json.JSONDecodeError:
                logger.error(f"❌ 请求 {request_id}: JSON解析失败")
                return {
                    "request_id": request_id,
                    "success": False,
                    "response_time": response_time,
                    "error": "JSON解析失败",
                    "status_code": 0
                }
        else:
            logger.error(f"❌ 请求 {request_id}: curl失败，返回码: {result.returncode}")
            return {
                "request_id": request_id,
                "success": False,
                "response_time": response_time,
                "error": f"curl失败: {result.stderr}",
                "status_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ 请求 {request_id}: 超时")
        return {
            "request_id": request_id,
            "success": False,
            "response_time": 300,
            "error": "请求超时",
            "status_code": 0
        }
    except Exception as e:
        logger.error(f"❌ 请求 {request_id}: 异常: {e}")
        return {
            "request_id": request_id,
            "success": False,
            "response_time": 0,
            "error": str(e),
            "status_code": 0
        }

def test_concurrent_requests(base_url: str = "http://localhost:12411", total_requests: int = 5):
    """测试并发请求"""
    logger.info(f"🎯 开始并发测试: {total_requests} 个请求")
    logger.info(f"🌐 目标URL: {base_url}")
    
    # 记录总体开始时间
    overall_start = time.time()
    
    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=total_requests) as executor:
        # 提交所有任务
        future_to_request = {
            executor.submit(send_single_request, i+1, base_url): i+1 
            for i in range(total_requests)
        }
        
        # 收集结果
        results = []
        for future in future_to_request:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                request_id = future_to_request[future]
                logger.error(f"❌ 请求 {request_id} 执行异常: {e}")
                results.append({
                    "request_id": request_id,
                    "success": False,
                    "error": str(e)
                })
    
    # 计算总体耗时
    overall_time = time.time() - overall_start
    
    # 分析结果
    analyze_results(results, overall_time)

def analyze_results(results: list, overall_time: float):
    """分析测试结果"""
    logger.info("=" * 60)
    logger.info("📊 并发测试结果分析")
    logger.info("=" * 60)
    
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.get("success", False))
    failed_requests = total_requests - successful_requests
    
    logger.info(f"总请求数: {total_requests}")
    logger.info(f"成功请求: {successful_requests}")
    logger.info(f"失败请求: {failed_requests}")
    logger.info(f"总耗时: {overall_time:.2f}s")
    
    if successful_requests > 0:
        # 响应时间统计
        response_times = [r.get("response_time", 0) for r in results if r.get("success", False)]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        logger.info(f"平均响应时间: {avg_response_time:.2f}s")
        logger.info(f"最大响应时间: {max_response_time:.2f}s")
        logger.info(f"最小响应时间: {min_response_time:.2f}s")
        
        # GPU使用统计
        gpu_usage = {}
        for result in results:
            if result.get("success") and result.get("gpu_id"):
                gpu_id = result["gpu_id"]
                gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
        
        logger.info("\n🎮 GPU使用情况:")
        for gpu_id, count in sorted(gpu_usage.items()):
            logger.info(f"  GPU {gpu_id}: {count} 个请求")
        
        # 并发度评估
        if overall_time < max_response_time * 1.5:
            logger.info("✅ 总体耗时接近最大响应时间，并发效果良好")
        else:
            logger.warning("⚠️ 总体耗时远大于最大响应时间，可能存在串行化")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    # 测试配置
    BASE_URL = "http://localhost:12411"
    TOTAL_REQUESTS = 5  # 减少请求数以便观察
    
    try:
        test_concurrent_requests(BASE_URL, TOTAL_REQUESTS)
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc()) 