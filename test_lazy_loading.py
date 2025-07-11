#!/usr/bin/env python3
"""
懒加载和动态模型切换测试脚本
验证模型是否按需加载和动态切换
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

async def test_lazy_loading_and_model_switching(base_url: str = "http://localhost:12411"):
    """测试懒加载和动态模型切换"""
    logger.info("🧪 开始懒加载和动态模型切换测试")
    logger.info(f"   服务地址: {base_url}")
    
    # 测试配置
    test_cases = [
        {
            "name": "首次请求 - flux1-dev",
            "model_id": "flux1-dev",
            "prompt": "A beautiful landscape with mountains, first request",
            "expected_gpu": None  # 首次请求，不知道会分配到哪个GPU
        },
        {
            "name": "相同模型 - flux1-dev",
            "model_id": "flux1-dev", 
            "prompt": "A futuristic city, same model request",
            "expected_gpu": None  # 应该复用已加载的模型
        },
        {
            "name": "不同模型 - flux1-depth-dev",
            "model_id": "flux1-depth-dev",
            "prompt": "A portrait of a person, different model request",
            "expected_gpu": None  # 应该卸载旧模型，加载新模型
        },
        {
            "name": "回到原模型 - flux1-dev",
            "model_id": "flux1-dev",
            "prompt": "A cute cat, back to original model",
            "expected_gpu": None  # 应该再次卸载和加载
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n{'='*60}")
            logger.info(f"测试 {i+1}/{len(test_cases)}: {test_case['name']}")
            logger.info(f"{'='*60}")
            
            # 构建请求数据
            data = {
                "prompt": test_case["prompt"],
                "model_id": test_case["model_id"],
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,  # 减少步数以加快测试
                "cfg": 3.5,
                "seed": random.randint(1, 1000000),
                "priority": 0,
                "mode": "text2img"
            }
            
            # 发送请求
            start_time = time.time()
            try:
                async with session.post(f"{base_url}/generate", json=data) as response:
                    response_time = time.time() - start_time
                    response_text = await response.text()
                    
                    if response.status == 200:
                        try:
                            response_json = await response.json()
                            task_id = response_json.get("task_id", "")
                            gpu_id = response_json.get("gpu_id", "")
                            model_id = response_json.get("model_id", "")
                            success = response_json.get("success", False)
                            
                            logger.info(f"✅ 请求成功")
                            logger.info(f"   任务ID: {task_id[:8]}")
                            logger.info(f"   物理GPU: {gpu_id}")
                            logger.info(f"   模型ID: {model_id}")
                            logger.info(f"   响应时间: {response_time:.2f}s")
                            logger.info(f"   成功状态: {success}")
                            
                            # 分析响应时间
                            if i == 0:
                                logger.info(f"   📊 首次加载时间: {response_time:.2f}s (包含模型加载)")
                            else:
                                logger.info(f"   📊 后续请求时间: {response_time:.2f}s")
                            
                        except Exception as e:
                            logger.error(f"❌ 响应解析失败: {e}")
                    else:
                        logger.error(f"❌ 请求失败，状态码: {response.status}")
                        logger.error(f"   响应内容: {response_text}")
                        
            except Exception as e:
                logger.error(f"❌ 请求异常: {e}")
            
            # 等待一段时间再进行下一个测试
            if i < len(test_cases) - 1:
                logger.info("⏳ 等待5秒后进行下一个测试...")
                await asyncio.sleep(5)

async def test_concurrent_model_switching(base_url: str = "http://localhost:12411"):
    """测试并发模型切换"""
    logger.info(f"\n{'='*60}")
    logger.info("🧪 开始并发模型切换测试")
    logger.info(f"{'='*60}")
    
    # 并发请求不同模型
    models = ["flux1-dev", "flux1-depth-dev", "flux1-fill-dev"]
    prompts = [
        "A beautiful sunset",
        "A portrait of a person", 
        "A futuristic city"
    ]
    
    async def single_request(session: aiohttp.ClientSession, model_id: str, prompt: str, request_id: int):
        """单个请求函数"""
        data = {
            "prompt": f"{prompt}, request {request_id}",
            "model_id": model_id,
            "height": 512,
            "width": 512,
            "num_inference_steps": 10,
            "cfg": 3.5,
            "seed": random.randint(1, 1000000),
            "priority": 0,
            "mode": "text2img"
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{base_url}/generate", json=data) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    response_json = await response.json()
                    gpu_id = response_json.get("gpu_id", "")
                    success = response_json.get("success", False)
                    
                    logger.info(f"✅ 并发请求 {request_id} 成功 (模型: {model_id}, GPU: {gpu_id}, 耗时: {response_time:.2f}s)")
                    return {
                        "success": True,
                        "model_id": model_id,
                        "gpu_id": gpu_id,
                        "response_time": response_time
                    }
                else:
                    logger.error(f"❌ 并发请求 {request_id} 失败，状态码: {response.status}")
                    return {"success": False, "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ 并发请求 {request_id} 异常: {e}")
            return {"success": False, "error": str(e)}
    
    # 创建并发任务
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(6):  # 6个并发请求
            model_id = models[i % len(models)]
            prompt = prompts[i % len(prompts)]
            task = single_request(session, model_id, prompt, i + 1)
            tasks.append(task)
        
        # 同时发送所有请求
        logger.info(f"📤 同时发送 {len(tasks)} 个并发请求...")
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 分析结果
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
        exception_results = [r for r in results if isinstance(r, Exception)]
        
        logger.info(f"\n📊 并发测试结果:")
        logger.info(f"   总请求数: {len(results)}")
        logger.info(f"   成功请求: {len(successful_results)}")
        logger.info(f"   失败请求: {len(failed_results)}")
        logger.info(f"   异常请求: {len(exception_results)}")
        logger.info(f"   总耗时: {total_time:.2f}s")
        
        # 分析GPU使用情况
        gpu_usage = {}
        for result in successful_results:
            gpu_id = result.get("gpu_id", "unknown")
            gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
        
        logger.info(f"\n物理GPU使用情况:")
        for gpu_id, count in sorted(gpu_usage.items()):
            logger.info(f"   物理GPU {gpu_id}: {count} 个请求")

async def main():
    """主函数"""
    BASE_URL = "http://localhost:12411"
    
    try:
        # 测试懒加载和模型切换
        await test_lazy_loading_and_model_switching(BASE_URL)
        
        # 等待一段时间
        logger.info("\n⏳ 等待10秒后进行并发测试...")
        await asyncio.sleep(10)
        
        # 测试并发模型切换
        await test_concurrent_model_switching(BASE_URL)
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 所有测试完成")
        logger.info(f"{'='*60}")
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main()) 