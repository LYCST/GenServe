#!/usr/bin/env python3
"""
GPU隔离并发测试脚本
测试使用CUDA_VISIBLE_DEVICES进行GPU隔离的并发生成
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any

async def test_single_request(session: aiohttp.ClientSession, prompt: str, request_id: int) -> Dict[str, Any]:
    """测试单个请求"""
    start_time = time.time()
    
    data = {
        "model_id": "flux1-dev",
        "prompt": f"{prompt} (request {request_id})",
        "num_inference_steps": 20,  # 减少步数以加快测试
        "seed": request_id,
        "height": 512,  # 减小尺寸以加快测试
        "width": 512
    }
    
    try:
        async with session.post("http://localhost:12411/generate", json=data) as response:
            result = await response.json()
            elapsed = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": result.get("success", False),
                "elapsed_time": elapsed,
                "device": result.get("device", "unknown"),
                "instance_id": result.get("instance_id", "unknown"),
                "cuda_visible_devices": result.get("cuda_visible_devices", "unknown"),
                "error": result.get("error", None)
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "request_id": request_id,
            "success": False,
            "elapsed_time": elapsed,
            "error": str(e)
        }

async def test_concurrent_requests(num_requests: int = 8, prompt: str = "a beautiful cat"):
    """测试并发请求"""
    print(f"🚀 开始测试 {num_requests} 个并发请求...")
    
    async with aiohttp.ClientSession() as session:
        # 创建并发任务
        tasks = [
            test_single_request(session, prompt, i+1) 
            for i in range(num_requests)
        ]
        
        # 并发执行
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_elapsed = time.time() - start_time
        
        # 分析结果
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"\n📊 测试结果:")
        print(f"总耗时: {total_elapsed:.2f}秒")
        print(f"成功请求: {len(successful)}/{num_requests}")
        print(f"失败请求: {len(failed)}")
        
        if successful:
            avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
            print(f"平均单个请求耗时: {avg_time:.2f}秒")
            
            # 统计使用的设备
            device_usage = {}
            for r in successful:
                device = r["device"]
                if device not in device_usage:
                    device_usage[device] = 0
                device_usage[device] += 1
            
            print(f"\n🎯 设备使用统计:")
            for device, count in device_usage.items():
                print(f"  {device}: {count} 个请求")
            
            print(f"\n🔧 GPU隔离详情:")
            for r in successful:
                print(f"  请求{r['request_id']}: {r['device']} (实例: {r['instance_id']}, CUDA_VISIBLE_DEVICES={r['cuda_visible_devices']})")
        
        if failed:
            print(f"\n❌ 失败请求详情:")
            for r in failed:
                print(f"  请求{r['request_id']}: {r['error']}")

async def test_service_status():
    """测试服务状态"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:12411/status") as response:
                if response.status == 200:
                    status = await response.json()
                    print("✅ 服务状态正常")
                    print(f"总实例数: {status.get('total_instances', 0)}")
                    print(f"忙碌实例数: {status.get('busy_instances', 0)}")
                    return True
                else:
                    print(f"❌ 服务状态异常: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return False

async def main():
    """主函数"""
    print("🔧 GPU隔离并发测试")
    print("=" * 50)
    
    # 检查服务状态
    if not await test_service_status():
        print("请先启动GenServe服务: python main.py")
        return
    
    print("\n" + "=" * 50)
    
    # 测试少量并发
    await test_concurrent_requests(4, "a cute dog playing in the park")
    
    print("\n" + "=" * 50)
    
    # 测试更多并发
    await test_concurrent_requests(8, "a futuristic city with flying cars")

if __name__ == "__main__":
    asyncio.run(main()) 