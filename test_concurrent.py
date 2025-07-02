#!/usr/bin/env python3
"""
GenServe 并发测试脚本
同时发送多个请求测试多GPU并发处理能力
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
import base64
import os
from typing import List, Dict, Any

class ConcurrentTester:
    """并发测试器"""
    
    def __init__(self, base_url: str = "http://localhost:12411"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    print("🟢 服务健康状态:")
                    print(f"  状态: {result.get('status')}")
                    print(f"  GPU可用: {result.get('gpu_available')}")
                    
                    concurrent_status = result.get('concurrent_status', {})
                    print(f"  工作线程: {concurrent_status.get('worker_threads')}")
                    print(f"  队列大小: {concurrent_status.get('queue_size')}")
                    
                    # 显示模型实例信息
                    model_instances = concurrent_status.get('model_instances', {})
                    for model_id, instances in model_instances.items():
                        print(f"  模型 {model_id}:")
                        for instance in instances:
                            status_icon = "🟢" if not instance['is_busy'] else "🔴"
                            print(f"    {status_icon} {instance['device']}: 忙碌={instance['is_busy']}, 总生成数={instance['total_generations']}")
                    
                    return True
                else:
                    print(f"❌ 健康检查失败: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 健康检查错误: {e}")
            return False
    
    async def generate_single_image(self, prompt: str, request_id: int) -> Dict[str, Any]:
        """生成单张图片"""
        payload = {
            "prompt": prompt,
            "model": "flux1-dev",
            "num_inference_steps": 20,
            "height": 1024,
            "width": 1024,
            "seed": 42 + request_id  # 不同的种子
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/generate", 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
            ) as response:
                
                elapsed_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    # 保存图片
                    image_path = None
                    if result.get("output"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f"concurrent_test_{request_id}_{timestamp}.png"
                        
                        image_data = base64.b64decode(result["output"])
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                    
                    return {
                        "success": True,
                        "request_id": request_id,
                        "prompt": prompt,
                        "device": result.get("device"),
                        "worker": result.get("worker"),
                        "task_id": result.get("task_id"),
                        "generation_time": result.get("elapsed_time", "unknown"),
                        "total_time": elapsed_time,
                        "image_path": image_path
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "request_id": request_id,
                        "error": f"HTTP {response.status}: {error_text}",
                        "total_time": elapsed_time
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "request_id": request_id,
                "error": "请求超时",
                "total_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def concurrent_test(self, prompts: List[str], concurrent_count: int = 4):
        """并发测试"""
        print(f"\n🚀 开始并发测试，同时发送 {concurrent_count} 个请求")
        print(f"测试提示词: {prompts}")
        
        # 创建并发任务
        tasks = []
        for i in range(concurrent_count):
            prompt = prompts[i % len(prompts)]  # 循环使用提示词
            task = self.generate_single_image(prompt, i + 1)
            tasks.append(task)
        
        # 记录开始时间
        start_time = time.time()
        
        # 并发执行所有任务
        print(f"⏱️  开始时间: {datetime.now().strftime('%H:%M:%S')}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 记录结束时间
        total_elapsed = time.time() - start_time
        print(f"⏱️  结束时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  总耗时: {total_elapsed:.2f}秒")
        
        # 分析结果
        self.analyze_results(results, total_elapsed)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]], total_elapsed: float):
        """分析测试结果"""
        print(f"\n📊 测试结果分析:")
        print("=" * 60)
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get("success")]
        exception_results = [r for r in results if not isinstance(r, dict)]
        
        print(f"✅ 成功: {len(successful_results)}")
        print(f"❌ 失败: {len(failed_results)}")
        print(f"💥 异常: {len(exception_results)}")
        
        if successful_results:
            print(f"\n🎯 成功请求详情:")
            devices_used = {}
            workers_used = {}
            generation_times = []
            
            for result in successful_results:
                req_id = result["request_id"]
                device = result.get("device", "unknown")
                worker = result.get("worker", "unknown")
                gen_time = result.get("generation_time", "unknown")
                total_time = result.get("total_time", 0)
                
                print(f"  请求 {req_id}: 设备={device}, 工作线程={worker}, 生成耗时={gen_time}, 总耗时={total_time:.2f}s")
                
                # 统计设备使用情况
                devices_used[device] = devices_used.get(device, 0) + 1
                workers_used[worker] = workers_used.get(worker, 0) + 1
                
                if isinstance(gen_time, str) and gen_time.endswith('s'):
                    try:
                        generation_times.append(float(gen_time[:-1]))
                    except:
                        pass
            
            print(f"\n📈 设备使用统计:")
            for device, count in devices_used.items():
                print(f"  {device}: {count} 次")
            
            print(f"\n👷 工作线程统计:")
            for worker, count in workers_used.items():
                print(f"  {worker}: {count} 次")
            
            if generation_times:
                avg_gen_time = sum(generation_times) / len(generation_times)
                print(f"\n⏱️  平均生成时间: {avg_gen_time:.2f}秒")
                print(f"⏱️  并发效率: {(avg_gen_time * len(successful_results)) / total_elapsed:.2f}x")
        
        if failed_results:
            print(f"\n❌ 失败请求详情:")
            for result in failed_results:
                req_id = result["request_id"]
                error = result.get("error", "未知错误")
                print(f"  请求 {req_id}: {error}")
        
        if exception_results:
            print(f"\n💥 异常详情:")
            for i, exc in enumerate(exception_results):
                print(f"  异常 {i+1}: {exc}")

async def main():
    """主函数"""
    print("=" * 60)
    print("GenServe 并发测试")
    print("=" * 60)
    
    # 测试提示词
    test_prompts = [
        "a beautiful landscape with mountains and lakes",
        "a cute cat sitting on a chair",
        "a futuristic city with flying cars",
        "a peaceful forest with sunlight filtering through trees",
        "an underwater scene with colorful fish",
        "a majestic eagle soaring in the sky"
    ]
    
    async with ConcurrentTester() as tester:
        # 1. 检查服务状态
        print("\n1. 检查服务状态...")
        if not await tester.check_health():
            print("服务未正常运行，请先启动服务")
            return
        
        # 2. 并发测试
        print("\n2. 执行并发测试...")
        
        # 测试不同的并发级别
        concurrent_levels = [2, 3, 4]  # 减少并发级别，避免过载
        
        for level in concurrent_levels:
            print(f"\n{'='*40}")
            print(f"测试并发级别: {level}")
            print(f"{'='*40}")
            
            results = await tester.concurrent_test(test_prompts, level)
            
            # 等待一段时间再进行下一轮测试
            if level != concurrent_levels[-1]:
                print(f"\n⏳ 等待 10 秒后进行下一轮测试...")
                await asyncio.sleep(10)
    
    print(f"\n✨ 所有测试完成！")
    print(f"生成的图片保存在当前目录中，文件名格式: concurrent_test_<request_id>_<timestamp>.png")

if __name__ == "__main__":
    asyncio.run(main()) 