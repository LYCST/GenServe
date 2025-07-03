#!/usr/bin/env python3
"""
GenServe 增强版并发测试脚本
测试多GPU并发处理能力和队列管理
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
import base64
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import uuid

@dataclass
class TestResult:
    """测试结果数据类"""
    request_id: str
    prompt: str
    start_time: float
    end_time: float
    success: bool
    device: str = ""
    worker: str = ""
    task_id: str = ""
    generation_time: float = 0.0
    queue_wait_time: float = 0.0
    error: str = ""

class EnhancedConcurrentTester:
    """增强版并发测试器"""
    
    def __init__(self, base_url: str = "http://localhost:12411"):
        self.base_url = base_url
        self.session = None
        self.test_results: List[TestResult] = []
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def get_service_status(self) -> Dict[str, Any]:
        """获取详细服务状态"""
        try:
            async with self.session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def display_service_info(self):
        """显示服务信息"""
        print("🔍 获取服务状态...")
        status = await self.get_service_status()
        
        if "error" in status:
            print(f"❌ 无法获取服务状态: {status['error']}")
            return False
        
        print("=" * 60)
        print("🚀 GenServe 服务状态")
        print("=" * 60)
        
        # 并发管理器状态
        concurrent_status = status.get("concurrent_manager", {})
        print(f"📊 并发管理器:")
        print(f"  运行状态: {'🟢 运行中' if concurrent_status.get('is_running') else '🔴 已停止'}")
        print(f"  全局队列: {concurrent_status.get('global_queue_size', 0)} 个任务")
        print(f"  总队列大小: {concurrent_status.get('total_queue_size', 0)} 个任务")
        print(f"  工作线程: {concurrent_status.get('worker_threads', 0)} 个")
        print(f"  忙碌实例: {concurrent_status.get('busy_instances', 0)}/{concurrent_status.get('total_instances', 0)}")
        
        # 统计信息
        stats = concurrent_status.get('stats', {})
        print(f"  总任务: {stats.get('total_tasks', 0)}")
        print(f"  已完成: {stats.get('completed_tasks', 0)}")
        print(f"  失败: {stats.get('failed_tasks', 0)}")
        print(f"  队列满拒绝: {stats.get('queue_full_rejections', 0)}")
        
        # 模型实例详情
        model_instances = concurrent_status.get('model_instances', {})
        for model_id, instances in model_instances.items():
            print(f"\n🤖 模型 {model_id}:")
            for instance in instances:
                status_icon = "🔴" if instance['is_busy'] else "🟢"
                current_task = f" (任务: {instance.get('current_task', 'None')})" if instance['is_busy'] else ""
                print(f"  {status_icon} {instance['device']}: 队列={instance['queue_size']}/{instance['max_queue_size']}, 总生成={instance['total_generations']}{current_task}")
        
        # GPU负载信息
        gpu_load = status.get("gpu_load", {})
        if gpu_load:
            print(f"\n💾 GPU内存使用:")
            for gpu_id, load_info in gpu_load.items():
                if load_info.get('available'):
                    utilization = load_info.get('utilization_percent', 0)
                    free_mb = load_info.get('free_mb', 0)
                    total_mb = load_info.get('total_mb', 0)
                    print(f"  {gpu_id}: {utilization:.1f}% 使用中, {free_mb:.0f}MB/{total_mb:.0f}MB 可用")
        
        print("=" * 60)
        return True
    
    async def generate_single_image(self, prompt: str, request_id: str, priority: int = 0) -> TestResult:
        """生成单张图片并记录详细信息"""
        payload = {
            "prompt": prompt,
            "model": "flux1-dev",
            "num_inference_steps": 20,
            "height": 512,
            "width": 512,
            "seed": abs(hash(request_id)) % 10000  # 基于request_id生成稳定的种子
        }
        
        result = TestResult(
            request_id=request_id,
            prompt=prompt,
            start_time=time.time(),
            end_time=0,
            success=False
        )
        
        try:
            async with self.session.post(
                f"{self.base_url}/generate", 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                
                result.end_time = time.time()
                
                if response.status == 200:
                    data = await response.json()
                    
                    result.success = True
                    result.device = data.get("device", "unknown")
                    result.worker = data.get("worker", "unknown")
                    result.task_id = data.get("task_id", "")
                    
                    # 解析生成时间
                    elapsed_str = data.get("elapsed_time", "0s")
                    if elapsed_str.endswith('s'):
                        try:
                            result.generation_time = float(elapsed_str[:-1])
                        except:
                            result.generation_time = 0.0
                    
                    # 计算队列等待时间
                    total_time = result.end_time - result.start_time
                    result.queue_wait_time = max(0, total_time - result.generation_time)
                    
                    # 保存图片（可选）
                    if data.get("output") and os.getenv("SAVE_TEST_IMAGES", "false").lower() == "true":
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f"test_{request_id}_{timestamp}.png"
                        
                        image_data = base64.b64decode(data["output"])
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                
                else:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text}"
                    
        except asyncio.TimeoutError:
            result.end_time = time.time()
            result.error = "请求超时"
        except Exception as e:
            result.end_time = time.time()
            result.error = str(e)
        
        return result
    
    async def burst_test(self, prompts: List[str], burst_size: int, burst_interval: float = 0) -> List[TestResult]:
        """突发测试 - 快速发送多个请求"""
        print(f"\n💥 突发测试: 发送 {burst_size} 个请求")
        
        tasks = []
        for i in range(burst_size):
            prompt = prompts[i % len(prompts)]
            request_id = f"burst_{i+1}_{uuid.uuid4().hex[:8]}"
            task = self.generate_single_image(prompt, request_id)
            tasks.append(task)
            
            # 突发间隔
            if burst_interval > 0 and i < burst_size - 1:
                await asyncio.sleep(burst_interval)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常
        valid_results = [r for r in results if isinstance(r, TestResult)]
        self.test_results.extend(valid_results)
        
        return valid_results
    
    async def sustained_load_test(self, prompts: List[str], duration: int, rate: float) -> List[TestResult]:
        """持续负载测试"""
        print(f"\n⏱️ 持续负载测试: {duration}秒, 每秒{rate}个请求")
        
        end_time = time.time() + duration
        results = []
        request_count = 0
        
        while time.time() < end_time:
            # 计算下一批请求数量
            interval = 1.0 / rate
            batch_size = max(1, int(rate))
            
            # 发送一批请求
            tasks = []
            for i in range(batch_size):
                if time.time() >= end_time:
                    break
                
                prompt = prompts[request_count % len(prompts)]
                request_id = f"sustained_{request_count+1}_{uuid.uuid4().hex[:8]}"
                task = self.generate_single_image(prompt, request_id)
                tasks.append(task)
                request_count += 1
            
            # 等待一段时间
            if tasks:
                # 不等待任务完成，继续发送
                asyncio.create_task(self._collect_results(tasks, results))
            
            await asyncio.sleep(interval)
        
        # 等待剩余任务完成
        print(f"⏳ 等待剩余任务完成...")
        await asyncio.sleep(30)  # 等待30秒让任务完成
        
        self.test_results.extend(results)
        return results
    
    async def _collect_results(self, tasks: List, results: List[TestResult]):
        """收集任务结果"""
        try:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in task_results:
                if isinstance(result, TestResult):
                    results.append(result)
        except Exception as e:
            print(f"收集结果时出错: {e}")
    
    def analyze_results(self, results: List[TestResult], test_name: str = "测试"):
        """分析测试结果"""
        if not results:
            print(f"❌ {test_name}: 没有结果数据")
            return
        
        print(f"\n📊 {test_name} 结果分析:")
        print("=" * 60)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✅ 成功: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"❌ 失败: {len(failed)}")
        
        if successful:
            # 时间统计
            total_times = [r.end_time - r.start_time for r in successful]
            generation_times = [r.generation_time for r in successful if r.generation_time > 0]
            queue_wait_times = [r.queue_wait_time for r in successful if r.queue_wait_time > 0]
            
            print(f"\n⏱️ 时间统计:")
            print(f"  总时间: 平均 {np.mean(total_times):.2f}s, 中位数 {np.median(total_times):.2f}s")
            if generation_times:
                print(f"  生成时间: 平均 {np.mean(generation_times):.2f}s, 中位数 {np.median(generation_times):.2f}s")
            if queue_wait_times:
                print(f"  队列等待: 平均 {np.mean(queue_wait_times):.2f}s, 中位数 {np.median(queue_wait_times):.2f}s")
            
            # 设备分布
            device_counts = {}
            worker_counts = {}
            for r in successful:
                device_counts[r.device] = device_counts.get(r.device, 0) + 1
                worker_counts[r.worker] = worker_counts.get(r.worker, 0) + 1
            
            print(f"\n🎯 设备分布:")
            for device, count in sorted(device_counts.items()):
                print(f"  {device}: {count} 次 ({count/len(successful)*100:.1f}%)")
            
            print(f"\n👷 工作线程分布:")
            for worker, count in sorted(worker_counts.items()):
                print(f"  {worker}: {count} 次 ({count/len(successful)*100:.1f}%)")
        
        if failed:
            print(f"\n❌ 失败原因:")
            error_counts = {}
            for r in failed:
                error_type = r.error.split(':')[0] if ':' in r.error else r.error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error, count in sorted(error_counts.items()):
                print(f"  {error}: {count} 次")
    
    def create_performance_chart(self, results: List[TestResult], save_path: str = "performance_chart.png"):
        """创建性能图表"""
        if not results:
            return
        
        successful = [r for r in results if r.success]
        if not successful:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GenServe 并发性能分析', fontsize=16)
        
        # 1. 时间线图
        start_times = [(r.start_time - successful[0].start_time) for r in successful]
        total_times = [r.end_time - r.start_time for r in successful]
        
        ax1.scatter(start_times, total_times, alpha=0.6)
        ax1.set_xlabel('请求开始时间 (秒)')
        ax1.set_ylabel('总处理时间 (秒)')
        ax1.set_title('请求处理时间分布')
        ax1.grid(True)
        
        # 2. 设备负载分布
        devices = [r.device for r in successful]
        device_counts = {d: devices.count(d) for d in set(devices)}
        
        ax2.bar(device_counts.keys(), device_counts.values())
        ax2.set_xlabel('设备')
        ax2.set_ylabel('任务数量')
        ax2.set_title('设备负载分布')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 响应时间直方图
        ax3.hist(total_times, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('总处理时间 (秒)')
        ax3.set_ylabel('频次')
        ax3.set_title('响应时间分布')
        ax3.grid(True)
        
        # 4. 队列等待时间
        queue_times = [r.queue_wait_time for r in successful if r.queue_wait_time > 0]
        if queue_times:
            ax4.hist(queue_times, bins=20, alpha=0.7, edgecolor='black', color='orange')
            ax4.set_xlabel('队列等待时间 (秒)')
            ax4.set_ylabel('频次')
            ax4.set_title('队列等待时间分布')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, '无队列等待数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('队列等待时间分布')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 性能图表已保存: {save_path}")

async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 GenServe 增强版并发测试")
    print("=" * 60)
    
    # 测试提示词
    test_prompts = [
        "a serene mountain landscape with a crystal clear lake reflecting the sky",
        "a playful golden retriever running through a field of sunflowers",
    ]
    
    async with EnhancedConcurrentTester() as tester:
        # 1. 显示服务信息
        print("\n1. 📋 检查服务状态...")
        if not await tester.display_service_info():
            print("❌ 服务未正常运行，请先启动服务")
            return
        
        # 2. 突发测试
        print("\n2. 💥 突发测试...")
        burst_results = await tester.burst_test(test_prompts, 2)  # 6个并发请求
        tester.analyze_results(burst_results, "突发测试")
        
        # # 等待服务稳定
        # await asyncio.sleep(5)
        
        # # 3. 持续负载测试
        # print("\n3. ⏱️ 持续负载测试...")
        # load_results = await tester.sustained_load_test(test_prompts, 60, 0.5)  # 60秒，每秒0.5个请求
        # tester.analyze_results(load_results, "持续负载测试")
        
        # # 4. 生成性能报告
        # print("\n4. 📊 生成性能报告...")
        # all_results = tester.test_results
        # tester.analyze_results(all_results, "综合测试")
        
        # # 创建图表（如果安装了matplotlib）
        # try:
        #     tester.create_performance_chart(all_results)
        # except ImportError:
        #     print("📈 要生成性能图表，请安装matplotlib: pip install matplotlib")
        # except Exception as e:
        #     print(f"📈 生成图表时出错: {e}")
        
        # # 5. 最终状态检查
        # print("\n5. 🔍 最终状态检查...")
        # await tester.display_service_info()
    
    print(f"\n✨ 测试完成！")
    print(f"📁 如果启用了图片保存（SAVE_TEST_IMAGES=true），图片将保存在当前目录")

if __name__ == "__main__":
    asyncio.run(main())