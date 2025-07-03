#!/usr/bin/env python3
"""
GenServe 增强版并发测试脚本
测试多GPU并发处理能力和队列管理
"""

import asyncio
import time
import json
from datetime import datetime
import base64
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid
import urllib.request
import urllib.parse
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.test_results: List[TestResult] = []
        self.session_lock = threading.Lock()
    
    def make_request(self, url: str, method: str = "GET", data: Optional[Dict] = None, timeout: int = 300) -> Dict[str, Any]:
        """发送HTTP请求"""
        try:
            if data:
                data_bytes = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(url, data=data_bytes, method=method)
                req.add_header('Content-Type', 'application/json')
            else:
                req = urllib.request.Request(url, method=method)
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = response.read().decode('utf-8')
                return json.loads(response_data)
        except urllib.error.HTTPError as e:
            error_data = e.read().decode('utf-8')
            return {"error": f"HTTP {e.code}: {error_data}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取详细服务状态"""
        return self.make_request(f"{self.base_url}/status")
    
    def display_service_info(self):
        """显示服务信息"""
        print("🔍 获取服务状态...")
        status = self.get_service_status()
        
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
            if isinstance(instances, list):
                for instance in instances:
                    if isinstance(instance, dict):
                        status_icon = "🔴" if instance.get('is_busy', False) else "🟢"
                        current_task = f" (任务: {instance.get('current_task', 'None')})" if instance.get('is_busy', False) else ""
                        print(f"  {status_icon} {instance.get('device', 'unknown')}: 队列={instance.get('queue_size', 0)}/{instance.get('max_queue_size', 0)}, 总生成={instance.get('total_generations', 0)}{current_task}")
                    else:
                        print(f"  ⚠️ 实例数据格式错误: {instance}")
            else:
                print(f"  ⚠️ 实例列表格式错误: {instances}")
        
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
    
    def generate_single_image(self, prompt: str, request_id: str, priority: int = 0) -> TestResult:
        """生成单张图片并记录详细信息"""
        payload = {
            "prompt": prompt,
            "model_id": "flux1-dev",  # 修正字段名
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
            response_data = self.make_request(
                f"{self.base_url}/generate", 
                method="POST",
                data=payload,
                timeout=300
            )
            
            result.end_time = time.time()
            
            # 检查响应是否成功
            if response_data.get("success", False):
                result.success = True
                result.device = response_data.get("gpu_id", "unknown")  # 映射 gpu_id -> device
                result.worker = response_data.get("model_id", "unknown")  # 映射 model_id -> worker
                result.task_id = response_data.get("task_id", "")
                
                # 解析生成时间 - elapsed_time 是 float 类型
                elapsed_time = response_data.get("elapsed_time", 0.0)
                if isinstance(elapsed_time, (int, float)):
                    result.generation_time = float(elapsed_time)
                else:
                    result.generation_time = 0.0
                
                # 计算队列等待时间
                total_time = result.end_time - result.start_time
                result.queue_wait_time = max(0, total_time - result.generation_time)
                
                # 保存图片（可选）- 映射 image_base64 -> output
                if response_data.get("image_base64") and os.getenv("SAVE_TEST_IMAGES", "false").lower() == "true":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"test_{request_id}_{timestamp}.png"
                    
                    image_data = base64.b64decode(response_data["image_base64"])
                    with open(image_path, "wb") as f:
                        f.write(image_data)
            else:
                # 处理失败情况
                result.error = response_data.get("error", "未知错误")
                
        except Exception as e:
            result.end_time = time.time()
            result.error = str(e)
        
        return result
    
    def burst_test(self, prompts: List[str], burst_size: int, max_workers: int = 10) -> List[TestResult]:
        """突发测试 - 使用线程池并发发送多个请求"""
        print(f"\n💥 突发测试: 发送 {burst_size} 个请求 (最大并发: {max_workers})")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_request = {}
            for i in range(burst_size):
                prompt = prompts[i % len(prompts)]
                request_id = f"burst_{i+1}_{uuid.uuid4().hex[:8]}"
                future = executor.submit(self.generate_single_image, prompt, request_id)
                future_to_request[future] = request_id
            
            # 收集结果，设置超时
            print(f"⏳ 等待 {burst_size} 个请求完成...")
            timeout_start = time.time()
            timeout_duration = 120.0  # 2分钟总体超时
            
            try:
                for future in as_completed(future_to_request):
                    # 检查是否超时
                    if time.time() - timeout_start > timeout_duration:
                        print(f"⚠️ 突发测试超时，跳过剩余 {len(future_to_request)} 个任务")
                        break
                        
                    try:
                        result = future.result(timeout=10.0)  # 单个任务10秒超时
                        print(result)   
                        results.append(result)
                        print(f"✅ 请求 {future_to_request[future]} 完成: {'成功' if result.success else '失败'}")
                    except Exception as e:
                        print(f"❌ 请求 {future_to_request[future]} 异常: {e}")
                        # 即使异常也要记录失败结果
                        failed_result = TestResult(
                            request_id=future_to_request[future],
                            prompt="",
                            start_time=time.time(),
                            end_time=time.time(),
                            success=False,
                            error=str(e)
                        )
                        results.append(failed_result)
                        
            except KeyboardInterrupt:
                print(f"\n⚠️ 测试被用户中断")
                # 记录剩余任务为失败
                for future in future_to_request:
                    if future not in [r.request_id for r in results]:
                        failed_result = TestResult(
                            request_id=future_to_request[future],
                            prompt="",
                            start_time=time.time(),
                            end_time=time.time(),
                            success=False,
                            error="测试被中断"
                        )
                        results.append(failed_result)
        
        print(f"📊 突发测试统计: 提交 {burst_size} 个请求，完成 {len(results)} 个结果")
        self.test_results.extend(results)
        return results
    
    def sustained_load_test(self, prompts: List[str], duration: int, rate: float, max_workers: int = 10) -> List[TestResult]:
        """持续负载测试"""
        print(f"\n⏱️ 持续负载测试: {duration}秒, 每秒{rate}个请求 (最大并发: {max_workers})")
        
        end_time = time.time() + duration
        results = []
        request_count = 0
        submitted_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_request = {}
            
            while time.time() < end_time:
                # 计算下一批请求数量
                interval = 1.0 / rate
                batch_size = max(1, int(rate))
                
                # 提交一批请求
                for i in range(batch_size):
                    if time.time() >= end_time:
                        break
                    
                    prompt = prompts[request_count % len(prompts)]
                    request_id = f"sustained_{submitted_count+1}_{uuid.uuid4().hex[:8]}"
                    future = executor.submit(self.generate_single_image, prompt, request_id)
                    future_to_request[future] = request_id
                    request_count += 1
                    submitted_count += 1
                
                # 收集已完成的结果
                completed_futures = []
                for future in list(future_to_request.keys()):
                    if future.done():
                        try:
                            # 设置超时，避免卡住
                            result = future.result(timeout=10.0)  # 10秒超时
                            print(result)   
                            results.append(result)
                            print(f"✅ 请求 {future_to_request[future]} 完成: {'成功' if result.success else '失败'}")
                        except Exception as e:
                            print(f"❌ 请求 {future_to_request[future]} 异常: {e}")
                        completed_futures.append(future)
                
                # 清理已完成的任务
                for future in completed_futures:
                    del future_to_request[future]
                
                # 等待一段时间
                time.sleep(interval)
            
            # 等待剩余任务完成，设置总体超时
            remaining_count = len(future_to_request)
            if remaining_count > 0:
                print(f"⏳ 等待剩余 {remaining_count} 个任务完成...")
                
                # 设置总体超时时间
                timeout_start = time.time()
                timeout_duration = 60.0  # 60秒总体超时
                
                try:
                    for future in as_completed(future_to_request):
                        # 检查是否超时
                        if time.time() - timeout_start > timeout_duration:
                            print(f"⚠️ 等待剩余任务超时，跳过剩余 {len(future_to_request)} 个任务")
                            break
                            
                        try:
                            result = future.result(timeout=5.0)  # 单个任务5秒超时
                            print(result)   
                            results.append(result)
                            print(f"✅ 请求 {future_to_request[future]} 完成: {'成功' if result.success else '失败'}")
                        except Exception as e:
                            print(f"❌ 请求 {future_to_request[future]} 异常: {e}")
                            # 即使异常也要记录失败结果
                            failed_result = TestResult(
                                request_id=future_to_request[future],
                                prompt="",
                                start_time=time.time(),
                                end_time=time.time(),
                                success=False,
                                error=str(e)
                            )
                            results.append(failed_result)
                            
                except KeyboardInterrupt:
                    print(f"\n⚠️ 测试被用户中断")
                    # 记录剩余任务为失败
                    for future in future_to_request:
                        if future not in [r.request_id for r in results]:
                            failed_result = TestResult(
                                request_id=future_to_request[future],
                                prompt="",
                                start_time=time.time(),
                                end_time=time.time(),
                                success=False,
                                error="测试被中断"
                            )
                            results.append(failed_result)
        
        print(f"📊 持续负载测试统计: 提交 {submitted_count} 个请求，完成 {len(results)} 个结果")
        self.test_results.extend(results)
        return results
    
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
            print(f"  总时间: 平均 {sum(total_times)/len(total_times):.2f}s, 中位数 {sorted(total_times)[len(total_times)//2]:.2f}s")
            if generation_times:
                print(f"  生成时间: 平均 {sum(generation_times)/len(generation_times):.2f}s, 中位数 {sorted(generation_times)[len(generation_times)//2]:.2f}s")
            if queue_wait_times:
                print(f"  队列等待: 平均 {sum(queue_wait_times)/len(queue_wait_times):.2f}s, 中位数 {sorted(queue_wait_times)[len(queue_wait_times)//2]:.2f}s")
            
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
    
    def create_simple_report(self, results: List[TestResult], save_path: str = "test_report.txt"):
        """创建简单的文本报告"""
        if not results:
            return
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("GenServe 并发测试报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总请求数: {len(results)}\n")
            
            successful = [r for r in results if r.success]
            f.write(f"成功请求: {len(successful)}\n")
            f.write(f"失败请求: {len(results) - len(successful)}\n")
            f.write(f"成功率: {len(successful)/len(results)*100:.1f}%\n\n")
            
            if successful:
                total_times = [r.end_time - r.start_time for r in successful]
                f.write(f"平均响应时间: {sum(total_times)/len(total_times):.2f}秒\n")
                f.write(f"最快响应时间: {min(total_times):.2f}秒\n")
                f.write(f"最慢响应时间: {max(total_times):.2f}秒\n\n")
                
                # 设备分布
                device_counts = {}
                for r in successful:
                    device_counts[r.device] = device_counts.get(r.device, 0) + 1
                
                f.write("设备负载分布:\n")
                for device, count in sorted(device_counts.items()):
                    f.write(f"  {device}: {count} 次 ({count/len(successful)*100:.1f}%)\n")
        
        print(f"📄 测试报告已保存: {save_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 GenServe 增强版并发测试")
    print("=" * 60)
    
    # 测试提示词
    test_prompts = [
        "a serene mountain landscape with a crystal clear lake reflecting the sky",
        "a playful golden retriever running through a field of sunflowers",
        "a futuristic cityscape with flying cars and neon lights",
        "a cozy coffee shop interior with warm lighting and wooden furniture",
        "a majestic dragon soaring through stormy clouds"
    ]
    
    tester = EnhancedConcurrentTester()
    
    # 1. 显示服务信息
    print("\n1. 📋 检查服务状态...")
    if not tester.display_service_info():
        print("❌ 服务未正常运行，请先启动服务")
        return
    
    # 2. 突发测试
    print("\n2. 💥 突发测试...")
    burst_results = tester.burst_test(test_prompts, 5, max_workers=5)  # 5个并发请求
    tester.analyze_results(burst_results, "突发测试")
    
    # 等待服务稳定
    time.sleep(5)
    
    # 3. 持续负载测试
    print("\n3. ⏱️ 持续负载测试...")
    load_results = tester.sustained_load_test(test_prompts, 30, 0.5, max_workers=8)  # 30秒，每秒0.5个请求
    tester.analyze_results(load_results, "持续负载测试")
    
    # 4. 生成综合报告
    print("\n4. 📊 生成综合报告...")
    all_results = tester.test_results
    tester.analyze_results(all_results, "综合测试")
    
    # 创建文本报告
    tester.create_simple_report(all_results)
    
    # 5. 最终状态检查
    print("\n5. 🔍 最终状态检查...")
    tester.display_service_info()
    
    print(f"\n✨ 测试完成！")
    print(f"📁 如果启用了图片保存（SAVE_TEST_IMAGES=true），图片将保存在当前目录")

if __name__ == "__main__":
    main()