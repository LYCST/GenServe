#!/usr/bin/env python3
"""
快速测试脚本
测试GenServe的性能和稳定性
"""

import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

class QuickTester:
    def __init__(self, base_url="http://localhost:12411"):
        self.base_url = base_url
        self.test_results = []
        
    def test_single_generation(self, task_id, prompt, timeout=180):
        """测试单次图片生成"""
        payload = {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "cfg": 3.5,
            "num_inference_steps": 50,
            "seed": random.randint(1, 1000000)
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=timeout
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get('success', False)
                return {
                    "task_id": task_id,
                    "success": success,
                    "elapsed_time": elapsed_time,
                    "error": result.get('error') if not success else None,
                    "gpu_id": result.get('gpu_id'),
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt
                }
            else:
                return {
                    "task_id": task_id,
                    "success": False,
                    "elapsed_time": elapsed_time,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "gpu_id": None,
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt
                }
        except requests.exceptions.Timeout:
            return {
                "task_id": task_id,
                "success": False,
                "elapsed_time": timeout,
                "error": "请求超时",
                "gpu_id": None,
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "success": False,
                "elapsed_time": time.time() - start_time,
                "error": str(e),
                "gpu_id": None,
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt
            }
    
    def test_concurrent_generation(self, num_requests=8, max_workers=4):
        """测试并发图片生成"""
        print(f"🚀 开始并发测试: {num_requests} 个请求，最大并发数: {max_workers}")
        
        # 测试提示词
        test_prompts = [
            "A beautiful sunset over mountains",
            "A futuristic city with flying cars",
            "A peaceful forest with sunlight filtering through trees",
            "A majestic dragon flying over a castle",
            "A serene lake reflecting the sky",
            "A space station orbiting Earth",
            "A magical garden with glowing flowers",
            "A steampunk airship in the clouds"
        ]
        
        # 扩展提示词列表
        while len(test_prompts) < num_requests:
            test_prompts.extend(test_prompts[:num_requests - len(test_prompts)])
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.test_single_generation, i, prompt): i 
                for i, prompt in enumerate(test_prompts[:num_requests])
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                
                # 实时显示进度
                completed = len(results)
                success_count = sum(1 for r in results if r['success'])
                print(f"📊 进度: {completed}/{num_requests} (成功: {success_count})")
                
                if result['success']:
                    print(f"  ✅ 任务 {result['task_id']}: GPU {result['gpu_id']}, 耗时 {result['elapsed_time']:.2f}s")
                else:
                    print(f"  ❌ 任务 {result['task_id']}: {result['error']}")
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        failure_count = len(results) - success_count
        
        # 统计GPU使用情况
        gpu_usage = {}
        for result in results:
            if result['success'] and result['gpu_id']:
                gpu_id = result['gpu_id']
                if gpu_id not in gpu_usage:
                    gpu_usage[gpu_id] = 0
                gpu_usage[gpu_id] += 1
        
        # 计算平均时间
        successful_times = [r['elapsed_time'] for r in results if r['success']]
        avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        print(f"\n📈 并发测试结果:")
        print(f"  总请求数: {num_requests}")
        print(f"  成功: {success_count}")
        print(f"  失败: {failure_count}")
        print(f"  成功率: {success_count/num_requests*100:.1f}%")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  平均耗时: {avg_time:.2f}秒")
        print(f"  吞吐量: {success_count/total_time:.2f} 请求/秒")
        
        if gpu_usage:
            print(f"  GPU使用分布:")
            for gpu_id, count in sorted(gpu_usage.items()):
                print(f"    GPU {gpu_id}: {count} 个任务")
        
        return results
    
    def test_service_status(self):
        """测试服务状态"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"🔧 服务状态:")
                print(f"  状态: {status.get('status', 'unknown')}")
                
                concurrent_status = status.get('concurrent_manager', {})
                print(f"  活跃进程: {concurrent_status.get('alive_processes', 0)}/8")
                print(f"  死亡进程: {concurrent_status.get('dead_processes', 0)}")
                print(f"  总重启次数: {concurrent_status.get('total_restarts', 0)}")
                print(f"  全局队列: {concurrent_status.get('global_queue_size', 0)}")
                
                return status
            else:
                print(f"❌ 获取服务状态失败: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 获取服务状态失败: {e}")
            return None
    
    def test_memory_cleanup(self):
        """测试内存清理效果"""
        print("🧹 测试内存清理效果...")
        
        # 先进行一些生成任务
        print("1. 执行初始生成任务...")
        initial_results = []
        for i in range(4):
            result = self.test_single_generation(i, f"Test image {i}")
            initial_results.append(result)
            time.sleep(1)
        
        # 等待内存清理
        print("2. 等待内存清理...")
        time.sleep(10)
        
        # 再次进行生成任务
        print("3. 执行后续生成任务...")
        final_results = []
        for i in range(4):
            result = self.test_single_generation(i+4, f"Test image {i+4}")
            final_results.append(result)
            time.sleep(1)
        
        # 分析结果
        initial_success = sum(1 for r in initial_results if r['success'])
        final_success = sum(1 for r in final_results if r['success'])
        
        print(f"📊 内存清理测试结果:")
        print(f"  初始任务成功率: {initial_success}/4 ({initial_success/4*100:.1f}%)")
        print(f"  后续任务成功率: {final_success}/4 ({final_success/4*100:.1f}%)")
        
        if initial_success == 4 and final_success == 4:
            print("✅ 内存清理测试通过")
        else:
            print("⚠️ 内存清理测试存在问题")
        
        return initial_results + final_results
    
    def run_full_test(self):
        """运行完整测试套件"""
        print("🧪 GenServe 完整性能测试")
        print("=" * 60)
        
        # 1. 检查服务状态
        print("1️⃣ 检查服务状态...")
        status = self.test_service_status()
        if not status:
            print("❌ 服务不可用，停止测试")
            return
        
        print()
        
        # 2. 测试内存清理
        print("2️⃣ 测试内存清理...")
        memory_results = self.test_memory_cleanup()
        
        print()
        
        # 3. 测试并发性能
        print("3️⃣ 测试并发性能...")
        concurrent_results = self.test_concurrent_generation(num_requests=16, max_workers=8)
        
        print()
        
        # 4. 最终状态检查
        print("4️⃣ 最终状态检查...")
        final_status = self.test_service_status()
        
        print()
        print("🎉 测试完成!")
        
        # 保存测试结果
        test_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "initial_status": status,
            "final_status": final_status,
            "memory_test_results": memory_results,
            "concurrent_test_results": concurrent_results
        }
        
        with open("test_results.json", "w") as f:
            json.dump(test_summary, f, indent=2, default=str)
        
        print("📄 测试结果已保存到 test_results.json")

def main():
    import sys
    
    tester = QuickTester()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "concurrent":
            num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 8
            max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
            tester.test_concurrent_generation(num_requests, max_workers)
        elif sys.argv[1] == "memory":
            tester.test_memory_cleanup()
        elif sys.argv[1] == "status":
            tester.test_service_status()
        else:
            print("用法: python quick_test.py [concurrent [num_requests] [max_workers] | memory | status]")
    else:
        tester.run_full_test()

if __name__ == "__main__":
    main() 