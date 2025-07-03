#!/usr/bin/env python3
"""
内存管理测试脚本
验证GPU任务间隔和内存清理功能
"""

import requests
import time
import json
from datetime import datetime

def test_memory_management():
    """测试内存管理功能"""
    print("🧪 内存管理测试")
    print("=" * 50)
    
    # 测试参数
    test_prompts = [
        "a beautiful landscape with mountains and lakes",
        "a futuristic city with flying cars",
        "a cozy coffee shop interior",
        "a majestic dragon in the sky",
        "a serene garden with flowers"
    ]
    
    base_url = "http://localhost:12411"
    
    print("📊 测试前检查服务状态...")
    try:
        status_response = requests.get(f"{base_url}/status", timeout=10)
        if status_response.status_code == 200:
            status = status_response.json()
            concurrent_status = status.get("concurrent_manager", {})
            print(f"  活跃进程: {concurrent_status.get('alive_processes', 0)}/{concurrent_status.get('total_processes', 0)}")
            print(f"  死亡进程: {concurrent_status.get('dead_processes', 0)}")
            print(f"  总重启次数: {concurrent_status.get('total_restarts', 0)}")
        else:
            print(f"❌ 无法获取服务状态: {status_response.status_code}")
            return
    except Exception as e:
        print(f"❌ 连接服务失败: {e}")
        return
    
    print(f"\n🔄 开始顺序测试 {len(test_prompts)} 个任务...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- 任务 {i}/{len(test_prompts)} ---")
        print(f"提示词: {prompt}")
        
        payload = {
            "prompt": prompt,
            "model_id": "flux1-dev",
            "num_inference_steps": 20,  # 减少步数加快测试
            "height": 512,
            "width": 512,
            "seed": 42 + i
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                timeout=120
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ 任务成功")
                    print(f"  GPU: {result.get('gpu_id', 'unknown')}")
                    print(f"  生成时间: {result.get('elapsed_time', 0):.2f}秒")
                    print(f"  总时间: {total_time:.2f}秒")
                    print(f"  任务ID: {result.get('task_id', 'unknown')}")
                    
                    # 检查是否保存到磁盘
                    if result.get("save_to_disk"):
                        print(f"  💾 已保存到磁盘")
                    
                    # 显示实际参数
                    params = result.get("params", {})
                    if params:
                        print(f"  实际参数: {json.dumps(params, indent=2)}")
                else:
                    print(f"❌ 任务失败: {result.get('error', 'unknown')}")
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"  响应: {response.text}")
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
        
        # 任务间隔
        if i < len(test_prompts):
            interval = 2.0  # 2秒间隔
            print(f"⏳ 等待 {interval} 秒后继续下一个任务...")
            time.sleep(interval)
    
    print(f"\n📊 测试后检查服务状态...")
    try:
        status_response = requests.get(f"{base_url}/status", timeout=10)
        if status_response.status_code == 200:
            status = status_response.json()
            concurrent_status = status.get("concurrent_manager", {})
            print(f"  活跃进程: {concurrent_status.get('alive_processes', 0)}/{concurrent_status.get('total_processes', 0)}")
            print(f"  死亡进程: {concurrent_status.get('dead_processes', 0)}")
            print(f"  总重启次数: {concurrent_status.get('total_restarts', 0)}")
            
            # 检查是否有进程死亡
            dead_processes = concurrent_status.get('dead_processes', 0)
            if dead_processes > 0:
                print(f"⚠️ 发现 {dead_processes} 个死亡进程")
            else:
                print(f"✅ 所有进程正常运行")
        else:
            print(f"❌ 无法获取服务状态: {status_response.status_code}")
    except Exception as e:
        print(f"❌ 连接服务失败: {e}")
    
    print(f"\n✨ 内存管理测试完成")

if __name__ == "__main__":
    test_memory_management() 