#!/usr/bin/env python3
"""
GPU进程状态检查脚本
"""

import requests
import json
import time

def check_service_status():
    """检查服务状态"""
    try:
        response = requests.get("http://localhost:12411/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 服务响应错误: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return None

def test_single_generation():
    """测试单个图片生成"""
    payload = {
        "prompt": "a simple test image",
        "model_id": "flux1-dev",
        "num_inference_steps": 10,  # 减少步数加快测试
        "height": 512,
        "width": 512,
        "seed": 42
    }
    
    try:
        print("🔄 发送测试请求...")
        response = requests.post(
            "http://localhost:12411/generate", 
            json=payload, 
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功: {result.get('success', False)}")
            if result.get('success'):
                print(f"  GPU: {result.get('gpu_id', 'unknown')}")
                print(f"  耗时: {result.get('elapsed_time', 0):.2f}秒")
                print(f"  任务ID: {result.get('task_id', 'unknown')}")
            else:
                print(f"  错误: {result.get('error', 'unknown')}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"  响应: {response.text}")
            
    except Exception as e:
        print(f"❌ 测试请求异常: {e}")

def main():
    print("🔍 GPU进程状态检查")
    print("=" * 50)
    
    # 1. 检查服务状态
    print("\n1. 检查服务状态...")
    status = check_service_status()
    if not status:
        print("❌ 服务不可用")
        return
    
    # 2. 显示并发管理器状态
    concurrent_status = status.get("concurrent_manager", {})
    print(f"📊 并发管理器状态:")
    print(f"  运行状态: {'🟢 运行中' if concurrent_status.get('is_running') else '🔴 已停止'}")
    print(f"  全局队列: {concurrent_status.get('global_queue_size', 0)} 个任务")
    print(f"  总进程数: {concurrent_status.get('total_processes', 0)}")
    print(f"  活跃进程: {concurrent_status.get('alive_processes', 0)}")
    
    # 3. 显示GPU进程详情
    gpu_processes = concurrent_status.get("gpu_processes", {})
    print(f"\n🤖 GPU进程详情:")
    for process_key, process_info in gpu_processes.items():
        status_icon = "🟢" if process_info.get("alive") else "🔴"
        print(f"  {status_icon} {process_key}:")
        print(f"    PID: {process_info.get('pid', 'unknown')}")
        print(f"    状态: {'活跃' if process_info.get('alive') else '死亡'}")
        print(f"    退出码: {process_info.get('exitcode', 'unknown')}")
        if not process_info.get("alive"):
            print(f"    ⚠️ 进程已死亡，可能需要重启服务")
    
    # 4. 测试单个生成任务
    print(f"\n2. 测试单个生成任务...")
    test_single_generation()
    
    print(f"\n✨ 检查完成")

if __name__ == "__main__":
    main() 