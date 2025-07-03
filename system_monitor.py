#!/usr/bin/env python3
"""
系统监控脚本
监控内存使用、进程状态和OOM事件
"""

import psutil
import time
import subprocess
import json
from datetime import datetime
import requests

def get_system_memory():
    """获取系统内存信息"""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent": memory.percent,
        "free_gb": memory.free / (1024**3)
    }

def get_gpu_memory():
    """获取GPU内存信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_memory = []
            for i, line in enumerate(lines):
                used, total = map(int, line.split(', '))
                gpu_memory.append({
                    "gpu_id": i,
                    "used_mb": used,
                    "total_mb": total,
                    "used_percent": (used / total) * 100
                })
            return gpu_memory
    except Exception as e:
        print(f"获取GPU内存信息失败: {e}")
    return []

def get_genserve_processes():
    """获取GenServe相关进程"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.cmdline())
                if 'genserve' in cmdline.lower() or 'main.py' in cmdline:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "memory_mb": proc.info['memory_info'].rss / (1024**2),
                        "cpu_percent": proc.info['cpu_percent'],
                        "cmdline": cmdline[:100] + "..." if len(cmdline) > 100 else cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def check_oom_events():
    """检查OOM事件"""
    try:
        result = subprocess.run(['dmesg'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            oom_events = []
            for line in lines:
                if 'killed process' in line.lower() or 'out of memory' in line.lower():
                    oom_events.append(line.strip())
            return oom_events[-5:]  # 返回最近5个OOM事件
    except Exception as e:
        print(f"检查OOM事件失败: {e}")
    return []

def get_service_status():
    """获取服务状态"""
    try:
        response = requests.get("http://localhost:12411/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"获取服务状态失败: {e}")
    return None

def monitor_system():
    """系统监控主函数"""
    print("🔍 系统监控")
    print("=" * 60)
    
    # 系统内存
    memory = get_system_memory()
    print(f"💾 系统内存:")
    print(f"  总内存: {memory['total_gb']:.1f}GB")
    print(f"  已使用: {memory['used_gb']:.1f}GB ({memory['percent']:.1f}%)")
    print(f"  可用: {memory['available_gb']:.1f}GB")
    print(f"  空闲: {memory['free_gb']:.1f}GB")
    
    # GPU内存
    gpu_memory = get_gpu_memory()
    if gpu_memory:
        print(f"\n🎮 GPU内存:")
        for gpu in gpu_memory:
            print(f"  GPU {gpu['gpu_id']}: {gpu['used_mb']}MB/{gpu['total_mb']}MB ({gpu['used_percent']:.1f}%)")
    
    # GenServe进程
    processes = get_genserve_processes()
    if processes:
        print(f"\n🐍 GenServe进程:")
        for proc in processes:
            print(f"  PID {proc['pid']}: {proc['memory_mb']:.1f}MB, CPU {proc['cpu_percent']:.1f}%")
            print(f"    {proc['cmdline']}")
    
    # 服务状态
    service_status = get_service_status()
    if service_status:
        concurrent_status = service_status.get("concurrent_manager", {})
        print(f"\n🚀 服务状态:")
        print(f"  活跃进程: {concurrent_status.get('alive_processes', 0)}/{concurrent_status.get('total_processes', 0)}")
        print(f"  死亡进程: {concurrent_status.get('dead_processes', 0)}")
        print(f"  总重启次数: {concurrent_status.get('total_restarts', 0)}")
        print(f"  全局队列: {concurrent_status.get('global_queue_size', 0)}")
    
    # OOM事件
    oom_events = check_oom_events()
    if oom_events:
        print(f"\n⚠️ 最近OOM事件:")
        for event in oom_events:
            print(f"  {event}")
    else:
        print(f"\n✅ 未发现OOM事件")
    
    print(f"\n⏰ 监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def continuous_monitor(interval=30):
    """持续监控"""
    print(f"🔄 开始持续监控，间隔 {interval} 秒...")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            monitor_system()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n⏹️ 监控已停止")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        continuous_monitor(interval)
    else:
        monitor_system() 