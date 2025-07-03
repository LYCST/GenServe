#!/usr/bin/env python3
"""
性能监控脚本
专门监控GenServe的性能指标和GPU内存使用趋势
"""

import psutil
import time
import subprocess
import json
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class PerformanceMonitor:
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.gpu_memory_history = {i: deque(maxlen=history_size) for i in range(8)}
        self.system_memory_history = deque(maxlen=history_size)
        self.cpu_usage_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
    def get_gpu_memory(self):
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
    
    def get_system_memory(self):
        """获取系统内存信息"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }
    
    def get_genserve_cpu_usage(self):
        """获取GenServe进程CPU使用率"""
        total_cpu = 0
        count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.cmdline())
                    if 'genserve' in cmdline.lower() or 'main.py' in cmdline:
                        total_cpu += proc.info['cpu_percent']
                        count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_cpu / count if count > 0 else 0
    
    def get_service_metrics(self):
        """获取服务性能指标"""
        try:
            response = requests.get("http://localhost:12411/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                concurrent_status = data.get("concurrent_manager", {})
                return {
                    "alive_processes": concurrent_status.get('alive_processes', 0),
                    "dead_processes": concurrent_status.get('dead_processes', 0),
                    "total_restarts": concurrent_status.get('total_restarts', 0),
                    "global_queue_size": concurrent_status.get('global_queue_size', 0)
                }
        except Exception as e:
            print(f"获取服务指标失败: {e}")
        return {}
    
    def collect_data(self):
        """收集性能数据"""
        timestamp = datetime.now()
        
        # GPU内存数据
        gpu_memory = self.get_gpu_memory()
        for gpu in gpu_memory:
            gpu_id = gpu['gpu_id']
            if gpu_id < 8:  # 只监控前8个GPU
                self.gpu_memory_history[gpu_id].append(gpu['used_percent'])
        
        # 系统内存数据
        system_memory = self.get_system_memory()
        self.system_memory_history.append(system_memory['percent'])
        
        # CPU使用率
        cpu_usage = self.get_genserve_cpu_usage()
        self.cpu_usage_history.append(cpu_usage)
        
        # 时间戳
        self.timestamps.append(timestamp)
        
        return {
            "timestamp": timestamp,
            "gpu_memory": gpu_memory,
            "system_memory": system_memory,
            "cpu_usage": cpu_usage,
            "service_metrics": self.get_service_metrics()
        }
    
    def print_performance_summary(self, data):
        """打印性能摘要"""
        print("🚀 GenServe 性能监控")
        print("=" * 60)
        
        # GPU内存使用情况
        print(f"🎮 GPU内存使用:")
        for gpu in data['gpu_memory']:
            gpu_id = gpu['gpu_id']
            used_percent = gpu['used_percent']
            status = "🟢" if used_percent < 50 else "🟡" if used_percent < 80 else "🔴"
            print(f"  {status} GPU {gpu_id}: {gpu['used_mb']}MB/{gpu['total_mb']}MB ({used_percent:.1f}%)")
        
        # 系统内存
        memory = data['system_memory']
        memory_status = "🟢" if memory['percent'] < 70 else "🟡" if memory['percent'] < 90 else "🔴"
        print(f"\n💾 系统内存: {memory_status} {memory['used_gb']:.1f}GB/{memory['total_gb']:.1f}GB ({memory['percent']:.1f}%)")
        
        # CPU使用率
        cpu_usage = data['cpu_usage']
        cpu_status = "🟢" if cpu_usage < 30 else "🟡" if cpu_usage < 70 else "🔴"
        print(f"🖥️  CPU使用率: {cpu_status} {cpu_usage:.1f}%")
        
        # 服务指标
        service = data['service_metrics']
        print(f"\n🔧 服务状态:")
        print(f"  活跃进程: {service.get('alive_processes', 0)}/8")
        print(f"  死亡进程: {service.get('dead_processes', 0)}")
        print(f"  总重启次数: {service.get('total_restarts', 0)}")
        print(f"  全局队列: {service.get('global_queue_size', 0)}")
        
        # 性能趋势
        if len(self.timestamps) > 1:
            print(f"\n📈 性能趋势:")
            
            # GPU内存趋势
            for gpu_id in range(8):
                if len(self.gpu_memory_history[gpu_id]) >= 2:
                    recent_avg = np.mean(list(self.gpu_memory_history[gpu_id])[-5:])
                    if len(self.gpu_memory_history[gpu_id]) >= 10:
                        older_avg = np.mean(list(self.gpu_memory_history[gpu_id])[-10:-5])
                        trend = "↗️" if recent_avg > older_avg + 5 else "↘️" if recent_avg < older_avg - 5 else "➡️"
                        print(f"  GPU {gpu_id}: {trend} 最近平均 {recent_avg:.1f}%")
            
            # 系统内存趋势
            if len(self.system_memory_history) >= 10:
                recent_avg = np.mean(list(self.system_memory_history)[-5:])
                older_avg = np.mean(list(self.system_memory_history)[-10:-5])
                trend = "↗️" if recent_avg > older_avg + 5 else "↘️" if recent_avg < older_avg - 5 else "➡️"
                print(f"  系统内存: {trend} 最近平均 {recent_avg:.1f}%")
        
        print(f"\n⏰ 监控时间: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def plot_performance_trends(self):
        """绘制性能趋势图"""
        if len(self.timestamps) < 10:
            print("数据点不足，无法绘制趋势图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GenServe 性能监控趋势', fontsize=16)
        
        # GPU内存趋势
        ax1 = axes[0, 0]
        for gpu_id in range(8):
            if len(self.gpu_memory_history[gpu_id]) > 0:
                ax1.plot(list(self.gpu_memory_history[gpu_id]), label=f'GPU {gpu_id}')
        ax1.set_title('GPU内存使用率趋势')
        ax1.set_ylabel('使用率 (%)')
        ax1.legend()
        ax1.grid(True)
        
        # 系统内存趋势
        ax2 = axes[0, 1]
        ax2.plot(list(self.system_memory_history), 'r-', label='系统内存')
        ax2.set_title('系统内存使用率趋势')
        ax2.set_ylabel('使用率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        # CPU使用率趋势
        ax3 = axes[1, 0]
        ax3.plot(list(self.cpu_usage_history), 'g-', label='CPU使用率')
        ax3.set_title('CPU使用率趋势')
        ax3.set_ylabel('使用率 (%)')
        ax3.legend()
        ax3.grid(True)
        
        # 服务指标
        ax4 = axes[1, 1]
        # 这里可以添加更多服务指标的可视化
        
        plt.tight_layout()
        plt.savefig('genserve_performance.png', dpi=150, bbox_inches='tight')
        print("📊 性能趋势图已保存为 genserve_performance.png")
    
    def continuous_monitor(self, interval=30, plot_interval=300):
        """持续监控"""
        print(f"🔄 开始持续性能监控，间隔 {interval} 秒...")
        print("按 Ctrl+C 停止监控")
        
        last_plot_time = time.time()
        
        try:
            while True:
                data = self.collect_data()
                self.print_performance_summary(data)
                
                # 定期生成趋势图
                current_time = time.time()
                if current_time - last_plot_time > plot_interval:
                    self.plot_performance_trends()
                    last_plot_time = current_time
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n⏹️ 性能监控已停止")
            # 生成最终趋势图
            self.plot_performance_trends()

def main():
    import sys
    
    monitor = PerformanceMonitor()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--continuous":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            plot_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 300
            monitor.continuous_monitor(interval, plot_interval)
        elif sys.argv[1] == "--plot":
            # 收集一些数据后生成图表
            for _ in range(10):
                monitor.collect_data()
                time.sleep(5)
            monitor.plot_performance_trends()
        else:
            print("用法: python performance_monitor.py [--continuous [interval] [plot_interval] | --plot]")
    else:
        # 单次监控
        data = monitor.collect_data()
        monitor.print_performance_summary(data)

if __name__ == "__main__":
    main() 