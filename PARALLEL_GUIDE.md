# GenServe 并行处理指南

## 🎉 恭喜！您的并行系统已经成功运行

根据您的测试结果，系统已经实现了真正的并行处理：

### ✅ 成功指标
- **并发度**: 10个请求同时处理
- **成功率**: 100%
- **GPU利用率**: 4个GPU同时工作
- **负载均衡**: 0.67分（良好水平）

## 🚀 如何使用并行系统

### 1. 启动服务
```bash
# 使用优化后的启动脚本
chmod +x start_optimized.sh
./start_optimized.sh

# 或使用原始启动脚本
./start.sh
```

### 2. 发送并发请求
```python
import asyncio
import aiohttp
import json

async def send_concurrent_requests():
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # 创建多个并发请求
        for i in range(10):
            data = {
                "prompt": f"Beautiful landscape {i}",
                "model_id": "flux1-dev",
                "height": 512,
                "width": 512,
                "num_inference_steps": 20,
                "priority": i % 3  # 不同优先级
            }
            
            task = session.post(
                "http://localhost:12411/generate",
                json=data
            )
            tasks.append(task)
        
        # 并发执行
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            result = await response.json()
            print(f"请求 {i+1}: GPU {result.get('gpu_id')}, 耗时 {result.get('elapsed_time')}s")

# 运行
asyncio.run(send_concurrent_requests())
```

## 🔧 性能优化工具

### 1. 性能测试脚本
```bash
python optimize_parallel_performance.py
```

这个脚本会：
- 测试不同规模的并发请求
- 分析GPU使用情况
- 评估负载均衡效果
- 提供优化建议

### 2. 配置优化脚本
```bash
python optimize_config.py
```

这个脚本会：
- 分析测试结果
- 自动调整配置参数
- 生成优化的启动脚本

## 📊 监控和调试

### 1. 查看服务状态
```bash
curl http://localhost:12411/status
```

### 2. 查看模型列表
```bash
curl http://localhost:12411/models
```

### 3. 健康检查
```bash
curl http://localhost:12411/health
```

## ⚙️ 关键配置参数

### 并发配置
```python
# 调度器睡眠时间（秒）
SCHEDULER_SLEEP_TIME = 0.05  # 更快的响应

# 全局队列大小
MAX_GLOBAL_QUEUE_SIZE = 150  # 更多缓冲

# 每个GPU队列大小
MAX_GPU_QUEUE_SIZE = 3  # 避免单个GPU过载

# 任务超时时间（秒）
TASK_TIMEOUT = 240  # 更长的超时
```

### 性能优化
```python
# GPU内存管理
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:64"

# 启用优化
ENABLE_OPTIMIZATION = "true"
MEMORY_EFFICIENT_ATTENTION = "true"
ENABLE_CPU_OFFLOAD = "true"
```

## 🎯 最佳实践

### 1. 请求优化
- 使用合适的图片尺寸（512x512用于测试）
- 减少推理步数（20步用于快速测试）
- 设置合理的优先级

### 2. 负载均衡
- 系统自动使用轮询负载均衡
- 监控GPU使用情况
- 避免单个GPU过载

### 3. 错误处理
- 设置合理的超时时间
- 处理队列满的情况
- 监控进程状态

## 🔍 故障排除

### 1. 服务无法启动
```bash
# 检查GPU状态
nvidia-smi

# 检查端口占用
netstat -tlnp | grep 12411

# 查看日志
tail -f logs/genserve.log
```

### 2. 请求失败
```bash
# 检查服务状态
curl http://localhost:12411/status

# 检查队列大小
curl http://localhost:12411/status | jq '.concurrent_manager.global_queue_size'
```

### 3. GPU进程死亡
```bash
# 查看进程状态
ps aux | grep gpu-worker

# 重启服务
pkill -f "python main.py"
./start_optimized.sh
```

## 📈 性能调优建议

### 基于您的测试结果

1. **响应时间优化**
   - 当前平均响应时间：49.75秒
   - 建议：减少推理步数或使用更快的GPU

2. **负载均衡优化**
   - 当前均衡度：0.67
   - 建议：检查GPU性能差异，调整分配策略

3. **并发度优化**
   - 当前并发效率：0.8
   - 建议：增加GPU数量或优化队列管理

## 🎮 高级功能

### 1. 优先级队列
```python
# 高优先级请求
data = {
    "prompt": "urgent request",
    "priority": 0  # 最高优先级
}

# 普通请求
data = {
    "prompt": "normal request", 
    "priority": 1  # 普通优先级
}
```

### 2. 任务状态查询
```python
# 查询任务状态
task_id = "your-task-id"
response = requests.get(f"http://localhost:12411/task/{task_id}")
```

### 3. 批量处理
```python
# 批量发送请求
async def batch_process(prompts):
    tasks = []
    for prompt in prompts:
        task = send_request(prompt)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## 🎉 总结

您的并行系统已经成功运行，具备以下特点：

✅ **真正的并行处理** - 多个GPU同时工作  
✅ **智能负载均衡** - 自动分配任务到可用GPU  
✅ **优先级支持** - 支持任务优先级调度  
✅ **容错机制** - 自动重启死亡进程  
✅ **性能监控** - 详细的统计和监控  

继续使用这些工具和最佳实践，您的并行图片生成服务将能够高效处理大量并发请求！ 