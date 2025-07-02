# GenServe

一个基于多种模型的图片生成API服务，支持智能GPU负载均衡。

## 🚀 核心特性

- **智能GPU负载均衡**：自动为每个模型选择负载最低的GPU
- **多模型支持**：支持Flux、SDXL等多种图片生成模型
- **插件化架构**：模型之间完全解耦，易于扩展
- **RESTful API**：标准化的HTTP接口
- **灵活配置**：每个模型可独立配置可用GPU列表

## 🎯 GPU负载均衡

### 配置模型GPU

在 `config.py` 中为每个模型配置可用的GPU列表：

```python
MODEL_GPU_CONFIG = {
    "flux1-dev": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],  # Flux可以使用GPU 0,1,2,3
    "sdxl-base": ["cuda:0", "cuda:1"],  # SDXL可以使用GPU 0,1
}
```

### 自动负载均衡

- **模型加载时**：自动选择负载最低的GPU进行加载
- **图片生成时**：每次生成都会重新评估GPU负载，选择最佳设备
- **动态迁移**：模型可以在GPU之间动态迁移以优化性能

## 📦 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务

```bash
python main.py
```

服务将在 http://localhost:8000 启动

## 🔧 API接口

### 基础接口

- `GET /models` - 获取支持的模型列表
- `GET /health` - 健康状态检查
- `GET /config` - 获取服务配置
- `GET /devices` - 获取设备信息
- `GET /gpu/load` - 获取GPU负载信息

### 图片生成

- `POST /generate` - 生成图片（自动选择最佳GPU）

### 模型管理

- `POST /models/{model_id}/load` - 手动加载模型
- `POST /models/{model_id}/unload` - 手动卸载模型
- `POST /models/{model_id}/device` - 设置模型设备

## 💡 使用示例

### 基础图片生成

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
       "model": "flux1-dev"
     }'
```

### 查看GPU负载

```bash
curl http://localhost:8000/gpu/load
```

### 查看设备信息

```bash
curl http://localhost:8000/devices
```

## 🏗️ 项目结构

```
GenServe/
├── main.py                 # 主应用文件
├── config.py              # 配置文件（包含GPU配置）
├── device_manager.py      # 设备管理器
├── models/                # 模型包
│   ├── base.py           # 基础模型类
│   ├── manager.py        # 模型管理器
│   └── flux_model.py     # Flux模型实现
└── test_api.py           # API测试脚本
```

## ⚙️ 配置说明

### 环境变量

```bash
# 服务配置
export HOST=0.0.0.0
export PORT=8000

# 设备配置
export USE_GPU=true
export TORCH_DTYPE=float16

# 模型管理
export AUTO_LOAD_MODELS=true
```

### GPU配置

在 `config.py` 中修改 `MODEL_GPU_CONFIG`：

```python
MODEL_GPU_CONFIG = {
    "flux1-dev": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    "sdxl-base": ["cuda:0", "cuda:1"],
    # 添加更多模型...
}
```

## 🔍 负载均衡策略

1. **GPU选择算法**：
   - 从配置的GPU列表中筛选出可用的GPU
   - 计算每个GPU的内存使用率
   - 选择使用率最低的GPU

2. **动态迁移**：
   - 每次生成图片时重新评估GPU负载
   - 如果发现更合适的GPU，自动迁移模型

3. **容错机制**：
   - 如果GPU不可用，自动降级到CPU
   - 如果迁移失败，使用当前设备继续运行

## 🧪 测试

运行测试脚本验证功能：

```bash
python test_api.py
```

测试包括：
- 基础API功能
- GPU负载均衡
- 多次生成测试
- 设备管理

## 📊 监控

### GPU负载信息

```json
{
  "gpu_load_info": {
    "cuda:0": {
      "allocated_mb": 2048.0,
      "cached_mb": 4096.0,
      "total_mb": 8192.0,
      "free_mb": 4096.0,
      "utilization_percent": 50.0,
      "available": true
    }
  },
  "model_gpu_config": {
    "flux1-dev": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
  }
}
```

## 🚨 故障排除

### 常见问题

1. **GPU内存不足**
   - 减少图片分辨率
   - 降低推理步数
   - 检查GPU配置是否正确

2. **模型加载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 验证GPU设备配置

3. **负载均衡不工作**
   - 检查 `MODEL_GPU_CONFIG` 配置
   - 验证GPU设备是否可用
   - 查看日志中的GPU选择信息

## 📈 性能优化

- 使用半精度浮点数（float16）
- 内存高效注意力机制
- Flash Attention 2 支持
- 智能GPU负载均衡
- 模型懒加载和按需卸载

## 🤝 贡献

欢迎提交Issue和Pull Request来改进GenServe！

## 📄 许可证

本项目采用MIT许可证。 