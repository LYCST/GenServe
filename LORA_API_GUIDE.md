# LoRA API 使用指南

## 概述

GenServe 支持 LoRA (Low-Rank Adaptation) 功能，允许在图片生成时应用预训练的 LoRA 模型来调整生成效果。

## 功能特性

- ✅ 支持多个 LoRA 同时使用
- ✅ 支持自定义 LoRA 权重
- ✅ 支持层级文件夹结构
- ✅ 自动处理特殊字符和路径
- ✅ 唯一性保证，避免冲突

## 配置

### LoRA 路径配置

在 `.env` 文件中配置 LoRA 路径：

```bash
# LoRA 文件路径
LORA_PATH=/home/shuzuan/prj/models/loras
```

### LoRA 文件要求

- 文件格式：`.safetensors`
- 支持多层级文件夹结构
- 文件名可以包含中文和特殊字符

## API 接口

### 1. 获取 LoRA 列表

**接口**: `GET /loras`

**响应示例**:
```json
{
  "success": true,
  "loras": [
    "flux/【战神视觉】F.1_3D电商运营设计_V1.safetensors",
    "style/realistic_v1.safetensors",
    "character/anime_v2.safetensors"
  ],
  "total": 3
}
```

### 2. 使用 LoRA 生成图片

**接口**: `POST /generate`

**请求参数**:
```json
{
  "prompt": "A beautiful landscape",
  "mode": "text2img",
  "model_id": "flux1-dev",
  "loras": [
    {
      "name": "flux/【战神视觉】F.1_3D电商运营设计_V1.safetensors",
      "weight": 0.8
    },
    {
      "name": "style/realistic_v1.safetensors",
      "weight": 0.5
    }
  ],
  "height": 1024,
  "width": 1024,
  "seed": 42
}
```

**Form-data 参数**:
```
prompt: A beautiful landscape
mode: text2img
model_id: flux1-dev
loras[0][name]: flux/【战神视觉】F.1_3D电商运营设计_V1.safetensors
loras[0][weight]: 0.8
loras[1][name]: style/realistic_v1.safetensors
loras[1][weight]: 0.5
height: 1024
width: 1024
seed: 42
```

## 参数说明

### loras 参数

- **name**: LoRA 文件名（包含相对路径）
- **weight**: LoRA 权重 (0.0 - 2.0，推荐 0.5 - 1.0)

### 权重建议

- **0.0 - 0.3**: 轻微影响
- **0.3 - 0.7**: 中等影响
- **0.7 - 1.0**: 强烈影响
- **1.0 - 2.0**: 非常强烈影响（谨慎使用）

## 使用示例

### cURL 示例

```bash
# 获取 LoRA 列表
curl -X GET "http://localhost:8000/loras"

# 使用单个 LoRA
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "mode": "text2img",
    "model_id": "flux1-dev",
    "loras": [
      {
        "name": "flux/【战神视觉】F.1_3D电商运营设计_V1.safetensors",
        "weight": 0.8
      }
    ],
    "height": 1024,
    "width": 1024
  }'

# 使用多个 LoRA
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "mode": "text2img",
    "model_id": "flux1-dev",
    "loras": [
      {
        "name": "flux/【战神视觉】F.1_3D电商运营设计_V1.safetensors",
        "weight": 0.8
      },
      {
        "name": "style/realistic_v1.safetensors",
        "weight": 0.5
      }
    ],
    "height": 1024,
    "width": 1024
  }'
```

### Python 示例

```python
import requests
import json

# 获取 LoRA 列表
response = requests.get("http://localhost:8000/loras")
loras = response.json()["loras"]
print("可用 LoRA:", loras)

# 使用 LoRA 生成图片
data = {
    "prompt": "A beautiful landscape",
    "mode": "text2img",
    "model_id": "flux1-dev",
    "loras": [
        {
            "name": "flux/【战神视觉】F.1_3D电商运营设计_V1.safetensors",
            "weight": 0.8
        }
    ],
    "height": 1024,
    "width": 1024
}

response = requests.post("http://localhost:8000/generate", json=data)
result = response.json()

if result["success"]:
    print("生成成功！")
    # 保存图片
    import base64
    from PIL import Image
    import io
    
    image_data = base64.b64decode(result["image_base64"])
    image = Image.open(io.BytesIO(image_data))
    image.save("output.png")
else:
    print("生成失败:", result["error"])
```

## 错误处理

### 常见错误

1. **LoRA 不存在**
   ```
   LoRA 'xxx.safetensors' 不存在
   ```

2. **PEFT 依赖未安装**
   ```
   PEFT库未安装，无法使用LoRA功能。请安装: pip install peft
   ```

3. **LoRA 文件格式错误**
   ```
   LoRA文件格式无效或与当前模型不兼容
   ```

4. **权重超出范围**
   ```
   LoRA权重必须在0.0-2.0之间
   ```

### 解决方案

1. **安装 PEFT 依赖**:
   ```bash
   ./install_peft.sh
   ```

2. **检查 LoRA 文件**:
   ```bash
   # 确保文件存在且格式正确
   ls -la /home/shuzuan/prj/models/loras/
   ```

3. **验证 LoRA 列表**:
   ```bash
   curl -X GET "http://localhost:8000/loras"
   ```

## 最佳实践

1. **权重调整**: 从低权重开始，逐步调整到理想效果
2. **多 LoRA 组合**: 可以组合多个 LoRA 实现复杂效果
3. **文件命名**: 使用描述性名称，便于管理
4. **路径组织**: 使用文件夹分类不同类型的 LoRA

## 故障排除

### 检查服务状态

```bash
# 检查 GPU 状态
curl -X GET "http://localhost:8000/gpu_status"

# 检查模型支持
curl -X GET "http://localhost:8000/models"
```

### 查看日志

```bash
# 查看服务日志
tail -f logs/genserve.log
```

### 重启服务

```bash
# 重启服务
./start.sh
```

## 更新日志

- **v1.0**: 基础 LoRA 支持
- **v1.1**: 修复 adapter_name 冲突问题
- **v1.2**: 支持多层级文件夹结构
- **v1.3**: 优化错误处理和日志记录 