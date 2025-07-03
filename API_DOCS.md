# GenServe API 文档

## 概述

GenServe是一个基于Flux模型的多GPU并发图片生成服务，支持多种图生图模式。

## 基础信息

- **服务地址**: `http://localhost:12411`
- **支持的模型**: FLUX.1-dev
- **支持的生成模式**: text2img, img2img, fill, controlnet
- **上传方式**: JSON(base64) 和 Form-data(文件上传)

## 上传方式对比

| 特性 | JSON (Base64) | Form-data |
|------|---------------|-----------|
| **数据大小** | +33% (编码开销) | 原始大小 |
| **上传速度** | 较慢 | 较快 |
| **内存占用** | 较高 | 较低 |
| **适用场景** | 小图片、简单集成 | 大图片、批量处理 |
| **开发便利性** | 简单 | 需要文件处理 |
| **网络传输** | 文本格式 | 二进制格式 |

**推荐使用场景**:
- **Form-data**: 1024x1024以上大图片、批量处理、移动端应用
- **Base64**: 小图片测试、简单集成、快速原型开发

## API 端点

### 1. 生成图片 (JSON方式)

**端点**: `POST /generate`

**描述**: 使用JSON格式和base64编码的图片数据生成图片

#### 请求参数

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| prompt | string | ✅ | - | 提示词 |
| model_id | string | ❌ | "flux1-dev" | 模型ID |
| mode | string | ❌ | "text2img" | 生成模式 |
| height | integer | ❌ | 1024 | 图片高度 |
| width | integer | ❌ | 1024 | 图片宽度 |
| num_inference_steps | integer | ❌ | 50 | 推理步数 |
| cfg | float | ❌ | 3.5 | 引导强度 |
| seed | integer | ❌ | 42 | 随机种子 |
| priority | integer | ❌ | 0 | 任务优先级 |
| strength | float | ❌ | 0.8 | img2img强度 |
| input_image | string | 条件 | null | 输入图片(base64) |
| mask_image | string | 条件 | null | 蒙版图片(base64) |
| control_image | string | 条件 | null | 控制图片(base64) |

#### 请求示例

```json
{
  "prompt": "A beautiful sunset over mountains",
  "mode": "img2img",
  "input_image": "base64_encoded_image_data",
  "strength": 0.7,
  "height": 1024,
  "width": 1024,
  "num_inference_steps": 50,
  "cfg": 3.5,
  "seed": 42
}
```

### 2. 生成图片 (Form-data方式)

**端点**: `POST /generate/img2img`

**描述**: 使用Form-data格式和文件上传生成图片（推荐用于大图片）

#### 请求参数

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| prompt | string | ✅ | - | 提示词 |
| model_id | string | ❌ | "flux1-dev" | 模型ID |
| mode | string | ❌ | "text2img" | 生成模式 |
| height | integer | ❌ | 1024 | 图片高度 |
| width | integer | ❌ | 1024 | 图片宽度 |
| num_inference_steps | integer | ❌ | 50 | 推理步数 |
| cfg | float | ❌ | 3.5 | 引导强度 |
| seed | integer | ❌ | 42 | 随机种子 |
| priority | integer | ❌ | 0 | 任务优先级 |
| strength | float | ❌ | 0.8 | img2img强度 |
| input_image | file | 条件 | null | 输入图片文件 |
| mask_image | file | 条件 | null | 蒙版图片文件 |
| control_image | file | 条件 | null | 控制图片文件 |

#### 请求示例

```bash
curl -X POST "http://localhost:12411/generate/img2img" \
     -F "prompt=A beautiful sunset over mountains" \
     -F "mode=img2img" \
     -F "input_image=@input_image.png" \
     -F "strength=0.7" \
     -F "height=1024" \
     -F "width=1024" \
     -F "num_inference_steps=50" \
     -F "cfg=3.5" \
     -F "seed=42"
```

#### 生成模式说明

##### 1. text2img (文本生成图片)
- **描述**: 根据文本提示词生成图片
- **必需参数**: `prompt`
- **可选参数**: 所有基础参数

##### 2. img2img (图片生成图片)
- **描述**: 基于输入图片生成新图片
- **必需参数**: `prompt`, `input_image`
- **可选参数**: `strength` (控制变化强度，0-1)

##### 3. fill (填充/修复)
- **描述**: 根据蒙版区域填充或修复图片
- **必需参数**: `prompt`, `input_image`, `mask_image`
- **说明**: 蒙版中白色区域将被重新生成

##### 4. controlnet (ControlNet控制)
- **描述**: 使用ControlNet进行精确控制生成
- **必需参数**: `prompt`, `input_image`, `control_image`
- **说明**: 需要ControlNet模型支持

#### 响应格式

```json
{
  "success": true,
  "task_id": "uuid-string",
  "image_base64": "base64_encoded_generated_image",
  "error": null,
  "elapsed_time": 15.23,
  "gpu_id": "cuda:0",
  "model_id": "flux1-dev",
  "mode": "text2img"
}
```

#### 错误响应

```json
{
  "success": false,
  "task_id": "uuid-string",
  "image_base64": null,
  "error": "错误描述",
  "elapsed_time": 0.0,
  "gpu_id": null,
  "model_id": "flux1-dev",
  "mode": "text2img"
}
```

### 3. 服务状态

**端点**: `GET /status`

**描述**: 获取服务状态信息

#### 响应格式

```json
{
  "status": "running",
  "concurrent_manager": {
    "alive_processes": 8,
    "dead_processes": 0,
    "total_restarts": 3,
    "global_queue_size": 0
  }
}
```

### 4. 模型列表

**端点**: `GET /models`

**描述**: 获取支持的模型列表

#### 响应格式

```json
[
  {
    "model_id": "flux1-dev",
    "model_name": "FLUX.1-dev",
    "description": "Black Forest Labs FLUX.1-dev model for high-quality image generation",
    "total_gpu_processes": 8,
    "available_gpu_processes": 8,
    "supported_features": ["text-to-image", "image-to-image", "fill", "controlnet"]
  }
]
```

## 使用示例

### Python 示例

```python
import requests
import base64
from PIL import Image
import io

# 服务地址
base_url = "http://localhost:12411"

# 1. JSON方式 (Base64) - 适合小图片
def text2img_json_example():
    payload = {
        "prompt": "A beautiful sunset over mountains, digital art",
        "mode": "text2img",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "cfg": 3.5,
        "seed": 42
    }
    
    response = requests.post(f"{base_url}/generate", json=payload)
    result = response.json()
    
    if result["success"]:
        # 保存生成的图片
        image_data = base64.b64decode(result["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image.save("generated_image.png")
        print(f"图片已保存，耗时: {result['elapsed_time']:.2f}秒")

# 2. Form-data方式 (文件上传) - 推荐用于大图片
def img2img_form_example():
    # 准备文件
    files = {
        'input_image': ('input.png', open('input_image.png', 'rb'), 'image/png')
    }
    
    data = {
        "prompt": "A beautiful landscape with mountains and trees",
        "mode": "img2img",
        "strength": "0.7",
        "height": "1024",
        "width": "1024",
        "num_inference_steps": "50",
        "cfg": "3.5",
        "seed": "42"
    }
    
    response = requests.post(f"{base_url}/generate/img2img", files=files, data=data)
    result = response.json()
    
    if result["success"]:
        # 保存生成的图片
        image_data = base64.b64decode(result["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image.save("img2img_result.png")
        print(f"图片已保存，耗时: {result['elapsed_time']:.2f}秒")

# 3. 填充/修复示例 (Form-data)
def fill_form_example():
    files = {
        'input_image': ('input.png', open('input_image.png', 'rb'), 'image/png'),
        'mask_image': ('mask.png', open('mask_image.png', 'rb'), 'image/png')
    }
    
    data = {
        "prompt": "A beautiful flower garden in the center",
        "mode": "fill",
        "height": "1024",
        "width": "1024",
        "num_inference_steps": "50",
        "cfg": "3.5",
        "seed": "42"
    }
    
    response = requests.post(f"{base_url}/generate/img2img", files=files, data=data)
    result = response.json()
    
    if result["success"]:
        image_data = base64.b64decode(result["image_base64"])
        image = Image.open(io.BytesIO(image_data))
        image.save("fill_result.png")
        print(f"图片已保存，耗时: {result['elapsed_time']:.2f}秒")
```

### cURL 示例

```bash
# JSON方式 - 文本生成图片
curl -X POST "http://localhost:12411/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A beautiful sunset over mountains",
       "mode": "text2img",
       "height": 1024,
       "width": 1024,
       "num_inference_steps": 50,
       "cfg": 3.5,
       "seed": 42
     }'

# Form-data方式 - 图片生成图片 (推荐)
curl -X POST "http://localhost:12411/generate/img2img" \
     -F "prompt=A beautiful landscape" \
     -F "mode=img2img" \
     -F "input_image=@input_image.png" \
     -F "strength=0.7" \
     -F "height=1024" \
     -F "width=1024"

# Form-data方式 - 填充/修复
curl -X POST "http://localhost:12411/generate/img2img" \
     -F "prompt=A beautiful flower garden" \
     -F "mode=fill" \
     -F "input_image=@input_image.png" \
     -F "mask_image=@mask_image.png" \
     -F "height=1024" \
     -F "width=1024"
```

## 图片格式要求

### 输入图片格式
- **格式**: PNG, JPEG, JPG
- **编码**: Base64 (JSON方式) 或 二进制文件 (Form-data方式)
- **尺寸**: 建议与输出尺寸一致，否则会自动调整
- **颜色模式**: RGB

### 蒙版图片格式 (fill模式)
- **格式**: PNG, JPEG, JPG
- **编码**: Base64 (JSON方式) 或 二进制文件 (Form-data方式)
- **颜色**: 白色区域(255,255,255)表示需要填充的区域，黑色区域(0,0,0)表示保持不变
- **尺寸**: 必须与输入图片一致

### 控制图片格式 (controlnet模式)
- **格式**: PNG, JPEG, JPG
- **编码**: Base64 (JSON方式) 或 二进制文件 (Form-data方式)
- **类型**: 根据ControlNet模型类型而定（如Canny边缘、深度图等）
- **尺寸**: 建议与输出尺寸一致

## 性能优化建议

### 1. 上传方式选择
- **小图片 (< 512x512)**: 可以使用JSON方式，简单方便
- **大图片 (≥ 1024x1024)**: 推荐使用Form-data方式，性能更好
- **批量处理**: 使用Form-data方式，减少内存占用

### 2. 参数调优
- **num_inference_steps**: 减少步数可提高速度，但可能影响质量
- **height/width**: 较小的尺寸生成更快
- **cfg**: 较低的值生成更快，但可能偏离提示词

### 3. 批量处理
- 使用不同的seed值生成多个变体
- 利用优先级参数控制任务顺序

### 4. 错误处理
- 设置合理的超时时间
- 处理GPU内存不足等错误
- 实现重试机制

## 错误代码

| HTTP状态码 | 错误描述 | 解决方案 |
|------------|----------|----------|
| 400 | 参数验证失败 | 检查必需参数和参数格式 |
| 503 | 服务未就绪 | 等待服务启动完成 |
| 500 | 内部服务器错误 | 检查服务日志 |
| 408 | 请求超时 | 增加超时时间或减少参数 |

## 监控和调试

### 1. 服务状态监控
```bash
curl http://localhost:12411/status
```

### 2. 性能监控
```bash
python performance_monitor.py --continuous 30
```

### 3. 上传方式性能测试
```bash
# 简单性能对比
python test_upload_methods.py simple

# 完整性能分析
python test_upload_methods.py full
```

### 4. 图生图功能测试
```bash
# 测试所有图生图功能
python test_img2img.py

# 测试特定模式
python test_img2img.py text2img
python test_img2img.py img2img
python test_img2img.py fill
python test_img2img.py controlnet
```

## 注意事项

1. **模型支持**: 确保已下载相应的Flux模型文件
2. **GPU内存**: 大尺寸图片需要更多GPU内存
3. **并发限制**: 每个GPU进程同时只能处理一个任务
4. **上传方式**: 
   - Base64会增加约33%的数据大小
   - Form-data直接传输二进制数据，性能更好
5. **网络传输**: 大图片的传输可能需要较长时间
6. **文件大小限制**: 建议单个图片文件不超过50MB 