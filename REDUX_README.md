# Redux功能使用说明

## 概述

GenServe现在支持Flux Redux功能，这是一个基于图片的生成模型，可以从输入图片生成高质量的变体。Redux使用两个pipeline：`FluxPriorReduxPipeline`和`FluxPipeline`，实现图片到图片的生成。

## 功能特点

- **图片到图片生成**：基于输入图片生成高质量变体
- **无需文本提示词**：主要依赖输入图片的内容
- **高质量输出**：使用Flux模型的高质量生成能力
- **支持多种输入格式**：JSON base64和Form-data文件上传

## 配置要求

### 环境变量配置

在`start.sh`中添加以下配置：

```bash
# Redux模型路径
export FLUX_REDUX_MODEL_PATH="/home/shuzuan/prj/models/FLUX.1-Redux-dev"

# Redux模型GPU分配
export FLUX_REDUX_GPUS="7,8"
```

### 模型路径要求

- **Redux模型路径**：必须指向包含`FluxPriorReduxPipeline`组件的模型目录
- **基础模型路径**：用于`FluxPipeline`，需要设置`text_encoder=None`和`text_encoder_2=None`

## API使用

### JSON格式请求

```json
{
  "prompt": "A beautiful landscape with mountains and trees, photorealistic",
  "mode": "redux",
  "input_image": "base64编码的输入图片",
  "height": 512,
  "width": 512,
  "num_inference_steps": 20,
  "cfg": 2.5,
  "seed": 42,
  "model_id": "flux1-redux-dev"
}
```

### Form-data格式请求

```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -F "prompt=A beautiful landscape with mountains and trees, photorealistic" \
  -F "mode=redux" \
  -F "input_image=@input_image.png" \
  -F "height=512" \
  -F "width=512" \
  -F "num_inference_steps=20" \
  -F "cfg=2.5" \
  -F "seed=42" \
  -F "model_id=flux1-redux-dev"
```

## 参数说明

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `mode` | string | 必须设置为"redux" |
| `input_image` | string/file | 输入图片（base64编码或文件） |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `height` | int | 1024 | 输出图片高度 |
| `width` | int | 1024 | 输出图片宽度 |
| `num_inference_steps` | int | 50 | 推理步数 |
| `cfg` | float | 2.5 | 引导强度（Redux推荐使用2.5） |
| `seed` | int | 42 | 随机种子 |
| `model_id` | string | "flux1-redux-dev" | 模型ID |

## 处理流程

Redux的处理流程分为两个步骤：

1. **Prior Pipeline处理**：
   - 使用`FluxPriorReduxPipeline`处理输入图片
   - 生成中间表示

2. **Base Pipeline处理**：
   - 使用`FluxPipeline`（无文本编码器）处理中间表示
   - 生成最终输出图片

## 使用示例

### Python示例

```python
import requests
import base64
from PIL import Image
import io

# 准备输入图片
input_image = Image.open("input.png")
buffer = io.BytesIO()
input_image.save(buffer, format="PNG")
input_base64 = base64.b64encode(buffer.getvalue()).decode()

# 发送请求
response = requests.post("http://localhost:12411/generate", json={
    "mode": "redux",
    "input_image": input_base64,
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "cfg": 2.5,
    "seed": 42
})

# 处理结果
if response.status_code == 200:
    result = response.json()
    if result["success"]:
        # 保存输出图片
        output_data = base64.b64decode(result["image_base64"])
        output_image = Image.open(io.BytesIO(output_data))
        output_image.save("redux_output.png")
        print("Redux生成成功！")
    else:
        print(f"生成失败: {result['error']}")
```

### JavaScript示例

```javascript
// 准备输入图片
const inputImage = document.getElementById('inputImage').files[0];
const reader = new FileReader();

reader.onload = function(e) {
    const base64 = e.target.result.split(',')[1];
    
    // 发送请求
    fetch('http://localhost:12411/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            mode: 'redux',
            input_image: base64,
            height: 512,
            width: 512,
            num_inference_steps: 20,
            cfg: 2.5,
            seed: 42
        })
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            // 显示输出图片
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + result.image_base64;
            document.body.appendChild(img);
        } else {
            console.error('生成失败:', result.error);
        }
    });
};

reader.readAsDataURL(inputImage);
```

## 注意事项

1. **模型路径**：确保Redux模型路径正确，包含所需的pipeline组件
2. **内存使用**：Redux需要加载两个pipeline，内存使用较高
3. **处理时间**：由于需要两个pipeline，处理时间可能较长
4. **输入图片质量**：输入图片质量会影响输出质量
5. **尺寸限制**：建议使用512x512或1024x1024等标准尺寸

## 错误处理

常见错误及解决方案：

- **模型加载失败**：检查模型路径和组件完整性
- **内存不足**：减少batch size或使用更小的图片尺寸
- **处理超时**：增加超时时间或减少推理步数

## 测试

使用提供的测试脚本验证Redux功能：

```bash
python3 test_redux.py
```

测试脚本会：
1. 创建测试输入图片
2. 发送Redux请求
3. 保存输出图片
4. 验证功能完整性 