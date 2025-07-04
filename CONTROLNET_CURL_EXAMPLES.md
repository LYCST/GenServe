# ControlNet 调用示例

本文档提供了ControlNet模式的curl调用示例，展示了如何使用不同的ControlNet类型进行图片生成。

## 更新说明

根据官方示例，ControlNet模式现在使用以下推荐参数：
- `guidance_scale`: 10.0 (推荐值)
- `num_inference_steps`: 30 (推荐值)

## 1. Depth ControlNet

使用深度图控制生成图片的空间结构。

### JSON方式调用

```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
    "mode": "controlnet",
    "controlnet_type": "depth",
    "input_image": "data:image/png;base64,YOUR_INPUT_IMAGE_BASE64",
    "control_image": "data:image/png;base64,YOUR_DEPTH_IMAGE_BASE64",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "cfg": 10.0,
    "seed": 42,
    "model_id": "flux1-dev"
  }'
```

### Form-data方式调用

```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -F "prompt=A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts." \
  -F "mode=controlnet" \
  -F "controlnet_type=depth" \
  -F "input_image=@input.jpg" \
  -F "control_image=@depth_map.png" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=30" \
  -F "cfg=10.0" \
  -F "seed=42" \
  -F "model_id=flux1-dev"
```

## 2. Canny ControlNet

使用Canny边缘检测图控制生成图片的轮廓。

### JSON方式调用

```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic city with skyscrapers and flying cars",
    "mode": "controlnet",
    "controlnet_type": "canny",
    "input_image": "data:image/png;base64,YOUR_INPUT_IMAGE_BASE64",
    "control_image": "data:image/png;base64,YOUR_CANNY_EDGES_BASE64",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "cfg": 10.0,
    "seed": 42,
    "model_id": "flux1-dev"
  }'
```

### Form-data方式调用

```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -F "prompt=A futuristic city with skyscrapers and flying cars" \
  -F "mode=controlnet" \
  -F "controlnet_type=canny" \
  -F "input_image=@input.jpg" \
  -F "control_image=@canny_edges.png" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=30" \
  -F "cfg=10.0" \
  -F "seed=42" \
  -F "model_id=flux1-dev"
```

## 3. OpenPose ControlNet

使用人体姿态图控制生成图片中的人物姿态。

### JSON方式调用

```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A person dancing in a beautiful garden",
    "mode": "controlnet",
    "controlnet_type": "openpose",
    "input_image": "data:image/png;base64,YOUR_INPUT_IMAGE_BASE64",
    "control_image": "data:image/png;base64,YOUR_POSE_IMAGE_BASE64",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "cfg": 10.0,
    "seed": 42,
    "model_id": "flux1-dev"
  }'
```

### Form-data方式调用

```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -F "prompt=A person dancing in a beautiful garden" \
  -F "mode=controlnet" \
  -F "controlnet_type=openpose" \
  -F "input_image=@input.jpg" \
  -F "control_image=@pose_image.png" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=30" \
  -F "cfg=10.0" \
  -F "seed=42" \
  -F "model_id=flux1-dev"
```

## 4. 纯ControlNet生成（无需input_image）

ControlNet模式可以直接从控制图片生成图片，无需提供input_image。

### JSON方式调用

```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "mode": "controlnet",
    "controlnet_type": "depth",
    "control_image": "data:image/png;base64,YOUR_DEPTH_IMAGE_BASE64",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "cfg": 10.0,
    "seed": 42
  }'
```

### Form-data方式调用

```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -F "prompt=A beautiful landscape" \
  -F "mode=controlnet" \
  -F "controlnet_type=depth" \
  -F "control_image=@depth_map.png" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=30" \
  -F "cfg=10.0" \
  -F "seed=42" \
  -F "model_id=flux1-dev"
```

## 参数说明

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `prompt` | string | 是 | - | 生成提示词 |
| `mode` | string | 是 | "text2img" | 生成模式，必须为 "controlnet" |
| `controlnet_type` | string | 是 | "depth" | ControlNet类型：depth/canny/openpose |
| `input_image` | string/file | 否 | - | 输入图片（base64或文件），controlnet模式下可选 |
| `control_image` | string/file | 是 | - | 控制图片（base64或文件），controlnet模式必需 |
| `height` | int | 否 | 1024 | 输出图片高度 |
| `width` | int | 否 | 1024 | 输出图片宽度 |
| `num_inference_steps` | int | 否 | 30 | 推理步数（推荐30） |
| `cfg` | float | 否 | 10.0 | 引导强度（推荐10.0） |
| `seed` | int | 否 | 42 | 随机种子 |
| `model_id` | string | 否 | "flux1-dev" | 模型ID |

## 控制图片要求

### Depth ControlNet
- 需要提供深度图（depth map）
- 可以使用深度估计工具生成，如MiDaS、DPT等
- 图片应该是灰度图，表示深度信息

### Canny ControlNet
- 需要提供Canny边缘检测图
- 可以使用OpenCV等工具生成
- 图片应该是黑白图，表示边缘信息

### OpenPose ControlNet
- 需要提供人体姿态图
- 可以使用OpenPose等工具生成
- 图片应该包含人体关键点和骨架信息

## 响应格式

```json
{
  "success": true,
  "task_id": "uuid-string",
  "image_base64": "base64_encoded_generated_image",
  "error": null,
  "elapsed_time": 15.23,
  "gpu_id": "cuda:0",
  "model_id": "flux1-dev",
  "mode": "controlnet",
  "controlnet_type": "depth"
}
```

## 注意事项

1. **预处理图片**: 用户需要提供已经预处理好的控制图片，服务不会自动进行预处理
2. **input_image可选**: ControlNet模式下，input_image是可选的，可以直接从control_image生成图片
3. **模型支持**: 确保配置了相应的ControlNet模型路径
4. **图片尺寸**: 控制图片的尺寸会影响生成效果，建议与输出尺寸匹配
5. **参数调优**: 可以根据具体需求调整 `cfg` 和 `num_inference_steps` 参数
6. **错误处理**: 如果控制图片格式不正确，会返回相应的错误信息 