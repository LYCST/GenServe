#!/bin/bash

# LoRA API 调用示例脚本
# 演示如何使用LoRA进行图片生成

BASE_URL="http://localhost:12411"

echo "🎨 LoRA API 调用示例"
echo "===================="

# 1. 获取LoRA列表
echo ""
echo "1️⃣ 获取可用LoRA列表:"
curl -s "$BASE_URL/loras" | jq '.'

# 2. 单个LoRA生成
echo ""
echo "2️⃣ 单个LoRA生成示例:"
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful anime girl with blue hair, high quality, detailed",
    "mode": "text2img",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "cfg": 7.5,
    "seed": 42,
    "model_id": "flux1-dev",
    "loras": [
      {
        "name": "styles/anime_style.safetensors",
        "weight": 0.8
      }
    ]
  }' | jq '.'

# 3. 多个LoRA组合
echo ""
echo "3️⃣ 多个LoRA组合示例:"
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful anime girl in oil painting style, high quality",
    "mode": "text2img",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "cfg": 7.5,
    "seed": 42,
    "model_id": "flux1-dev",
    "loras": [
      {
        "name": "styles/anime_style.safetensors",
        "weight": 0.6
      },
      {
        "name": "styles/oil_painting.safetensors",
        "weight": 0.4
      }
    ]
  }' | jq '.'

# 4. LoRA + ControlNet
echo ""
echo "4️⃣ LoRA + ControlNet 示例:"
# 创建简单的测试图片 (base64编码的1x1像素图片)
TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"A robot made of exotic candies and chocolates, high quality, detailed\",
    \"mode\": \"controlnet\",
    \"controlnet_type\": \"depth\",
    \"control_image\": \"data:image/png;base64,$TEST_IMAGE\",
    \"height\": 1024,
    \"width\": 1024,
    \"num_inference_steps\": 30,
    \"cfg\": 10.0,
    \"seed\": 42,
    \"model_id\": \"flux1-dev\",
    \"loras\": [
      {
        \"name\": \"styles/realistic_style.safetensors\",
        \"weight\": 0.5
      }
    ]
  }" | jq '.'

# 5. Form-data格式 (文件上传)
echo ""
echo "5️⃣ Form-data格式示例:"
curl -X POST "$BASE_URL/generate/upload" \
  -F "prompt=A beautiful anime girl with blue hair, high quality, detailed" \
  -F "mode=text2img" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=30" \
  -F "cfg=7.5" \
  -F "seed=42" \
  -F "model_id=flux1-dev" \
  -F 'loras=[{"name":"styles/anime_style.safetensors","weight":0.8}]' | jq '.'

# 6. 错误处理示例
echo ""
echo "6️⃣ 错误处理示例 (无效LoRA):"
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test prompt",
    "mode": "text2img",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "cfg": 7.5,
    "seed": 42,
    "model_id": "flux1-dev",
    "loras": [
      {
        "name": "non_existent_lora",
        "weight": 0.8
      }
    ]
  }' | jq '.'

echo ""
echo "✅ LoRA API 示例调用完成"
echo ""
echo "📋 使用说明:"
echo "1. 确保服务正在运行 (http://localhost:12411)"
echo "2. 确保LoRA文件已放置在配置的路径中"
echo "3. 使用 /loras 接口获取可用LoRA列表"
echo "4. 根据实际LoRA文件名修改示例中的name参数"
echo "5. 调整weight参数控制LoRA影响强度 (0-2)" 