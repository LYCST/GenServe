#!/bin/bash

# GenServe API 调用示例
# 使用 curl 命令测试不同的生成模式

echo "🎨 GenServe API 调用示例"
echo "=========================="

# 服务地址
API_URL="http://localhost:12411"

# 检查服务状态
echo "📋 检查服务状态..."
echo "支持的模型："
curl -s "${API_URL}/models" | jq '.models[].model_id' 2>/dev/null || echo "无法获取模型列表"

echo -e "\n服务健康状态："
curl -s "${API_URL}/health" | jq '.' 2>/dev/null || echo "无法获取健康状态"

echo -e "\n================================\n"

# 示例1：基础文生图
echo "📝 示例1：基础文生图"
curl -X POST "${API_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains, vibrant colors, photorealistic",
    "model_id": "flux1-dev",
    "mode": "text2img",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20,
    "cfg": 3.5,
    "seed": 42
  }' | jq '.' 2>/dev/null || echo "请求失败"

echo -e "\n================================\n"

# 示例2：简单文生图（使用默认参数）
echo "📝 示例2：简单文生图"
curl -X POST "${API_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute cat sitting on a windowsill, soft lighting",
    "model_id": "flux1-dev"
  }' | jq '.' 2>/dev/null || echo "请求失败"

echo -e "\n================================\n"

# 示例3：图生图（不指定尺寸，使用图片原始尺寸）
echo "📝 示例3：图生图（使用图片原始尺寸）"
echo "使用Form-data格式，不指定height和width："
echo "curl -X POST \"${API_URL}/generate/upload\" \\"
echo "  -F \"prompt=enhance this image with better lighting\" \\"
echo "  -F \"model_id=flux1-dev\" \\"
echo "  -F \"mode=img2img\" \\"
echo "  -F \"strength=0.8\" \\"
echo "  -F \"input_image=@your_image.png\""

echo -e "\n================================\n"

# 示例4：图生图（指定尺寸）
echo "📝 示例4：图生图（指定尺寸）"
echo "使用Form-data格式，指定height和width："
echo "curl -X POST \"${API_URL}/generate/upload\" \\"
echo "  -F \"prompt=enhance this image with better lighting\" \\"
echo "  -F \"model_id=flux1-dev\" \\"
echo "  -F \"mode=img2img\" \\"
echo "  -F \"strength=0.8\" \\"
echo "  -F \"height=1024\" \\"
echo "  -F \"width=1024\" \\"
echo "  -F \"input_image=@your_image.png\""

echo -e "\n================================\n"

# 示例5：Fill模式（使用图片原始尺寸）
echo "📝 示例5：Fill模式（使用图片原始尺寸）"
echo "使用Form-data格式，不指定height和width："
echo "curl -X POST \"${API_URL}/generate/upload\" \\"
echo "  -F \"prompt=a white paper cup\" \\"
echo "  -F \"model_id=flux1-dev\" \\"
echo "  -F \"mode=fill\" \\"
echo "  -F \"input_image=@your_image.png\" \\"
echo "  -F \"mask_image=@your_mask.png\""

echo -e "\n================================\n"

# 示例6：ControlNet（使用控制图片原始尺寸）
echo "📝 示例6：ControlNet（使用控制图片原始尺寸）"
echo "使用Form-data格式，不指定height和width："
echo "curl -X POST \"${API_URL}/generate/upload\" \\"
echo "  -F \"prompt=a beautiful landscape\" \\"
echo "  -F \"model_id=flux1-dev\" \\"
echo "  -F \"mode=controlnet\" \\"
echo "  -F \"controlnet_type=depth\" \\"
echo "  -F \"input_image=@your_image.png\" \\"
echo "  -F \"control_image=@your_depth_map.png\""

echo -e "\n================================\n"

# 示例7：检查服务配置
echo "📝 示例7：检查服务配置"
echo "服务状态详情："
curl -s "${API_URL}/status" | jq '.' 2>/dev/null || echo "无法获取状态详情"

echo -e "\n🎉 示例完成！"
echo ""
echo "📋 使用说明："
echo "1. 文生图：使用 /generate 端点，JSON格式"
echo "2. 图生图：需要上传input_image，不指定尺寸则使用图片原始尺寸"
echo "3. Fill模式：需要上传input_image和mask_image，不指定尺寸则使用图片原始尺寸"
echo "4. ControlNet：需要上传input_image和control_image，不指定尺寸则使用控制图片原始尺寸"
echo ""
echo "🔧 尺寸处理规则："
echo "- 如果指定了height和width参数，使用指定的尺寸"
echo "- 如果没有指定尺寸参数，使用上传图片的原始尺寸"
echo "- Fill模式推荐使用guidance_scale=30"
echo ""
echo "🔧 故障排除："
echo "1. 检查服务是否正在运行"
echo "2. 检查模型是否正确加载"
echo "3. 检查端口是否正确 (默认: 12411)"
echo "4. Fill模式需要专门的Fill模型支持" 