# GenServe API Curl 请求指南

## 概述

GenServe 是一个基于Flux模型的多GPU并发图片生成服务，支持多种生成模式和图片上传方式。

**服务地址**: `http://localhost:12411`

## 0. 认证说明

### 0.1 API密钥认证

所有API请求都需要提供有效的API密钥。支持两种认证方式：

#### 方式1: Authorization Header (推荐)
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" ...
```

#### 方式2: X-API-Key Header
```bash
curl -H "X-API-Key: YOUR_API_KEY" ...
```

### 0.2 权限级别

| 权限 | 说明 | 可访问的接口 |
|------|------|-------------|
| `generation` | 图片生成权限 | `/generate`, `/generate/upload` |
| `readonly` | 只读权限 | `/`, `/health`, `/status`, `/models`, `/loras`, `/task/{id}` |
| `admin` | 管理员权限 | 所有接口 + `/auth/keys`, `/auth/generate-key` |
| `all` | 所有权限 | 所有接口 |

### 0.3 配置API密钥

在启动脚本中配置API密钥：

```bash
# 格式：key:name:permissions
export API_KEY_1="abc123def456:developer:generation,readonly"
export API_KEY_2="xyz789ghi012:user:generation"
export API_KEY_3="admin123admin456:admin:all"
```

### 0.4 默认密钥

如果未配置任何API密钥，系统将使用默认密钥：
- **默认密钥**: `genserve-default-key-2024`
- **权限**: `all`

## 1. 基础信息接口

### 1.1 获取服务状态
```bash
curl -X GET "http://localhost:12411/" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 1.2 健康检查
```bash
curl -X GET "http://localhost:12411/health" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 1.3 获取服务状态详情
```bash
curl -X GET "http://localhost:12411/status" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 1.4 获取支持的模型列表
```bash
curl -X GET "http://localhost:12411/models" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 1.5 获取LoRA模型列表
```bash
curl -X GET "http://localhost:12411/loras" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 1.6 获取API密钥列表（仅管理员）
```bash
curl -X GET "http://localhost:12411/auth/keys" \
  -H "Authorization: Bearer ADMIN_API_KEY"
```

### 1.7 生成新的API密钥（仅管理员）
```bash
curl -X POST "http://localhost:12411/auth/generate-key" \
  -H "Authorization: Bearer ADMIN_API_KEY" \
  -F "name=新用户" \
  -F "permissions=generation,readonly"
```

### 1.8 删除API密钥（仅管理员）

#### 1.8.1 通过完整密钥删除
```bash
curl -X POST "http://localhost:12411/auth/delete-key" \
  -H "Authorization: Bearer ADMIN_API_KEY" \
  -F "api_key=要删除的完整API密钥"
```

#### 1.8.2 通过key_id删除（推荐）
```bash
curl -X POST "http://localhost:12411/auth/delete-key-by-id" \
  -H "Authorization: Bearer ADMIN_API_KEY" \
  -F "key_id=前8位-后4位"
```

#### 1.8.3 通过用户名删除（不推荐，容易误删）
```bash
curl -X POST "http://localhost:12411/auth/delete-key-by-name" \
  -H "Authorization: Bearer ADMIN_API_KEY" \
  -F "name=要删除的用户名"
```

### 1.9 重新加载API密钥（仅管理员）
```bash
curl -X POST "http://localhost:12411/auth/reload-keys" \
  -H "Authorization: Bearer ADMIN_API_KEY"
```

## 2. 图片生成接口

### 2.1 JSON格式生成 (Base64图片)

#### 2.1.1 文本生成图片 (Text-to-Image)
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫坐在花园里",
    "model_id": "flux1-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "text2img"
  }'
```

#### 2.1.2 图片生成图片 (Image-to-Image)
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫坐在花园里",
    "model_id": "flux1-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "img2img",
    "strength": 0.8,
    "input_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }'
```

#### 2.1.3 图片填充 (Fill/Inpainting)
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫",
    "model_id": "flux1-fill-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "fill",
    "input_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "mask_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }'
```

#### 2.1.4 ControlNet深度控制
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫坐在花园里",
    "model_id": "flux1-depth-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "controlnet",
    "controlnet_type": "depth",
    "control_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }'
```

#### 2.1.5 ControlNet边缘控制
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫坐在花园里",
    "model_id": "flux1-canny-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "controlnet",
    "controlnet_type": "canny",
    "control_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }'
```

#### 2.1.6 ControlNet姿态控制
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫坐在花园里",
    "model_id": "flux1-openpose-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "controlnet",
    "controlnet_type": "openpose",
    "control_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }'
```

#### 2.1.7 Redux图像增强
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "高质量图像",
    "model_id": "flux1-redux-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "redux",
    "input_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  }'
```

#### 2.1.8 使用LoRA模型
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫坐在花园里",
    "model_id": "flux1-dev",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "cfg": 3.5,
    "seed": 42,
    "priority": 0,
    "mode": "text2img",
    "loras": [
      {
        "name": "cute_cat_lora",
        "weight": 0.8
      }
    ]
  }'
```

### 2.2 Form-data格式生成 (文件上传)

#### 2.2.1 文本生成图片
```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "prompt=一只可爱的小猫坐在花园里" \
  -F "model_id=flux1-dev" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=50" \
  -F "cfg=3.5" \
  -F "seed=42" \
  -F "priority=0" \
  -F "mode=text2img"
```

#### 2.2.2 图片生成图片
```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "prompt=一只可爱的小猫坐在花园里" \
  -F "model_id=flux1-dev" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=50" \
  -F "cfg=3.5" \
  -F "seed=42" \
  -F "priority=0" \
  -F "mode=img2img" \
  -F "strength=0.8" \
  -F "input_image=@/path/to/input_image.png"
```

#### 2.2.3 图片填充
```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "prompt=一只可爱的小猫" \
  -F "model_id=flux1-fill-dev" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=50" \
  -F "cfg=3.5" \
  -F "seed=42" \
  -F "priority=0" \
  -F "mode=fill" \
  -F "input_image=@/path/to/input_image.png" \
  -F "mask_image=@/path/to/mask_image.png"
```

#### 2.2.4 ControlNet深度控制
```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "prompt=一只可爱的小猫坐在花园里" \
  -F "model_id=flux1-depth-dev" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=50" \
  -F "cfg=3.5" \
  -F "seed=42" \
  -F "priority=0" \
  -F "mode=controlnet" \
  -F "controlnet_type=depth" \
  -F "control_image=@/path/to/depth_image.png"
```

#### 2.2.5 使用LoRA模型 (Form-data)
```bash
curl -X POST "http://localhost:12411/generate/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "prompt=一只可爱的小猫坐在花园里" \
  -F "model_id=flux1-dev" \
  -F "height=1024" \
  -F "width=1024" \
  -F "num_inference_steps=50" \
  -F "cfg=3.5" \
  -F "seed=42" \
  -F "priority=0" \
  -F "mode=text2img" \
  -F 'loras=[{"name":"cute_cat_lora","weight":0.8}]'
```

## 3. 任务管理接口

### 3.1 获取任务结果
```bash
curl -X GET "http://localhost:12411/task/{task_id}" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

示例：
```bash
curl -X GET "http://localhost:12411/task/abc123def456" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 4. 参数说明

### 4.1 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | string | 必需 | 文本提示词 |
| `model_id` | string | "flux1-dev" | 模型ID |
| `height` | int | 1024 | 输出图片高度 |
| `width` | int | 1024 | 输出图片宽度 |
| `num_inference_steps` | int | 50 | 推理步数 (1-100) |
| `cfg` | float | 3.5 | 引导强度 (0-50) |
| `seed` | int | 42 | 随机种子 |
| `priority` | int | 0 | 任务优先级 |
| `mode` | string | "text2img" | 生成模式 |

### 4.2 生成模式

| 模式 | 说明 | 必需参数 |
|------|------|----------|
| `text2img` | 文本生成图片 | prompt |
| `img2img` | 图片生成图片 | prompt, input_image |
| `fill` | 图片填充 | prompt, input_image, mask_image |
| `controlnet` | 控制网络 | prompt, control_image |
| `redux` | 图像增强 | prompt, input_image |

### 4.3 支持的模型

| 模型ID | 说明 | 支持的模式 |
|--------|------|------------|
| `flux1-dev` | 基础Flux模型 | text2img, img2img |
| `flux1-fill-dev` | 填充模型 | text2img, img2img, fill |
| `flux1-depth-dev` | 深度控制模型 | controlnet |
| `flux1-canny-dev` | 边缘控制模型 | controlnet |
| `flux1-openpose-dev` | 姿态控制模型 | controlnet |
| `flux1-redux-dev` | 图像增强模型 | redux |

### 4.4 ControlNet类型

| 类型 | 说明 |
|------|------|
| `depth` | 深度图控制 |
| `canny` | 边缘图控制 |
| `openpose` | 姿态图控制 |

### 4.5 LoRA配置

```json
{
  "loras": [
    {
      "name": "lora_model_name",
      "weight": 0.8
    }
  ]
}
```

## 5. 响应格式

### 5.1 成功响应
```json
{
  "success": true,
  "task_id": "abc123def456",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "metadata": {
    "prompt": "一只可爱的小猫",
    "model_id": "flux1-dev",
    "mode": "text2img",
    "generation_time": 45.2,
    "seed": 42,
    "cfg": 3.5,
    "steps": 50
  },
  "auth_info": {
    "user": "developer",
    "permissions": ["generation", "readonly"]
  }
}
```

### 5.2 错误响应
```json
{
  "success": false,
  "error": "错误信息",
  "detail": "详细错误信息"
}
```

### 5.3 认证错误响应
```json
{
  "detail": "缺少API密钥。请在Authorization header中使用Bearer token或在X-API-Key header中提供密钥"
}
```

## 6. 使用示例

### 6.1 批量生成脚本
```bash
#!/bin/bash

# 设置API密钥
API_KEY="your-api-key-here"

# 批量生成不同风格的图片
prompts=(
  "一只可爱的小猫"
  "一只威武的狮子"
  "一只优雅的天鹅"
)

for prompt in "${prompts[@]}"; do
  echo "生成: $prompt"
  curl -X POST "http://localhost:12411/generate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "{
      \"prompt\": \"$prompt\",
      \"model_id\": \"flux1-dev\",
      \"height\": 1024,
      \"width\": 1024,
      \"num_inference_steps\": 50,
      \"cfg\": 3.5,
      \"seed\": 42,
      \"mode\": \"text2img\"
    }" | jq '.image' | base64 -d > "output_${prompt// /_}.png"
done
```

### 6.2 监控任务状态
```bash
#!/bin/bash

# 设置API密钥
API_KEY="your-api-key-here"

# 提交任务并监控状态
response=$(curl -s -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫",
    "mode": "text2img"
  }')

task_id=$(echo $response | jq -r '.task_id')

echo "任务ID: $task_id"

# 轮询任务状态
while true; do
  result=$(curl -s -X GET "http://localhost:12411/task/$task_id" \
    -H "Authorization: Bearer $API_KEY")
  status=$(echo $result | jq -r '.status')
  
  echo "任务状态: $status"
  
  if [ "$status" = "completed" ]; then
    echo "任务完成!"
    echo $result | jq '.image' | base64 -d > "output.png"
    break
  elif [ "$status" = "failed" ]; then
    echo "任务失败!"
    break
  fi
  
  sleep 5
done
```

### 6.3 生成新的API密钥
```bash
#!/bin/bash

# 管理员API密钥
ADMIN_KEY="admin-key-2024"

# 生成新用户密钥
curl -X POST "http://localhost:12411/auth/generate-key" \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -F "name=新用户" \
  -F "permissions=generation,readonly" | jq '.api_key'
```

## 7. 注意事项

1. **API密钥**: 所有请求都必须提供有效的API密钥
2. **权限控制**: 不同权限级别可以访问不同的接口
3. **速率限制**: 每个API密钥每分钟最多100个请求
4. **图片格式**: 支持PNG、JPG等常见格式
5. **Base64编码**: 图片需要转换为base64格式，包含data URI前缀
6. **文件大小**: 建议单个图片文件不超过10MB
7. **并发限制**: 根据GPU数量和配置，系统会自动管理并发任务
8. **超时设置**: 任务默认超时时间为240秒
9. **优先级**: 数值越小优先级越高

## 8. 错误处理

常见错误及解决方案：

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| 401 Unauthorized | 缺少或无效的API密钥 | 检查API密钥是否正确 |
| 403 Forbidden | 权限不足 | 检查API密钥权限是否足够 |
| 429 Too Many Requests | 请求频率过高 | 降低请求频率或等待 |
| 503 Service Unavailable | 服务未就绪 | 等待服务启动完成 |
| 400 Bad Request | 参数错误 | 检查参数格式和必需字段 |
| 422 Validation Error | 参数验证失败 | 检查参数范围和格式 |
| 500 Internal Server Error | 服务器内部错误 | 查看服务日志 |

## 9. 性能优化建议

1. **使用合适的步数**: 50步通常能提供良好的质量和速度平衡
2. **调整CFG值**: 3.5-7.0是常用范围，值越高越符合提示词但可能过于刻板
3. **合理设置优先级**: 重要任务设置较低优先级数值
4. **批量处理**: 使用脚本批量提交任务以提高效率
5. **监控资源**: 定期检查GPU使用情况和内存状态
6. **API密钥管理**: 为不同用户分配不同权限的API密钥 

## 10. API密钥文件加密

### 10.1 加密功能概述

GenServe支持对API密钥文件进行加密存储，提高安全性：

- **加密算法**: Fernet对称加密
- **加密文件**: `api_keys.enc` (替代明文 `api_keys.json`)
- **自动处理**: 系统自动加密/解密，对用户透明
- **向后兼容**: 支持从明文文件自动迁移到加密文件

### 10.2 启用加密

#### 10.2.1 生成加密密钥
```bash
python generate_encryption_key.py
```

#### 10.2.2 配置加密密钥
在启动脚本中添加：
```bash
export API_KEYS_ENCRYPTION_KEY="your-44-character-encryption-key-here"
```

#### 10.2.3 重启服务
```bash
./start_optimized.sh
```

### 10.3 加密方式

#### 10.3.1 自定义加密密钥（推荐）
```bash
# 生成随机密钥
python generate_encryption_key.py
# 选择选项1，将生成的密钥添加到启动脚本
```

#### 10.3.2 基于密码生成
```bash
python generate_encryption_key.py
# 选择选项2，输入密码
```

#### 10.3.3 基于默认API密钥生成
```bash
python generate_encryption_key.py
# 选择选项3，自动基于DEFAULT_API_KEY生成
```

### 10.4 安全建议

1. **密钥管理**: 将加密密钥保存在安全的地方
2. **版本控制**: 不要将加密密钥提交到Git等版本控制系统
3. **定期更换**: 定期更换加密密钥
4. **备份**: 备份加密密钥，丢失后将无法解密API密钥文件
5. **权限控制**: 确保加密文件只有服务进程可以访问

### 10.5 测试加密功能
```bash
python test_encryption.py
```

### 10.6 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 无法解密文件 | 加密密钥错误 | 检查API_KEYS_ENCRYPTION_KEY是否正确 |
| 加密密钥长度错误 | 密钥格式不正确 | 确保密钥为44个字符的base64编码 |
| 文件权限错误 | 文件访问权限不足 | 检查文件权限设置 |
| 加密功能未启用 | 缺少cryptography库 | 安装: `pip install cryptography` | 