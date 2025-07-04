# LoRA API 使用示例

本文档提供了LoRA相关API的使用示例和说明。

## 配置说明

### 环境变量配置

在启动服务前，可以通过环境变量配置LoRA路径：

```bash
# 设置LoRA基础路径
export LORA_BASE_PATH="/home/shuzuan/prj/models/loras"

# 启动服务
./start.sh
```

### 启动脚本配置

在 `start.sh` 中配置LoRA路径：

```bash
# LoRA模型路径 (可选，如果不配置则使用默认路径)
export LORA_BASE_PATH="/home/shuzuan/prj/models/loras"
```

## API 接口

### 1. 获取LoRA列表

**端点**: `GET /loras`

**描述**: 获取所有可用的LoRA模型列表

**请求示例**:

```bash
curl -X GET "http://localhost:12411/loras"
```

**响应格式**:

```json
{
  "success": true,
  "loras": [
    {
      "name": "example_lora",
      "path": "category1/example_lora.safetensors",
      "full_path": "/home/shuzuan/prj/models/loras/category1/example_lora.safetensors",
      "size_mb": 144.5,
      "type": "safetensors"
    },
    {
      "name": "another_lora",
      "path": "category2/another_lora.safetensors",
      "full_path": "/home/shuzuan/prj/models/loras/category2/another_lora.safetensors",
      "size_mb": 67.2,
      "type": "safetensors"
    }
  ],
  "total_count": 2,
  "base_path": "/home/shuzuan/prj/models/loras"
}
```

**错误响应**:

```json
{
  "success": false,
  "error": "LoRA路径不存在",
  "loras": [],
  "total_count": 0,
  "base_path": "/home/shuzuan/prj/models/loras"
}
```

## 文件结构支持

### 支持的文件夹结构

LoRA API支持多层级文件夹结构，例如：

```
/home/shuzuan/prj/models/loras/
├── character/
│   ├── anime_girl.safetensors
│   └── robot_boy.safetensors
├── style/
│   ├── oil_painting.safetensors
│   └── watercolor.safetensors
├── object/
│   ├── car.safetensors
│   └── building.safetensors
└── single_lora.safetensors
```

### 文件要求

- **文件格式**: 仅支持 `.safetensors` 文件
- **文件大小**: 自动计算并显示文件大小（MB）
- **文件路径**: 返回相对于基础路径的相对路径

## 使用示例

### 1. 获取所有LoRA

```bash
curl -X GET "http://localhost:12411/loras" | jq
```

### 2. 获取LoRA数量

```bash
curl -X GET "http://localhost:12411/loras" | jq '.total_count'
```

### 3. 获取特定类别的LoRA

```bash
# 获取所有LoRA，然后过滤特定类别
curl -X GET "http://localhost:12411/loras" | jq '.loras[] | select(.path | startswith("character/"))'
```

### 4. 按大小排序

```bash
# 按文件大小排序（从大到小）
curl -X GET "http://localhost:12411/loras" | jq '.loras | sort_by(.size_mb) | reverse'
```

## 测试脚本

使用提供的测试脚本验证LoRA功能：

```bash
# 运行LoRA API测试
python3 test_lora_api.py
```

测试脚本会：
1. 检查服务状态
2. 验证LoRA文件结构
3. 测试LoRA列表API
4. 显示使用说明

## 注意事项

1. **路径配置**: 确保 `LORA_BASE_PATH` 环境变量正确设置
2. **文件权限**: 确保服务有读取LoRA文件夹的权限
3. **文件格式**: 只扫描 `.safetensors` 文件，其他格式会被忽略
4. **性能考虑**: 大量LoRA文件可能影响扫描性能
5. **错误处理**: API会优雅处理路径不存在等错误情况

## 后续功能

当前版本提供LoRA列表功能，后续版本将支持：

1. **LoRA加载**: 在图片生成时加载指定的LoRA
2. **LoRA权重**: 支持设置LoRA权重参数
3. **LoRA组合**: 支持同时使用多个LoRA
4. **LoRA缓存**: 缓存已加载的LoRA以提高性能
5. **LoRA管理**: 提供LoRA的增删改查管理功能 