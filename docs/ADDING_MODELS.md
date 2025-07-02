# 添加新模型指南

本文档介绍如何在GenServe中添加新的图片生成模型。

## 架构概述

GenServe使用插件化架构，每个模型都是独立的类，继承自`BaseModel`基类。这种设计使得：

- 模型之间完全解耦
- 可以轻松添加新模型
- 每个模型可以有自己的参数和配置
- 支持不同的模型类型和功能

## 添加新模型的步骤

### 1. 创建模型类

在`models/`目录下创建新的模型文件，例如`my_model.py`：

```python
from .base import BaseModel
from diffusers import DiffusionPipeline
import torch
from typing import Dict, Any

class MyModel(BaseModel):
    def __init__(self):
        super().__init__(
            model_id="my-model",
            model_name="My Custom Model",
            description="我的自定义模型描述"
        )
        self.model_path = "your/model/path"
    
    def load(self) -> bool:
        # 实现模型加载逻辑
        pass
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # 实现图片生成逻辑
        pass
    
    def get_default_params(self) -> Dict[str, Any]:
        # 返回默认参数
        pass
    
    def validate_params(self, **kwargs) -> bool:
        # 验证参数
        pass
    
    def get_supported_features(self) -> list:
        # 返回支持的功能
        pass
```

### 2. 实现必需的方法

#### `load()` 方法
负责加载模型到内存中：

```python
def load(self) -> bool:
    try:
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        
        self.is_loaded = True
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False
```

#### `generate()` 方法
负责生成图片：

```python
def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
    if not self.is_loaded:
        raise RuntimeError("模型未加载")
    
    # 获取和验证参数
    params = self.get_default_params()
    params.update(kwargs)
    
    if not self.validate_params(**params):
        raise ValueError("参数验证失败")
    
    # 生成图片
    result = self.pipe(prompt=prompt, **params)
    image = result.images[0]
    
    # 转换为base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "success": True,
        "image": image,
        "base64": img_base64,
        "elapsed_time": elapsed_time,
        "save_to_disk": save_to_disk,
        "params": params
    }
```

#### `get_default_params()` 方法
返回模型的默认参数：

```python
def get_default_params(self) -> Dict[str, Any]:
    return {
        "num_inference_steps": 20,
        "seed": 42,
        "cfg": 7.5,
        "height": 1024,
        "width": 1024,
        "save_disk_path": None
    }
```

#### `validate_params()` 方法
验证输入参数：

```python
def validate_params(self, **kwargs) -> bool:
    # 检查必需参数
    required_params = ['num_inference_steps', 'seed', 'cfg', 'height', 'width']
    for param in required_params:
        if param not in kwargs:
            return False
    
    # 验证参数范围
    if kwargs['num_inference_steps'] < 1 or kwargs['num_inference_steps'] > 100:
        return False
    
    return True
```

#### `get_supported_features()` 方法
返回模型支持的功能：

```python
def get_supported_features(self) -> list:
    return ["text-to-image", "image-to-image"]  # 根据实际功能返回
```

### 3. 注册模型

在`models/manager.py`中注册新模型：

```python
from .my_model import MyModel

class ModelManager:
    def _register_default_models(self):
        # 注册现有模型
        self.register_model(FluxModel())
        
        # 注册新模型
        self.register_model(MyModel())
```

### 4. 更新依赖

如果新模型需要额外的依赖，在`requirements.txt`中添加：

```
# 新模型需要的依赖
new-dependency==1.0.0
```

## 示例：添加SDXL模型

参考`models/sdxl_model.py`文件，这是一个完整的SDXL模型实现示例。

## 测试新模型

1. 启动服务：
```bash
python main.py
```

2. 检查模型是否已注册：
```bash
curl http://localhost:8000/models
```

3. 测试图片生成：
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "a beautiful landscape",
       "model": "your-model-id",
       "num_inference_steps": 20
     }'
```

## 最佳实践

1. **错误处理**：确保所有方法都有适当的错误处理
2. **日志记录**：使用logger记录重要的操作和错误
3. **参数验证**：严格验证输入参数，防止无效请求
4. **资源管理**：正确管理GPU内存，及时释放不需要的资源
5. **文档**：为每个方法添加清晰的文档字符串

## 支持的功能类型

目前支持的功能类型包括：
- `text-to-image`：文本到图片生成
- `image-to-image`：图片到图片转换
- `inpainting`：图片修复
- `outpainting`：图片扩展

根据模型的实际能力，在`get_supported_features()`中返回相应的功能列表。 