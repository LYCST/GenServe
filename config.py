import os
from typing import Dict, Any, List
import torch

class Config:
    """应用配置类 - 简化版本"""
    
    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 12411))
    
    # 设备配置
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")
    DEFAULT_GPU_DEVICE = os.getenv("DEFAULT_GPU_DEVICE", "0")
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 安全配置
    MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "1000"))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
    
    # 模型管理配置
    AUTO_LOAD_MODELS = os.getenv("AUTO_LOAD_MODELS", "true").lower() == "true"
    
    # 性能配置
    ENABLE_OPTIMIZATION = os.getenv("ENABLE_OPTIMIZATION", "true").lower() == "true"
    MEMORY_EFFICIENT_ATTENTION = os.getenv("MEMORY_EFFICIENT_ATTENTION", "true").lower() == "true"
    
    # API配置
    ENABLE_CORS = os.getenv("ENABLE_CORS", "true").lower() == "true"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # 模型GPU配置 - 每个模型可用的GPU列表
    MODEL_GPU_CONFIG = {
        "flux1-dev": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],  # Flux可以使用GPU 0,1,2,3
        # 可以添加更多模型配置
    }
    
    # 模型路径配置
    MODEL_PATHS = {
        "flux1-dev": "/home/shuzuan/prj/models/flux1-dev",
        # 可以添加更多模型路径
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "service": {
                "host": cls.HOST,
                "port": cls.PORT,
                "enable_cors": cls.ENABLE_CORS,
                "cors_origins": cls.CORS_ORIGINS
            },
            "device": {
                "use_gpu": cls.USE_GPU,
                "torch_dtype": cls.TORCH_DTYPE,
                "default_gpu_device": cls.DEFAULT_GPU_DEVICE,
                "enable_optimization": cls.ENABLE_OPTIMIZATION,
                "memory_efficient_attention": cls.MEMORY_EFFICIENT_ATTENTION
            },
            "model_management": {
                "auto_load_models": cls.AUTO_LOAD_MODELS,
                "model_gpu_config": cls.MODEL_GPU_CONFIG,
                "model_paths": cls.MODEL_PATHS
            },
            "security": {
                "max_prompt_length": cls.MAX_PROMPT_LENGTH,
                "max_image_size": cls.MAX_IMAGE_SIZE
            },
            "logging": {
                "level": cls.LOG_LEVEL
            }
        }
    
    @classmethod
    def get_model_gpu_config(cls, model_id: str) -> List[str]:
        """获取指定模型的GPU配置"""
        return cls.MODEL_GPU_CONFIG.get(model_id, ["cuda:0"])
    
    @classmethod
    def get_model_path(cls, model_id: str) -> str:
        """获取指定模型的路径"""
        return cls.MODEL_PATHS.get(model_id, "")
    
    @classmethod
    def validate_prompt(cls, prompt: str) -> bool:
        """验证提示词"""
        return len(prompt) <= cls.MAX_PROMPT_LENGTH and len(prompt) > 0
    
    @classmethod
    def validate_image_size(cls, height: int, width: int) -> bool:
        """验证图片尺寸"""
        return (0 < height <= cls.MAX_IMAGE_SIZE and 
                0 < width <= cls.MAX_IMAGE_SIZE)
    
    @classmethod
    def get_torch_dtype(cls):
        """获取PyTorch数据类型"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(cls.TORCH_DTYPE, torch.float16)
    
    @classmethod
    def validate_gpu_device(cls, device: str) -> bool:
        """验证GPU设备是否有效"""
        if device == "cpu":
            return True
        
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
                return torch.cuda.is_available() and gpu_id < torch.cuda.device_count()
            except (ValueError, IndexError):
                return False
        
        return False 