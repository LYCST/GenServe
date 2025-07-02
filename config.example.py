# GenServe 配置示例
# 复制此文件为 config_local.py 并根据需要修改配置

import os
from typing import Dict, Any, List

class ConfigExample:
    """配置示例 - 展示如何自定义配置"""
    
    # 服务配置
    HOST = "0.0.0.0"
    PORT = 8000
    
    # 设备配置
    USE_GPU = True
    TORCH_DTYPE = "float16"  # float16, float32, bfloat16
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 安全配置
    MAX_PROMPT_LENGTH = 1000
    MAX_IMAGE_SIZE = 2048
    MAX_CONCURRENT_REQUESTS = 10
    
    # 模型管理配置
    AUTO_LOAD_MODELS = True
    MODEL_CACHE_DIR = None  # 模型缓存目录
    MODEL_DOWNLOAD_TIMEOUT = 300  # 下载超时时间（秒）
    
    # 性能配置
    ENABLE_OPTIMIZATION = True
    MEMORY_EFFICIENT_ATTENTION = True
    
    # API配置
    ENABLE_CORS = True
    CORS_ORIGINS = ["*"]  # 允许的域名列表
    API_PREFIX = ""
    
    # 监控配置
    ENABLE_METRICS = False
    METRICS_PORT = 9090
    
    # 自定义模型配置示例
    CUSTOM_MODELS = {
        "flux1-dev": {
            "enabled": True,
            "auto_load": True,
            "max_batch_size": 1
        },
        "sdxl-base": {
            "enabled": False,  # 禁用SDXL模型
            "auto_load": False,
            "max_batch_size": 1
        }
    }
    
    @classmethod
    def get_service_config(cls) -> Dict[str, Any]:
        """获取服务配置"""
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "api_prefix": cls.API_PREFIX,
            "enable_cors": cls.ENABLE_CORS,
            "cors_origins": cls.CORS_ORIGINS
        }
    
    @classmethod
    def get_device_config(cls) -> Dict[str, Any]:
        """获取设备配置"""
        return {
            "use_gpu": cls.USE_GPU,
            "torch_dtype": cls.TORCH_DTYPE,
            "enable_optimization": cls.ENABLE_OPTIMIZATION,
            "memory_efficient_attention": cls.MEMORY_EFFICIENT_ATTENTION
        }
    
    @classmethod
    def get_model_management_config(cls) -> Dict[str, Any]:
        """获取模型管理配置"""
        return {
            "auto_load_models": cls.AUTO_LOAD_MODELS,
            "model_cache_dir": cls.MODEL_CACHE_DIR,
            "model_download_timeout": cls.MODEL_DOWNLOAD_TIMEOUT,
            "custom_models": cls.CUSTOM_MODELS
        }
    
    @classmethod
    def get_security_config(cls) -> Dict[str, Any]:
        """获取安全配置"""
        return {
            "max_prompt_length": cls.MAX_PROMPT_LENGTH,
            "max_image_size": cls.MAX_IMAGE_SIZE,
            "max_concurrent_requests": cls.MAX_CONCURRENT_REQUESTS
        }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            "level": cls.LOG_LEVEL,
            "format": cls.LOG_FORMAT
        }
    
    @classmethod
    def get_monitoring_config(cls) -> Dict[str, Any]:
        """获取监控配置"""
        return {
            "enable_metrics": cls.ENABLE_METRICS,
            "metrics_port": cls.METRICS_PORT
        }
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "service": cls.get_service_config(),
            "device": cls.get_device_config(),
            "model_management": cls.get_model_management_config(),
            "security": cls.get_security_config(),
            "logging": cls.get_logging_config(),
            "monitoring": cls.get_monitoring_config()
        }
    
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
        import torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(cls.TORCH_DTYPE, torch.float16) 