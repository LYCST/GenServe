import os
from typing import Dict, Any, List
import torch

class Config:
    """应用配置类 - 改进版本"""
    
    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 12411))
    
    # 设备配置
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    TORCH_DTYPE = os.getenv("TORCH_DTYPE", "bfloat16")  # Flux推荐使用bfloat16
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
    ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
    
    # API配置
    ENABLE_CORS = os.getenv("ENABLE_CORS", "true").lower() == "true"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # 并发配置
    MAX_GLOBAL_QUEUE_SIZE = int(os.getenv("MAX_GLOBAL_QUEUE_SIZE", "100"))
    MAX_GPU_QUEUE_SIZE = int(os.getenv("MAX_GPU_QUEUE_SIZE", "10"))
    TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))  # 5分钟
    SCHEDULER_SLEEP_TIME = float(os.getenv("SCHEDULER_SLEEP_TIME", "0.1"))
    
    # 模型GPU配置 - 支持更灵活的配置
    @classmethod
    def get_model_gpu_config(cls) -> Dict[str, List[str]]:
        """动态获取模型GPU配置"""
        # 从环境变量读取配置
        flux_gpus = os.getenv("FLUX_GPUS", "")
        if flux_gpus:
            gpu_list = [f"cuda:{gpu.strip()}" for gpu in flux_gpus.split(",") if gpu.strip().isdigit()]
        else:
            # 默认配置：使用所有可用GPU
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_list = [f"cuda:{i}" for i in range(gpu_count)]
            else:
                gpu_list = ["cpu"]
        
        return {
            "flux1-dev": gpu_list,
            # 可以添加更多模型配置
        }
    
    # 模型路径配置
    @classmethod
    def get_model_paths(cls) -> Dict[str, str]:
        """动态获取模型路径配置"""
        return {
            "flux1-dev": os.getenv("FLUX_MODEL_PATH", "/home/shuzuan/prj/models/flux1-dev"),
            # 可以添加更多模型路径
        }
    
    MODEL_PATHS = property(get_model_paths)
    
    # GPU负载均衡策略
    GPU_LOAD_BALANCE_STRATEGY = os.getenv("GPU_LOAD_BALANCE_STRATEGY", "queue_length")  # 可选: queue_length, memory_usage, round_robin
    
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
                "memory_efficient_attention": cls.MEMORY_EFFICIENT_ATTENTION,
                "enable_cpu_offload": cls.ENABLE_CPU_OFFLOAD
            },
            "model_management": {
                "auto_load_models": cls.AUTO_LOAD_MODELS,
                "model_gpu_config": cls.get_model_gpu_config(),
                "model_paths": cls.get_model_paths()
            },
            "concurrent": {
                "max_global_queue_size": cls.MAX_GLOBAL_QUEUE_SIZE,
                "max_gpu_queue_size": cls.MAX_GPU_QUEUE_SIZE,
                "task_timeout": cls.TASK_TIMEOUT,
                "scheduler_sleep_time": cls.SCHEDULER_SLEEP_TIME,
                "load_balance_strategy": cls.GPU_LOAD_BALANCE_STRATEGY
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
    def get_model_gpu_config_static(cls, model_id: str) -> List[str]:
        """获取指定模型的GPU配置（向后兼容）"""
        config = cls.get_model_gpu_config()
        return config.get(model_id, ["cuda:0" if torch.cuda.is_available() else "cpu"])
    
    @classmethod
    def get_model_path(cls, model_id: str) -> str:
        """获取指定模型的路径"""
        return cls.get_model_paths().get(model_id, "")
    
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
        return dtype_map.get(cls.TORCH_DTYPE, torch.bfloat16)
    
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
    
    @classmethod
    def get_available_gpus(cls) -> List[str]:
        """获取所有可用GPU设备"""
        if not torch.cuda.is_available():
            return ["cpu"]
        
        gpu_count = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(gpu_count)]
    
    @classmethod
    def auto_detect_gpu_config(cls) -> Dict[str, List[str]]:
        """自动检测GPU配置"""
        available_gpus = cls.get_available_gpus()
        
        if available_gpus == ["cpu"]:
            return {"flux1-dev": ["cpu"]}
        
        # 为每个模型分配所有可用GPU
        return {
            "flux1-dev": available_gpus
        }
    
    @classmethod
    def print_config_summary(cls):
        """打印配置摘要"""
        config = cls.get_config()
        
        print("=" * 50)
        print("GenServe 配置摘要")
        print("=" * 50)
        
        print(f"服务地址: {config['service']['host']}:{config['service']['port']}")
        print(f"GPU可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU数量: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        print(f"模型GPU配置:")
        for model_id, gpus in config['model_management']['model_gpu_config'].items():
            print(f"  {model_id}: {gpus}")
        
        print(f"并发配置:")
        print(f"  全局队列大小: {config['concurrent']['max_global_queue_size']}")
        print(f"  GPU队列大小: {config['concurrent']['max_gpu_queue_size']}")
        print(f"  任务超时: {config['concurrent']['task_timeout']}秒")
        print(f"  负载均衡策略: {config['concurrent']['load_balance_strategy']}")
        
        print("=" * 50) 