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
    
    # PyTorch内存管理配置
    PYTORCH_CUDA_ALLOC_CONF = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6")
    
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
    MAX_CONCURRENT_REQUESTS = 10
    TASK_TIMEOUT = 180  # 任务超时时间（秒）
    SCHEDULER_SLEEP_TIME = 0.1  # 调度器睡眠时间（秒）
    MAX_GLOBAL_QUEUE_SIZE = 100  # 全局队列最大大小
    MAX_GPU_QUEUE_SIZE = 5  # 每个GPU队列最大大小
    GPU_TASK_CLEANUP_WAIT_TIME = 1.5  # GPU任务完成后清理等待时间（秒）
    GPU_MEMORY_THRESHOLD_MB = 800  # GPU内存阈值，超过此值强制清理（MB）
    GPU_MEMORY_CLEANUP_INTERVAL = 5  # GPU内存清理间隔（秒）
    ENABLE_AGGRESSIVE_CLEANUP = True  # 启用激进内存清理
    
    # 模型GPU配置 - 支持更灵活的配置
    @classmethod
    def get_model_gpu_config(cls) -> Dict[str, List[str]]:
        """动态获取模型GPU配置 - 只为已配置的模型分配GPU"""
        # 获取已配置的模型路径
        configured_models = cls.get_model_paths()
        
        # 如果没有配置任何模型，返回空配置
        if not configured_models:
            return {}
        
        # 从环境变量读取配置
        flux_gpus = os.getenv("FLUX_GPUS", "")
        flux_depth_gpus = os.getenv("FLUX_DEPTH_GPUS", "")
        flux_fill_gpus = os.getenv("FLUX_FILL_GPUS", "")
        flux_canny_gpus = os.getenv("FLUX_CANNY_GPUS", "")
        flux_openpose_gpus = os.getenv("FLUX_OPENPOSE_GPUS", "")
        
        # 解析GPU列表
        def parse_gpu_list(gpu_str: str) -> List[str]:
            if not gpu_str:
                return []
            return [f"cuda:{gpu.strip()}" for gpu in gpu_str.split(",") if gpu.strip().isdigit()]
        
        # 获取所有可用GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            all_gpus = [f"cuda:{i}" for i in range(gpu_count)]
        else:
            all_gpus = ["cpu"]
        
        # 为每个已配置的模型分配GPU
        config = {}
        
        # 基础模型 - 使用FLUX_GPUS或所有GPU
        if "flux1-dev" in configured_models:
            if flux_gpus:
                config["flux1-dev"] = parse_gpu_list(flux_gpus)
            else:
                config["flux1-dev"] = all_gpus
        
        # Depth模型 - 使用FLUX_DEPTH_GPUS或基础模型的GPU
        if "flux1-depth-dev" in configured_models:
            if flux_depth_gpus:
                config["flux1-depth-dev"] = parse_gpu_list(flux_depth_gpus)
            elif "flux1-dev" in config:
                config["flux1-depth-dev"] = config["flux1-dev"]
            else:
                config["flux1-depth-dev"] = all_gpus
        
        # Fill模型 - 使用FLUX_FILL_GPUS或基础模型的GPU
        if "flux1-fill-dev" in configured_models:
            if flux_fill_gpus:
                config["flux1-fill-dev"] = parse_gpu_list(flux_fill_gpus)
            elif "flux1-dev" in config:
                config["flux1-fill-dev"] = config["flux1-dev"]
            else:
                config["flux1-fill-dev"] = all_gpus
        
        # Canny模型 - 使用FLUX_CANNY_GPUS或基础模型的GPU
        if "flux1-canny-dev" in configured_models:
            if flux_canny_gpus:
                config["flux1-canny-dev"] = parse_gpu_list(flux_canny_gpus)
            elif "flux1-dev" in config:
                config["flux1-canny-dev"] = config["flux1-dev"]
            else:
                config["flux1-canny-dev"] = all_gpus
        
        # OpenPose模型 - 使用FLUX_OPENPOSE_GPUS或基础模型的GPU
        if "flux1-openpose-dev" in configured_models:
            if flux_openpose_gpus:
                config["flux1-openpose-dev"] = parse_gpu_list(flux_openpose_gpus)
            elif "flux1-dev" in config:
                config["flux1-openpose-dev"] = config["flux1-dev"]
            else:
                config["flux1-openpose-dev"] = all_gpus
        
        return config
    
    # 模型路径配置
    @classmethod
    def get_model_paths(cls) -> Dict[str, str]:
        """动态获取模型路径配置 - 只返回已配置的路径"""
        all_paths = {
            "flux1-dev": os.getenv("FLUX_MODEL_PATH"),
            "flux1-depth-dev": os.getenv("FLUX_DEPTH_MODEL_PATH"),
            "flux1-fill-dev": os.getenv("FLUX_FILL_MODEL_PATH"),
            "flux1-canny-dev": os.getenv("FLUX_CANNY_MODEL_PATH"),
            "flux1-openpose-dev": os.getenv("FLUX_OPENPOSE_MODEL_PATH"),
        }
        
        # 只返回已配置的路径
        configured_paths = {}
        for model_id, path in all_paths.items():
            if path:  # 只有配置了路径的模型才添加
                configured_paths[model_id] = path
        
        return configured_paths
    
    MODEL_PATHS = property(get_model_paths)
    
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
                "scheduler_sleep_time": cls.SCHEDULER_SLEEP_TIME
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
    def get_gpu_load_info(cls) -> Dict[str, Any]:
        """获取GPU负载信息"""
        if not torch.cuda.is_available():
            return {"cpu": {"available": True}}
        
        gpu_info = {}
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                total = torch.cuda.get_device_properties(i).total_memory / 1024**2
                gpu_info[f"cuda:{i}"] = {
                    "allocated_mb": allocated,
                    "total_mb": total,
                    "utilization": allocated / total if total > 0 else 0
                }
            except Exception as e:
                gpu_info[f"cuda:{i}"] = {"error": str(e)}
        
        return gpu_info
    
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
        
        print("=" * 50)