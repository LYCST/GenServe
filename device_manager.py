import torch
import logging
from typing import Dict, Any, List, Optional
from config import Config

logger = logging.getLogger(__name__)

class DeviceManager:
    """设备管理器，专门处理GPU设备相关功能"""
    
    @staticmethod
    def get_available_devices() -> Dict[str, Any]:
        """获取可用设备信息"""
        devices = {
            "cpu": {
                "available": True,
                "type": "cpu"
            }
        }
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                
                devices[f"cuda:{i}"] = {
                    "available": True,
                    "type": "gpu",
                    "name": device_name,
                    "total_memory_mb": total_memory / 1024**2
                }
        
        return devices
    
    @staticmethod
    def get_device_usage() -> Dict[str, Any]:
        """获取设备使用情况"""
        usage = {}
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                allocated = torch.cuda.memory_allocated(i)
                cached = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                
                usage[f"cuda:{i}"] = {
                    "allocated_mb": allocated / 1024**2,
                    "cached_mb": cached / 1024**2,
                    "total_mb": total / 1024**2,
                    "free_mb": (total - cached) / 1024**2,
                    "utilization_percent": (cached / total) * 100
                }
        
        return usage
    
    @staticmethod
    def validate_device(device: str) -> bool:
        """验证设备是否有效"""
        if device == "cpu":
            return True
        
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
                return torch.cuda.is_available() and gpu_id < torch.cuda.device_count()
            except (ValueError, IndexError):
                return False
        
        return False
    
    @staticmethod
    def get_available_gpus() -> List[str]:
        """获取可用的GPU设备列表"""
        if not torch.cuda.is_available():
            return []
        
        gpu_count = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(gpu_count)]
    
    @staticmethod
    def select_best_gpu_for_model(model_id: str) -> str:
        """为指定模型选择最佳的GPU设备"""
        # 获取模型可用的GPU列表
        gpu_config = Config.get_model_gpu_config()
        available_gpus = gpu_config.get(model_id, [])
        
        if not torch.cuda.is_available():
            logger.info(f"CUDA不可用，模型 {model_id} 将使用CPU")
            return "cpu"
        
        # 过滤出实际可用的GPU
        valid_gpus = []
        for gpu in available_gpus:
            if DeviceManager.validate_device(gpu):
                valid_gpus.append(gpu)
        
        if not valid_gpus:
            logger.warning(f"模型 {model_id} 配置的GPU都不可用，使用CPU")
            return "cpu"
        
        # 如果只有一个GPU，直接返回
        if len(valid_gpus) == 1:
            logger.info(f"模型 {model_id} 使用唯一可用GPU: {valid_gpus[0]}")
            return valid_gpus[0]
        
        # 选择负载最低的GPU
        best_gpu = valid_gpus[0]
        min_utilization = float('inf')
        
        for gpu in valid_gpus:
            try:
                gpu_id = int(gpu.split(":")[1])
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                utilization = allocated / total
                
                logger.debug(f"GPU {gpu} 使用率: {utilization:.2%}")
                
                if utilization < min_utilization:
                    min_utilization = utilization
                    best_gpu = gpu
                    
            except Exception as e:
                logger.warning(f"获取GPU {gpu} 使用率失败: {e}")
                continue
        
        logger.info(f"模型 {model_id} 选择GPU: {best_gpu} (使用率: {min_utilization:.2%})")
        return best_gpu
    
    @staticmethod
    def get_gpu_load_info() -> Dict[str, Dict[str, Any]]:
        """获取所有GPU的负载信息"""
        load_info = {}
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_id = f"cuda:{i}"
                try:
                    allocated = torch.cuda.memory_allocated(i)
                    cached = torch.cuda.memory_reserved(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    utilization = cached / total
                    
                    load_info[gpu_id] = {
                        "allocated_mb": allocated / 1024**2,
                        "cached_mb": cached / 1024**2,
                        "total_mb": total / 1024**2,
                        "free_mb": (total - cached) / 1024**2,
                        "utilization_percent": utilization * 100,
                        "available": True
                    }
                except Exception as e:
                    logger.warning(f"获取GPU {gpu_id} 负载信息失败: {e}")
                    load_info[gpu_id] = {
                        "available": False,
                        "error": str(e)
                    }
        
        return load_info 