from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image
import logging
import torch
from device_manager import DeviceManager

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """模型基类，定义所有模型必须实现的接口"""
    
    def __init__(self, model_id: str, model_name: str, description: str = "", gpu_device: Optional[str] = None):
        self.model_id = model_id
        self.model_name = model_name
        self.description = description
        self.is_loaded = False
        self.pipe = None
        
        # 设置GPU设备
        if gpu_device is not None:
            if DeviceManager.validate_device(gpu_device):
                self.gpu_device = gpu_device
            else:
                logger.warning(f"无效的GPU设备 {gpu_device}，使用默认设备")
                self.gpu_device = self._get_default_device()
        else:
            self.gpu_device = self._get_default_device()
    
    def _get_default_device(self) -> str:
        """获取默认设备"""
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    
    def _select_best_gpu(self) -> str:
        """为当前模型选择最佳GPU"""
        return DeviceManager.select_best_gpu_for_model(self.model_id)
    
    @abstractmethod
    def load(self) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成图片"""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        pass
    
    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """验证参数"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "id": self.model_id,
            "name": self.model_name,
            "description": self.description,
            "is_loaded": self.is_loaded,
            "supported_features": self.get_supported_features(),
            "device": self.gpu_device,
            "available_gpus": DeviceManager.get_gpu_load_info(),
            "gpu_memory": self._get_gpu_memory_info() if self.gpu_device != "cpu" else None
        }
    
    def _get_gpu_memory_info(self) -> Optional[Dict[str, Any]]:
        """获取GPU内存信息"""
        if self.gpu_device == "cpu":
            return None
        
        try:
            gpu_id = int(self.gpu_device.split(":")[1])
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            cached_memory = torch.cuda.memory_reserved(gpu_id)
            
            return {
                "total_mb": total_memory / 1024**2,
                "allocated_mb": allocated_memory / 1024**2,
                "cached_mb": cached_memory / 1024**2,
                "free_mb": (total_memory - cached_memory) / 1024**2
            }
        except Exception as e:
            logger.warning(f"获取GPU内存信息失败: {e}")
            return None
    
    @abstractmethod
    def get_supported_features(self) -> list:
        """获取支持的功能列表"""
        pass
    
    def unload(self):
        """卸载模型"""
        if self.pipe:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        logger.info(f"模型 {self.model_id} 已卸载")
    
    def validate_prompt(self, prompt: str) -> bool:
        """验证提示词"""
        return len(prompt) > 0 and len(prompt) <= 1000
    
    def validate_image_size(self, height: int, width: int) -> bool:
        """验证图片尺寸"""
        return (0 < height <= 2048 and 0 < width <= 2048)
    
    def get_device(self) -> str:
        """获取设备类型"""
        return self.gpu_device
    
    def get_optimization_kwargs(self) -> Dict[str, Any]:
        """获取优化参数 - 默认使用float16"""
        kwargs = {}
        kwargs["torch_dtype"] = torch.float16
        return kwargs
    
    def set_gpu_device(self, device: str) -> bool:
        """设置GPU设备"""
        if not DeviceManager.validate_device(device):
            logger.error(f"无效的GPU设备: {device}")
            return False
        
        if self.is_loaded:
            logger.warning(f"模型已加载，无法更改设备。请先卸载模型")
            return False
        
        self.gpu_device = device
        logger.info(f"模型 {self.model_id} 设备已设置为: {device}")
        return True
    
    def get_gpu_device(self) -> str:
        """获取当前GPU设备"""
        return self.gpu_device
    
    def _ensure_model_on_device(self, target_device: str) -> bool:
        """确保模型在指定设备上"""
        if not self.is_loaded:
            logger.error("模型未加载")
            return False
        
        if self.gpu_device != target_device:
            logger.info(f"将模型从 {self.gpu_device} 移动到 {target_device}")
            try:
                if target_device != "cpu":
                    self.pipe = self.pipe.to(target_device)
                self.gpu_device = target_device
                return True
            except Exception as e:
                logger.error(f"移动模型到设备 {target_device} 失败: {e}")
                return False
        
        return True 