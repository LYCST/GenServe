from typing import Dict, List, Optional, Any
import logging
from .base import BaseModel
from .flux_model import FluxModel
from .sdxl_model import SDXLModel

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器，负责管理所有模型"""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """注册默认模型"""
        self.register_model(FluxModel())
        # self.register_model(SDXLModel())  # 可选启用
    
    def register_model(self, model: BaseModel):
        """注册模型"""
        if model.model_id in self.models:
            logger.warning(f"模型 {model.model_id} 已存在，将被覆盖")
        
        self.models[model.model_id] = model
        logger.info(f"模型 {model.model_id} 已注册，设备: {model.gpu_device}")
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """获取模型"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型信息"""
        return [model.get_info() for model in self.models.values()]
    
    def load_model(self, model_id: str) -> bool:
        """加载指定模型"""
        model = self.get_model(model_id)
        if not model:
            logger.error(f"模型 {model_id} 不存在")
            return False
        
        if model.is_loaded:
            logger.info(f"模型 {model_id} 已经加载")
            return True
        
        return model.load()
    
    def load_all_models(self) -> Dict[str, bool]:
        """加载所有模型"""
        results = {}
        for model_id in self.models:
            results[model_id] = self.load_model(model_id)
        return results
    
    def unload_model(self, model_id: str):
        """卸载指定模型"""
        model = self.get_model(model_id)
        if model:
            model.unload()
    
    def unload_all_models(self):
        """卸载所有模型"""
        for model in self.models.values():
            model.unload()
    
    def generate_image(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """使用指定模型生成图片"""
        model = self.get_model(model_id)
        if not model:
            return {
                "success": False,
                "error": f"模型 {model_id} 不存在"
            }
        
        if not model.is_loaded:
            return {
                "success": False,
                "error": f"模型 {model_id} 未加载"
            }
        
        try:
            return model.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"使用模型 {model_id} 生成图片失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型配置"""
        model = self.get_model(model_id)
        if not model:
            return None
        
        return {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "description": model.description,
            "default_params": model.get_default_params(),
            "supported_features": model.get_supported_features(),
            "device": model.gpu_device
        }
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型配置"""
        configs = {}
        for model_id in self.models:
            config = self.get_model_config(model_id)
            if config:
                configs[model_id] = config
        return configs
    
    def set_model_device(self, model_id: str, device: str) -> bool:
        """设置模型设备"""
        model = self.get_model(model_id)
        if not model:
            logger.error(f"模型 {model_id} 不存在")
            return False
        
        return model.set_gpu_device(device) 