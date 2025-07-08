import logging
from typing import Dict, Any, Optional, List
from fastapi import HTTPException
from config import Config

logger = logging.getLogger(__name__)

class ValidationUtils:
    """参数验证工具类 - 集中处理所有验证逻辑"""
    
    # 支持的生成模式
    SUPPORTED_MODES = ["text2img", "img2img", "fill", "controlnet", "redux"]
    
    # 支持的ControlNet类型
    SUPPORTED_CONTROLNET_TYPES = ["depth", "canny", "openpose"]
    
    @classmethod
    def validate_generation_request(
        cls,
        model_id: str,
        prompt: str,
        height: int,
        width: int,
        mode: str,
        controlnet_type: str = "depth",
        input_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        control_image: Optional[str] = None,
        loras: Optional[List[Dict[str, Any]]] = None,
        supported_models: Optional[List[str]] = None
    ) -> None:
        """验证生成请求的所有参数"""
        
        # 验证模型支持
        if supported_models and model_id not in supported_models:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型: {model_id}，支持的模型: {supported_models}"
            )
        
        # 验证基础参数
        cls._validate_basic_params(prompt, height, width, mode)
        
        # 验证模式特定参数
        cls._validate_mode_specific_params(mode, input_image, mask_image, control_image, controlnet_type)
        
        # 验证LoRA参数
        if loras:
            cls._validate_loras(loras)
    
    @classmethod
    def _validate_basic_params(cls, prompt: str, height: int, width: int, mode: str) -> None:
        """验证基础参数"""
        # 验证提示词
        if not Config.validate_prompt(prompt):
            raise HTTPException(status_code=400, detail="提示词长度超出限制")
        
        # 验证图片尺寸
        if not Config.validate_image_size(height, width):
            raise HTTPException(status_code=400, detail="图片尺寸超出限制")
        
        # 验证生成模式
        if mode not in cls.SUPPORTED_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的生成模式: {mode}，支持的模式: {cls.SUPPORTED_MODES}"
            )
    
    @classmethod
    def _validate_mode_specific_params(
        cls,
        mode: str,
        input_image: Optional[str],
        mask_image: Optional[str],
        control_image: Optional[str],
        controlnet_type: str
    ) -> None:
        """验证模式特定参数"""
        if mode == "img2img" and not input_image:
            raise HTTPException(status_code=400, detail="img2img模式需要提供input_image")
        elif mode == "fill":
            if not input_image and not mask_image:
                raise HTTPException(status_code=400, detail="fill模式需要提供input_image和mask_image")
            elif not input_image:
                raise HTTPException(status_code=400, detail="fill模式需要提供input_image")
            elif not mask_image:
                raise HTTPException(status_code=400, detail="fill模式需要提供mask_image")
        elif mode == "controlnet":
            if not control_image:
                raise HTTPException(status_code=400, detail="controlnet模式需要提供control_image")
            if controlnet_type.lower() not in cls.SUPPORTED_CONTROLNET_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的controlnet类型: {controlnet_type}，支持的类型: {cls.SUPPORTED_CONTROLNET_TYPES}"
                )
        elif mode == "redux" and not input_image:
            raise HTTPException(status_code=400, detail="redux模式需要提供input_image")
    
    @classmethod
    def _validate_loras(cls, loras: List[Dict[str, Any]]) -> None:
        """验证LoRA参数"""
        for lora in loras:
            if not isinstance(lora, dict):
                raise HTTPException(status_code=400, detail="每个LoRA必须是字典格式")
            if 'name' not in lora:
                raise HTTPException(status_code=400, detail="每个LoRA必须包含name字段")
            if 'weight' not in lora:
                raise HTTPException(status_code=400, detail="每个LoRA必须包含weight字段")
            
            # 验证LoRA是否存在
            lora_path = Config.get_lora_path(lora['name'])
            if not lora_path:
                raise HTTPException(
                    status_code=400,
                    detail=f"LoRA '{lora['name']}' 不存在，请检查 /loras 接口获取可用LoRA列表"
                )
            
            # 验证权重范围
            weight = lora.get('weight', 1.0)
            if not isinstance(weight, (int, float)) or weight < 0 or weight > 2:
                raise HTTPException(status_code=400, detail="LoRA权重必须在0-2之间")
    
    @classmethod
    def validate_controlnet_type(cls, controlnet_type: str) -> bool:
        """验证ControlNet类型"""
        return controlnet_type.lower() in cls.SUPPORTED_CONTROLNET_TYPES
    
    @classmethod
    def get_supported_controlnet_types(cls) -> List[str]:
        """获取支持的ControlNet类型"""
        return cls.SUPPORTED_CONTROLNET_TYPES.copy()
    
    @classmethod
    def get_supported_modes(cls) -> List[str]:
        """获取支持的生成模式"""
        return cls.SUPPORTED_MODES.copy() 