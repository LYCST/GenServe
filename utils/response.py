from typing import Dict, Any, Optional
from pydantic import BaseModel

class GenerateResponse(BaseModel):
    """生成响应模型"""
    success: bool
    task_id: str
    image_base64: Optional[str] = None
    error: Optional[str] = None
    elapsed_time: Optional[float] = None
    gpu_id: Optional[str] = None
    model_id: Optional[str] = None
    mode: Optional[str] = None
    controlnet_type: Optional[str] = None

class ResponseUtils:
    """响应构建工具类 - 统一处理响应构建逻辑"""
    
    @classmethod
    def build_generation_response(
        cls,
        result: Dict[str, Any],
        mode: str,
        controlnet_type: Optional[str] = None
    ) -> GenerateResponse:
        """构建生成响应"""
        return GenerateResponse(
            success=result.get("success", False),
            task_id=result.get("task_id", ""),
            image_base64=result.get("image_base64"),
            error=result.get("error"),
            elapsed_time=result.get("elapsed_time"),
            gpu_id=result.get("gpu_id"),
            model_id=result.get("model_id"),
            mode=mode,
            controlnet_type=controlnet_type
        )
    
    @classmethod
    def build_error_response(
        cls,
        error: str,
        task_id: str = "",
        gpu_id: Optional[str] = None,
        model_id: Optional[str] = None,
        mode: Optional[str] = None,
        controlnet_type: Optional[str] = None
    ) -> GenerateResponse:
        """构建错误响应"""
        return GenerateResponse(
            success=False,
            task_id=task_id,
            error=error,
            gpu_id=gpu_id,
            model_id=model_id,
            mode=mode,
            controlnet_type=controlnet_type
        ) 