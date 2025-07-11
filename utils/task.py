from typing import Dict, Any, Optional, List

class TaskUtils:
    """任务参数构建工具类 - 统一处理任务参数构建逻辑"""
    
    @classmethod
    def build_task_params(
        cls,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        cfg: float = 3.5,
        seed: int = 42,
        mode: str = "text2img",
        strength: float = 0.8,
        input_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        control_image: Optional[str] = None,
        controlnet_type: str = "depth",
        controlnet_conditioning_scale: Optional[float] = None,
        control_guidance_start: Optional[float] = None,
        control_guidance_end: Optional[float] = None,
        loras: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """构建任务参数"""
        return {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "cfg": cfg,
            "seed": seed,
            "mode": mode,
            "strength": strength,
            "input_image": input_image,
            "mask_image": mask_image,
            "control_image": control_image,
            "controlnet_type": controlnet_type.lower(),
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "loras": loras or []
        }
    
    @classmethod
    def build_task_data(
        cls,
        task_id: str,
        prompt: str,
        params: Dict[str, Any],
        model_id: str = "flux1-dev"
    ) -> Dict[str, Any]:
        """构建任务数据（用于GPU进程）"""
        return {
            "task_id": task_id,
            "prompt": prompt,
            "model_id": model_id,  # 添加模型ID
            "height": params.get('height', 1024),
            "width": params.get('width', 1024),
            "cfg": params.get('cfg', 3.5),
            "num_inference_steps": params.get('num_inference_steps', 50),
            "seed": params.get('seed', 42),
            "mode": params.get('mode', 'text2img'),
            "strength": params.get('strength', 0.8),
            "input_image": params.get('input_image'),
            "mask_image": params.get('mask_image'),
            "control_image": params.get('control_image'),
            "controlnet_type": params.get('controlnet_type', 'depth'),
            "loras": params.get('loras', [])
        } 