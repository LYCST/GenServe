import base64
import io
import time
import torch
from typing import Dict, Any
from PIL import Image
from diffusers import DiffusionPipeline
from .base import BaseModel
import logging

logger = logging.getLogger(__name__)

class SDXLModel(BaseModel):
    """SDXL模型实现示例"""
    
    def __init__(self):
        super().__init__(
            model_id="sdxl-base",
            model_name="Stable Diffusion XL Base",
            description="Stability AI Stable Diffusion XL base model"
        )
        self.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    
    def load(self) -> bool:
        """加载SDXL模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # 如果有GPU则使用GPU
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                logger.info(f"模型 {self.model_name} 已加载到GPU")
            else:
                logger.info(f"模型 {self.model_name} 已加载到CPU")
            
            self.is_loaded = True
            logger.info(f"模型 {self.model_name} 加载完成")
            return True
            
        except Exception as e:
            logger.error(f"模型 {self.model_name} 加载失败: {e}")
            self.is_loaded = False
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成图片"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        # 获取参数，使用默认值填充缺失的参数
        params = self.get_default_params()
        params.update(kwargs)
        
        # 验证参数
        if not self.validate_params(**params):
            raise ValueError("参数验证失败")
        
        start_time = time.time()
        
        try:
            # 设置随机种子
            if torch.cuda.is_available():
                torch.cuda.manual_seed(params['seed'])
            else:
                torch.manual_seed(params['seed'])
            
            logger.info(f"开始生成图片，提示词: {prompt}")
            
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    num_inference_steps=params['num_inference_steps'],
                    guidance_scale=params['cfg'],
                    height=params['height'],
                    width=params['width']
                )
            
            image = result.images[0]
            
            # 转换为base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            elapsed_time = time.time() - start_time
            
            # 如果指定了保存路径，保存图片
            save_to_disk = False
            if params.get('save_disk_path'):
                try:
                    image.save(params['save_disk_path'])
                    save_to_disk = True
                    logger.info(f"图片已保存到: {params['save_disk_path']}")
                except Exception as e:
                    logger.warning(f"保存图片失败: {e}")
            
            logger.info(f"图片生成完成，耗时: {elapsed_time:.2f}秒")
            
            return {
                "success": True,
                "image": image,
                "base64": img_base64,
                "elapsed_time": elapsed_time,
                "save_to_disk": save_to_disk,
                "params": params
            }
            
        except Exception as e:
            logger.error(f"图片生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "num_inference_steps": 25,
            "seed": 42,
            "cfg": 7.5,
            "height": 1024,
            "width": 1024,
            "save_disk_path": None
        }
    
    def validate_params(self, **kwargs) -> bool:
        """验证参数"""
        # 检查必需参数
        required_params = ['num_inference_steps', 'seed', 'cfg', 'height', 'width']
        for param in required_params:
            if param not in kwargs:
                return False
        
        # 验证参数范围
        if kwargs['num_inference_steps'] < 1 or kwargs['num_inference_steps'] > 100:
            return False
        
        if kwargs['cfg'] < 0.1 or kwargs['cfg'] > 20:
            return False
        
        # SDXL支持的最大分辨率是1024x1024
        if kwargs['height'] < 64 or kwargs['height'] > 1024:
            return False
        
        if kwargs['width'] < 64 or kwargs['width'] > 1024:
            return False
        
        return True
    
    def get_supported_features(self) -> list:
        """获取支持的功能列表"""
        return ["text-to-image"] 