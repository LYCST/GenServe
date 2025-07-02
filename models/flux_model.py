import base64
import io
import time
import torch
from typing import Dict, Any, Optional
from PIL import Image
from diffusers import FluxPipeline
from .base import BaseModel
from config import Config
import logging
import os

logger = logging.getLogger(__name__)

class FluxModel(BaseModel):
    """Flux模型实现"""
    
    def __init__(self, gpu_device: Optional[str] = None):
        super().__init__(
            model_id="flux1-dev",
            model_name="FLUX.1-dev",
            description="Black Forest Labs FLUX.1-dev model for high-quality image generation",
            gpu_device=gpu_device
        )
        self.model_path = Config.get_model_path("flux1-dev")
    
    def load(self) -> bool:
        """加载Flux模型"""
        try:
            # 选择最佳GPU进行加载
            best_gpu = self._select_best_gpu()
            logger.info(f"正在加载模型: {self.model_name} 到设备: {best_gpu}")
            logger.info(f"模型路径: {self.model_path}")
            
            # 检查模型路径是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                return False
            
            # 尝试多种加载方式
            load_success = False
            
            # 方法1：使用FluxPipeline with CPU offload
            try:
                logger.info("尝试使用FluxPipeline加载模型...")
                self.pipe = FluxPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                # 启用CPU卸载以节省GPU内存 - 不指定gpu_id让其自动选择
                self.pipe.enable_model_cpu_offload()
                load_success = True
                logger.info("FluxPipeline with CPU offload加载成功")
                
            except Exception as e:
                logger.warning(f"FluxPipeline加载失败: {e}")
                
                # 方法2：使用DiffusionPipeline with CPU offload
                try:
                    logger.info("尝试使用DiffusionPipeline加载模型...")
                    from diffusers import DiffusionPipeline
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    # 启用CPU卸载 - 不指定gpu_id
                    self.pipe.enable_model_cpu_offload()
                    load_success = True
                    logger.info("DiffusionPipeline with CPU offload加载成功")
                    
                except Exception as e2:
                    logger.warning(f"DiffusionPipeline加载失败: {e2}")
                    
                    # 方法3：使用更宽松的参数
                    try:
                        logger.info("尝试使用宽松参数加载模型...")
                        from diffusers import DiffusionPipeline
                        self.pipe = DiffusionPipeline.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float16,  # 改用float16
                            local_files_only=True,
                            trust_remote_code=True
                        )
                        # 启用CPU卸载 - 不指定gpu_id
                        self.pipe.enable_model_cpu_offload()
                        load_success = True
                        logger.info("宽松参数with CPU offload加载成功")
                        
                    except Exception as e3:
                        logger.error(f"所有加载方法都失败了: {e3}")
                        return False
            
            if not load_success:
                return False
            
            # 不手动移动到GPU，让CPU offload自动管理
            self.gpu_device = best_gpu
            logger.info(f"模型 {self.model_name} 已启用CPU offload，目标GPU: {best_gpu.upper()}")
            
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
        
        # 验证提示词
        if not self.validate_prompt(prompt):
            raise ValueError("提示词验证失败")
        
        # 获取参数，使用默认值填充缺失的参数
        params = self.get_default_params()
        params.update(kwargs)
        
        # 验证参数
        if not self.validate_params(**params):
            raise ValueError("参数验证失败")
        
        # 使用模型加载时选择的GPU设备
        device = self.gpu_device
        logger.debug(f"🎯 使用设备进行生成: {device}")
        
        start_time = time.time()
        
        try:
            # 设置随机种子 - 使用CPU generator如示例所示
            generator = torch.Generator("cpu").manual_seed(params['seed'])
            
            logger.info(f"开始生成图片，提示词: {prompt}，设备: {device}")
            
            with torch.no_grad():
                # 使用与工作示例相同的参数
                result = self.pipe(
                    prompt=prompt,
                    height=params['height'],
                    width=params['width'],
                    guidance_scale=params['cfg'],
                    num_inference_steps=params['num_inference_steps'],
                    max_sequence_length=512,  # 添加这个关键参数
                    generator=generator
                )
            
            image = result.images[0]
            
            # 转换为base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            elapsed_time = time.time() - start_time
            
            # 清理GPU内存
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            
            # 如果指定了保存路径，保存图片
            save_to_disk = False
            if params.get('save_disk_path'):
                try:
                    image.save(params['save_disk_path'])
                    save_to_disk = True
                    logger.info(f"图片已保存到: {params['save_disk_path']}")
                except Exception as e:
                    logger.warning(f"保存图片失败: {e}")
            
            logger.info(f"图片生成完成，耗时: {elapsed_time:.2f}秒，设备: {device}")
            
            return {
                "success": True,
                "image": image,
                "base64": img_base64,
                "elapsed_time": elapsed_time,
                "save_to_disk": save_to_disk,
                "params": params,
                "device": device
            }
            
        except Exception as e:
            logger.error(f"图片生成失败: {e}")
            # 清理GPU内存
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "device": device
            }
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "num_inference_steps": 50,  # Flux推荐使用50步
            "seed": 42,
            "cfg": 3.5,  # Flux推荐使用3.5
            "height": 1024,  # 改为1024x1024如示例
            "width": 1024,   # 改为1024x1024如示例
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
        
        # 验证图片尺寸
        if not self.validate_image_size(kwargs['height'], kwargs['width']):
            return False
        
        return True
    
    def get_supported_features(self) -> list:
        """获取支持的功能列表"""
        return ["text-to-image"]
    
    def get_optimization_kwargs(self) -> Dict[str, Any]:
        """获取优化参数 - Flux使用bfloat16"""
        kwargs = {}
        kwargs["torch_dtype"] = torch.bfloat16
        return kwargs 