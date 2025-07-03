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
import threading
import gc

logger = logging.getLogger(__name__)

class FluxModel(BaseModel):
    """Flux模型实现 - 支持GPU隔离"""
    
    def __init__(self, gpu_device: Optional[str] = None, physical_gpu_id: Optional[str] = None):
        super().__init__(
            model_id="flux1-dev",
            model_name="FLUX.1-dev",
            description="Black Forest Labs FLUX.1-dev model for high-quality image generation",
            gpu_device=gpu_device
        )
        self.model_path = Config.get_model_path("flux1-dev")
        self.physical_gpu_id = physical_gpu_id if physical_gpu_id else self._extract_gpu_id(gpu_device)
        self._generation_lock = threading.Lock()
        self._instance_id = f"flux_{gpu_device}_{id(self)}"
        
        # 设置环境变量，确保整个实例生命周期使用同一个GPU
        if self.physical_gpu_id != "cpu":
            self._original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.physical_gpu_id
            logger.info(f"创建FluxModel实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id}")
    
    def __del__(self):
        """析构函数，恢复环境变量"""
        if hasattr(self, '_original_cuda_visible') and self.physical_gpu_id != "cpu":
            if self._original_cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_visible
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    def _extract_gpu_id(self, device: str) -> str:
        """从设备名称提取GPU ID"""
        if device == "cpu":
            return "cpu"
        elif device and device.startswith("cuda:"):
            return device.split(":")[1]
        else:
            return "0"
    
    def load(self) -> bool:
        """加载Flux模型"""
        try:
            # 检查模型路径是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                return False
            
            logger.info(f"正在加载模型: {self.model_name}，物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")
            
            # 对于GPU隔离的情况，始终使用 cuda:0（因为只有一个可见GPU）
            if self.physical_gpu_id != "cpu":
                # 确保CUDA_VISIBLE_DEVICES已设置
                current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if current_visible != self.physical_gpu_id:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.physical_gpu_id
                    logger.info(f"设置 CUDA_VISIBLE_DEVICES={self.physical_gpu_id}")
                
                # 验证GPU是否可用
                if not torch.cuda.is_available():
                    logger.error(f"GPU {self.physical_gpu_id} 不可用")
                    return False
                
                # 加载模型
                self.pipe = FluxPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                
                # 使用CPU offload到cuda:0（在隔离环境中这是唯一可见的GPU）
                self.pipe.enable_model_cpu_offload(device="cuda:0")
                logger.info(f"FluxPipeline加载成功，使用物理GPU {self.physical_gpu_id} (逻辑设备: cuda:0)")
                
            else:
                # CPU模式
                self.pipe = FluxPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                self.pipe = self.pipe.to("cpu")
                logger.info(f"FluxPipeline加载到CPU成功")
            
            self.is_loaded = True
            logger.info(f"模型 {self.model_name} 加载完成，物理GPU: {self.physical_gpu_id}")
            return True
            
        except Exception as e:
            logger.error(f"模型 {self.model_name} 加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成图片 - 线程安全版本"""
        with self._generation_lock:
            return self._generate_internal(prompt, **kwargs)
    
    def _generate_internal(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """内部生成方法"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        # 验证提示词
        if not self.validate_prompt(prompt):
            raise ValueError("提示词验证失败")
        
        # 获取参数
        params = self.get_default_params()
        params.update(kwargs)
        
        # 验证参数
        if not self.validate_params(**params):
            raise ValueError("参数验证失败")
        
        start_time = time.time()
        
        try:
            # 确保环境变量正确
            if self.physical_gpu_id != "cpu":
                current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if current_visible != self.physical_gpu_id:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.physical_gpu_id
                    logger.warning(f"修正CUDA_VISIBLE_DEVICES为: {self.physical_gpu_id}")
                
                # 在隔离环境中使用cuda:0
                device = "cuda:0"
            else:
                device = "cpu"
            
            logger.info(f"开始生成图片，提示词: {prompt}，逻辑设备: {device}, 物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")
            
            # 设置随机种子
            generator = torch.Generator("cpu").manual_seed(params['seed'])
            
            with torch.no_grad():
                # 生成图片
                result = self.pipe(
                    prompt=prompt,
                    height=params['height'],
                    width=params['width'],
                    guidance_scale=params['cfg'],
                    num_inference_steps=params['num_inference_steps'],
                    max_sequence_length=512,
                    generator=generator
                )
            
            image = result.images[0]
            
            # 转换为base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            elapsed_time = time.time() - start_time
            
            # 清理GPU内存
            if self.physical_gpu_id != "cpu":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 保存图片（如果需要）
            save_to_disk = False
            if params.get('save_disk_path'):
                try:
                    image.save(params['save_disk_path'])
                    save_to_disk = True
                    logger.info(f"图片已保存到: {params['save_disk_path']}")
                except Exception as e:
                    logger.warning(f"保存图片失败: {e}")
            
            logger.info(f"图片生成完成，耗时: {elapsed_time:.2f}秒，物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")
            
            return {
                "success": True,
                "image": image,
                "base64": img_base64,
                "elapsed_time": elapsed_time,
                "save_to_disk": save_to_disk,
                "params": params,
                "device": self.gpu_device,
                "physical_gpu_id": self.physical_gpu_id,
                "instance_id": self._instance_id
            }
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU内存不足: {e} (物理GPU: {self.physical_gpu_id})")
            
            # 尝试清理内存
            self._emergency_cleanup()
            
            return {
                "success": False,
                "error": f"GPU {self.physical_gpu_id} 内存不足: {str(e)}",
                "elapsed_time": time.time() - start_time,
                "device": self.gpu_device,
                "physical_gpu_id": self.physical_gpu_id,
                "instance_id": self._instance_id
            }
            
        except Exception as e:
            logger.error(f"图片生成失败: {e} (实例: {self._instance_id})")
            
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "device": self.gpu_device,
                "physical_gpu_id": self.physical_gpu_id,
                "instance_id": self._instance_id
            }
    
    def _emergency_cleanup(self):
        """紧急清理 - 在生成失败时使用"""
        logger.warning(f"执行紧急清理 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        
        try:
            if self.physical_gpu_id != "cpu":
                # 清理GPU内存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # 强制垃圾回收
                gc.collect()
                
                # 再次清理
                torch.cuda.empty_cache()
                
            logger.info("紧急清理完成")
            
        except Exception as e:
            logger.error(f"紧急清理时出错: {e}")
    
    def unload(self):
        """卸载模型"""
        logger.info(f"开始卸载模型 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        
        # 标记为未加载
        self.is_loaded = False
        
        # 清理pipeline
        if hasattr(self, 'pipe') and self.pipe is not None:
            try:
                # 禁用CPU offload
                if hasattr(self.pipe, 'disable_model_cpu_offload'):
                    self.pipe.disable_model_cpu_offload()
                
                # 删除pipeline
                del self.pipe
                self.pipe = None
                
                # 清理GPU内存
                if self.physical_gpu_id != "cpu":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
            except Exception as e:
                logger.error(f"卸载模型时出错: {e}")
        
        logger.info(f"模型 {self.model_id} 已卸载 (物理GPU: {self.physical_gpu_id})")
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "num_inference_steps": 50,
            "seed": 42,
            "cfg": 3.5,
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