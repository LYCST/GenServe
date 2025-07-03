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
    
    def __init__(self, gpu_device: Optional[str] = None, isolated_gpu_id: Optional[str] = None):
        super().__init__(
            model_id="flux1-dev",
            model_name="FLUX.1-dev",
            description="Black Forest Labs FLUX.1-dev model for high-quality image generation",
            gpu_device=gpu_device
        )
        self.model_path = Config.get_model_path("flux1-dev")
        # 添加线程锁保护
        self._generation_lock = threading.Lock()
        # 为每个实例创建唯一标识
        self._instance_id = f"flux_{gpu_device}_{id(self)}"
        # 记录GPU隔离信息
        self.isolated_gpu_id = isolated_gpu_id  # 在隔离环境中的GPU ID (通常是"0")
        self.physical_gpu_id = self._get_gpu_id_from_device(gpu_device or "cpu")  # 物理GPU ID
        
        logger.info(f"创建FluxModel实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id}, 隔离GPU: {self.isolated_gpu_id}")

    def _get_gpu_id_from_device(self, device: str) -> str:
        """从设备名称提取GPU ID"""
        if device == "cpu":
            return "cpu"
        elif device.startswith("cuda:"):
            return device.split(":")[1]
        else:
            return "0"  # 默认使用GPU 0
    
    def _update_device_for_task(self, target_device: str):
        """更新设备配置以准备任务执行"""
        # 这个方法由并发管理器调用，确保模型使用正确的设备
        if target_device.startswith("cuda:"):
            physical_gpu_id = target_device.split(":")[1]
            self.physical_gpu_id = physical_gpu_id
            logger.debug(f"更新任务设备配置: 物理GPU {physical_gpu_id}, 逻辑GPU cuda:0")
    
    def _deep_gpu_cleanup(self, target_device: Optional[str] = None):
        """深度GPU显存清理 - GPU隔离版本"""
        try:
            # 在GPU隔离环境中，总是清理设备0
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                with torch.cuda.device(0):
                    # 强制同步
                    torch.cuda.synchronize()
                    # 清理缓存
                    torch.cuda.empty_cache()
                    # 重置统计
                    torch.cuda.reset_peak_memory_stats()
                    # 再次同步
                    torch.cuda.synchronize()
                logger.debug(f"已清理GPU显存 (物理GPU: {self.physical_gpu_id}, 逻辑GPU: 0)")
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            logger.warning(f"深度GPU清理时出错: {e}")
    
    def _safe_pipeline_cleanup(self):
        """安全清理pipeline及其组件"""
        try:
            if hasattr(self, 'pipe') and self.pipe is not None:
                logger.info(f"开始清理pipeline组件 (实例: {self._instance_id})")
                
                # 获取pipeline中的所有组件
                components_to_cleanup = []
                
                # 检查并收集需要清理的组件
                for attr_name in ['transformer', 'vae', 'text_encoder', 'text_encoder_2', 
                                'scheduler', 'tokenizer', 'tokenizer_2']:
                    if hasattr(self.pipe, attr_name):
                        component = getattr(self.pipe, attr_name)
                        if component is not None:
                            components_to_cleanup.append((attr_name, component))
                
                # 逐个清理组件
                for name, component in components_to_cleanup:
                    try:
                        # 如果组件有参数，尝试移动到CPU
                        if hasattr(component, 'to') and hasattr(component, 'parameters'):
                            component.to('cpu')
                            logger.debug(f"已将 {name} 移动到CPU")
                        
                        # 如果组件有cuda()方法，说明可能在GPU上
                        if hasattr(component, 'cuda'):
                            try:
                                component.cpu()
                                logger.debug(f"已将 {name} 移动到CPU")
                            except:
                                pass
                                
                    except Exception as e:
                        logger.warning(f"清理组件 {name} 时出错: {e}")
                
                # 清理pipeline本身
                try:
                    # 禁用CPU offload
                    if hasattr(self.pipe, 'disable_model_cpu_offload'):
                        self.pipe.disable_model_cpu_offload()
                        logger.debug("已禁用模型CPU offload")
                except Exception as e:
                    logger.warning(f"禁用CPU offload时出错: {e}")
                
                # 尝试将整个pipeline移动到CPU
                try:
                    self.pipe.to('cpu')
                    logger.debug("已将pipeline移动到CPU")
                except Exception as e:
                    logger.warning(f"移动pipeline到CPU时出错: {e}")
                
                # 删除pipeline引用
                del self.pipe
                self.pipe = None
                logger.info(f"Pipeline已删除 (实例: {self._instance_id})")
                
        except Exception as e:
            logger.error(f"清理pipeline时出错: {e}")
    
    def load(self) -> bool:
        """加载Flux模型 - GPU隔离版本"""
        try:
            # 检查模型路径是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                return False
            
            logger.info(f"正在加载模型: {self.model_name} (实例: {self._instance_id})")
            logger.info(f"模型路径: {self.model_path}")
            logger.info(f"目标设备: {self.gpu_device} (物理GPU: {self.physical_gpu_id})")
            
            # 获取Flux特定配置
            flux_config = Config.get_flux_config()
            logger.info(f"Flux配置: {flux_config}")
            
            try:
                # 使用FluxPipeline
                logger.info("尝试使用FluxPipeline加载模型...")
                self.pipe = FluxPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=flux_config["torch_dtype"],
                    use_safetensors=True,
                    local_files_only=True
                )
                
                # 根据设备类型进行不同的处理
                if self.gpu_device and self.gpu_device.startswith("cuda:"):
                    # 对于GPU设备，使用CPU offload
                    # 在GPU隔离环境中，总是指定cuda:0
                    target_device = "cuda:0"
                    logger.info(f"启用模型CPU offload，目标设备: {target_device}")
                    
                    # 使用配置的CPU offload设置
                    if flux_config["use_cpu_offload"]:
                        self.pipe.enable_model_cpu_offload(device=target_device)
                        logger.info("已启用模型CPU offload")
                    else:
                        logger.info("CPU offload已禁用")
                    
                    # 验证设备设置
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        current_device = torch.cuda.current_device()
                        device_name = torch.cuda.get_device_name(0)
                        logger.info(f"GPU隔离环境验证: 当前设备={current_device}, 设备名称={device_name}")
                    
                    logger.info(f"FluxPipeline with CPU offload加载成功 (实例: {self._instance_id})")
                else:
                    # 对于CPU设备，直接使用CPU
                    self.pipe = self.pipe.to("cpu")
                    logger.info(f"FluxPipeline加载到CPU成功 (实例: {self._instance_id})")
                
                self.is_loaded = True
                logger.info(f"模型 {self.model_name} 加载完成 (实例: {self._instance_id})")
                return True
                
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"模型 {self.model_name} 加载失败: {e} (实例: {self._instance_id})")
            self.is_loaded = False
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成图片 - 线程安全版本"""
        # 使用线程锁确保同一时间只有一个生成任务
        with self._generation_lock:
            return self._generate_internal(prompt, **kwargs)
    
    def _generate_internal(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """内部生成方法 - GPU隔离版本"""
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
        
        # 使用逻辑设备（在GPU隔离环境中总是cuda:0）
        logical_device = "cuda:0" if self.gpu_device.startswith("cuda:") else "cpu"
        logger.debug(f"🎯 使用逻辑设备: {logical_device}, 物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")
        
        start_time = time.time()
        
        try:
            # 设置随机种子 - 使用CPU generator
            generator = torch.Generator("cpu").manual_seed(params['seed'])
            
            logger.info(f"开始生成图片，提示词: {prompt}，逻辑设备: {logical_device}, 物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")
            
            # 确保在正确的GPU上下文中
            if logical_device.startswith("cuda:") and torch.cuda.is_available():
                torch.cuda.set_device(0)  # 在GPU隔离环境中总是使用设备0
                logger.debug(f"设置CUDA设备为: 0 (物理GPU: {self.physical_gpu_id})")
            
            with torch.no_grad():
                # 使用CPU offload的FluxPipeline
                result = self.pipe(
                    prompt=prompt,
                    height=params['height'],
                    width=params['width'],
                    guidance_scale=params['cfg'],
                    num_inference_steps=params['num_inference_steps'],
                    max_sequence_length=512,  # 关键参数
                    generator=generator
                )
            
            image = result.images[0]
            
            # 转换为base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            elapsed_time = time.time() - start_time
            
            # 成功后的轻量级清理
            self._deep_gpu_cleanup()
            
            # 如果指定了保存路径，保存图片
            save_to_disk = False
            if params.get('save_disk_path'):
                try:
                    image.save(params['save_disk_path'])
                    save_to_disk = True
                    logger.info(f"图片已保存到: {params['save_disk_path']}")
                except Exception as e:
                    logger.warning(f"保存图片失败: {e}")
            
            logger.info(f"图片生成完成，耗时: {elapsed_time:.2f}秒，逻辑设备: {logical_device}, 物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")
            
            return {
                "success": True,
                "image": image,
                "base64": img_base64,
                "elapsed_time": elapsed_time,
                "save_to_disk": save_to_disk,
                "params": params,
                "device": self.gpu_device,  # 返回原始设备名
                "logical_device": logical_device,
                "physical_gpu": self.physical_gpu_id,
                "instance_id": self._instance_id
            }
            
        except Exception as e:
            logger.error(f"图片生成失败: {e} (实例: {self._instance_id})")
            
            # 失败时的彻底清理
            self._emergency_cleanup()
            
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "device": self.gpu_device,
                "logical_device": logical_device,
                "physical_gpu": self.physical_gpu_id,
                "instance_id": self._instance_id
            }
    
    def _emergency_cleanup(self):
        """紧急清理 - GPU隔离版本"""
        logger.warning(f"执行紧急清理 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        
        try:
            # 1. 先尝试深度GPU清理
            self._deep_gpu_cleanup()
            
            # 2. 清理pipeline组件
            self._safe_pipeline_cleanup()
            
            # 3. 强制垃圾回收
            gc.collect()
            
            # 4. 再次清理GPU
            self._deep_gpu_cleanup()
            
            # 5. 重新加载模型（如果需要）
            if not self.is_loaded:
                logger.info(f"尝试重新加载模型 (实例: {self._instance_id})")
                self.load()
            
        except Exception as e:
            logger.error(f"紧急清理时出错: {e}")
    
    def unload(self):
        """卸载模型 - 增强版本"""
        logger.info(f"开始卸载模型 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        
        # 标记为未加载
        self.is_loaded = False
        
        # 安全清理pipeline
        self._safe_pipeline_cleanup()
        
        # 深度清理GPU
        self._deep_gpu_cleanup()
        
        logger.info(f"模型 {self.model_id} 已完全卸载 (实例: {self._instance_id})")
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "num_inference_steps": 50,  # Flux推荐使用50步
            "seed": 42,
            "cfg": 3.5,  # Flux推荐使用3.5
            "height": 1024,  # 1024x1024
            "width": 1024,   # 1024x1024
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