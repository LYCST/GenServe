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
        
        # 不在初始化时设置环境变量，而是在加载和生成时动态设置
        logger.info(f"创建FluxModel实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id}")
    
    def __del__(self):
        """析构函数"""
        # 不需要在这里恢复环境变量，因为我们在每次操作后都会恢复
        pass
    
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
            
            # 保存原始环境变量
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            
            try:
                # 对于GPU隔离的情况，临时设置环境变量
                if self.physical_gpu_id != "cpu":
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
                
            finally:
                # 恢复原始环境变量
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    
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
        """内部生成方法 - 线程级GPU隔离版本"""
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
        
        # 保存当前CUDA设备状态
        original_device = None
        if torch.cuda.is_available():
            original_device = torch.cuda.current_device()
        
        try:
            # 设置随机种子
            generator = torch.Generator("cpu").manual_seed(params['seed'])
            
            # 线程级GPU隔离 - 使用torch.cuda.set_device()
            if self.physical_gpu_id != "cpu":
                target_gpu_id = int(self.physical_gpu_id)
                torch.cuda.set_device(target_gpu_id)
                device = f"cuda:{target_gpu_id}"
                logger.info(f"线程 {threading.current_thread().name} 设置GPU设备: {device}")
            else:
                device = "cpu"

            logger.info(f"开始生成图片，提示词: {prompt}，设备: {device}, 物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")

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
            
            # 清理当前GPU内存
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
            
            logger.info(f"图片生成完成，耗时: {elapsed_time:.2f}秒，设备: {device}, 物理GPU: {self.physical_gpu_id} (实例: {self._instance_id})")
            
            return {
                "success": True,
                "image": image,
                "base64": img_base64,
                "elapsed_time": elapsed_time,
                "save_to_disk": save_to_disk,
                "params": params,
                "device": device,
                "physical_gpu_id": self.physical_gpu_id,
                "instance_id": self._instance_id,
                "thread_name": threading.current_thread().name
            }
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU内存不足: {e} (物理GPU: {self.physical_gpu_id})")
            
            # 尝试清理内存
            self._emergency_cleanup()
            
            return {
                "success": False,
                "error": f"GPU {self.physical_gpu_id} 内存不足: {str(e)}",
                "elapsed_time": time.time() - start_time,
                "device": device if 'device' in locals() else "unknown",
                "physical_gpu_id": self.physical_gpu_id,
                "instance_id": self._instance_id,
                "thread_name": threading.current_thread().name
            }
            
        except Exception as e:
            logger.error(f"图片生成失败: {e} (实例: {self._instance_id})")
            
            self._emergency_cleanup()

            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "device": device if 'device' in locals() else "unknown",
                "physical_gpu_id": self.physical_gpu_id,
                "instance_id": self._instance_id,
                "thread_name": threading.current_thread().name
            }
        finally:
            # 恢复原始CUDA设备
            if original_device is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(original_device)
                    logger.debug(f"线程 {threading.current_thread().name} 恢复GPU设备: cuda:{original_device}")
                except Exception as e:
                    logger.warning(f"恢复GPU设备时出错: {e}")
    
    def _emergency_cleanup(self):
        """紧急清理 - 在生成失败时使用"""
        logger.warning(f"执行紧急清理 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        
        try:
            if self.physical_gpu_id != "cpu":
                # 清理GPU内存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 安全清理pipeline及其组件
                self._safe_pipeline_cleanup()
                
                # 强制垃圾回收
                gc.collect()
                
                # 再次清理
                torch.cuda.empty_cache()
                
            logger.info("紧急清理完成")
            
        except Exception as e:
            logger.error(f"紧急清理时出错: {e}")


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