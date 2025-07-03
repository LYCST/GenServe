import base64
import io
import time
import torch
from typing import Dict, Any, Optional, Union
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxFillPipeline, FluxControlNetImg2ImgPipeline
from .base import BaseModel
from config import Config
import logging
import os
import threading
import gc
import numpy as np

logger = logging.getLogger(__name__)

class FluxModel(BaseModel):
    """Flux模型实现 - 支持GPU隔离和多模式图生图"""
    
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
        
        # 支持多种pipeline类型
        self.pipelines = {
            "text2img": None,
            "img2img": None,
            "fill": None,
            "controlnet": None
        }
        
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
    
    def _load_pipeline(self, pipeline_type: str) -> bool:
        """加载指定类型的pipeline"""
        try:
            if pipeline_type == "text2img":
                self.pipelines[pipeline_type] = FluxPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
            elif pipeline_type == "img2img":
                self.pipelines[pipeline_type] = FluxImg2ImgPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
            elif pipeline_type == "fill":
                # 使用Fill模型路径
                fill_model_path = os.path.join(os.path.dirname(self.model_path), "flux1-fill-dev")
                if os.path.exists(fill_model_path):
                    self.pipelines[pipeline_type] = FluxFillPipeline.from_pretrained(
                        fill_model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True
                    )
                else:
                    logger.warning(f"Fill模型路径不存在: {fill_model_path}")
                    return False
            elif pipeline_type == "controlnet":
                # 使用ControlNet模型路径
                controlnet_model_path = os.path.join(os.path.dirname(self.model_path), "flux1-canny-dev")
                if os.path.exists(controlnet_model_path):
                    self.pipelines[pipeline_type] = FluxControlNetImg2ImgPipeline.from_pretrained(
                        controlnet_model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True
                    )
                else:
                    logger.warning(f"ControlNet模型路径不存在: {controlnet_model_path}")
                    return False
            
            # 启用CPU offload
            if self.pipelines[pipeline_type]:
                self.pipelines[pipeline_type].enable_model_cpu_offload(device="cuda:0")
                logger.info(f"成功加载 {pipeline_type} pipeline")
                return True
                
        except Exception as e:
            logger.error(f"加载 {pipeline_type} pipeline失败: {e}")
            return False
        
        return False
    
    def _get_pipeline(self, pipeline_type: str):
        """获取或加载pipeline"""
        if self.pipelines[pipeline_type] is None:
            if not self._load_pipeline(pipeline_type):
                raise RuntimeError(f"无法加载 {pipeline_type} pipeline")
        return self.pipelines[pipeline_type]
    
    def _load_image(self, image_data: Union[str, bytes, Image.Image]) -> Image.Image:
        """加载图片数据"""
        if isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, str):
            # 可能是base64或文件路径
            if image_data.startswith('data:image'):
                # base64 with data URL
                import base64
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_bytes))
            elif os.path.exists(image_data):
                # 文件路径
                return Image.open(image_data)
            else:
                # 纯base64
                image_bytes = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        else:
            raise ValueError("不支持的图片数据格式")
    
    def _resize_image(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """调整图片尺寸"""
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return image
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成图片 - 支持多种模式"""
        with self._generation_lock:
            return self._generate_internal(prompt, **kwargs)
    
    def _generate_internal(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """内部生成方法 - 支持多种图生图模式"""
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
            
            # 线程级GPU隔离
            if self.physical_gpu_id != "cpu":
                target_gpu_id = int(self.physical_gpu_id)
                torch.cuda.set_device(target_gpu_id)
                device = f"cuda:{target_gpu_id}"
                logger.info(f"线程 {threading.current_thread().name} 设置GPU设备: {device}")
            else:
                device = "cpu"

            # 确定生成模式
            generation_mode = params.get('mode', 'text2img')
            logger.info(f"开始{generation_mode}生成，提示词: {prompt}，设备: {device}")

            with torch.no_grad():
                if generation_mode == 'text2img':
                    result = self._generate_text2img(prompt, params, generator)
                elif generation_mode == 'img2img':
                    result = self._generate_img2img(prompt, params, generator)
                elif generation_mode == 'fill':
                    result = self._generate_fill(prompt, params, generator)
                elif generation_mode == 'controlnet':
                    result = self._generate_controlnet(prompt, params, generator)
                else:
                    raise ValueError(f"不支持的生成模式: {generation_mode}")
            
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
            
            logger.info(f"{generation_mode}生成完成，耗时: {elapsed_time:.2f}秒")
            
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
                "thread_name": threading.current_thread().name,
                "mode": generation_mode
            }
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU内存不足: {e}")
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
            logger.error(f"图片生成失败: {e}")
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
                except Exception as e:
                    logger.warning(f"恢复GPU设备时出错: {e}")
    
    def _generate_text2img(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        """文本生成图片"""
        pipe = self._get_pipeline("text2img")
        return pipe(
            prompt=prompt,
            height=params['height'],
            width=params['width'],
            guidance_scale=params['cfg'],
            num_inference_steps=params['num_inference_steps'],
            max_sequence_length=512,
            generator=generator
        )
    
    def _generate_img2img(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        """图片生成图片"""
        pipe = self._get_pipeline("img2img")
        
        # 加载输入图片
        input_image = self._load_image(params['input_image'])
        input_image = self._resize_image(input_image, params['width'], params['height'])
        
        return pipe(
            prompt=prompt,
            image=input_image,
            strength=params.get('strength', 0.8),
            guidance_scale=params['cfg'],
            num_inference_steps=params['num_inference_steps'],
            max_sequence_length=512,
            generator=generator
        )
    
    def _generate_fill(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        """填充/修复生成"""
        pipe = self._get_pipeline("fill")
        
        # 加载输入图片和蒙版
        input_image = self._load_image(params['input_image'])
        mask_image = self._load_image(params['mask_image'])
        
        # 调整尺寸
        input_image = self._resize_image(input_image, params['width'], params['height'])
        mask_image = self._resize_image(mask_image, params['width'], params['height'])
        
        return pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            height=params['height'],
            width=params['width'],
            guidance_scale=params['cfg'],
            num_inference_steps=params['num_inference_steps'],
            max_sequence_length=512,
            generator=generator
        )
    
    def _generate_controlnet(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        """ControlNet生成"""
        pipe = self._get_pipeline("controlnet")
        
        # 加载输入图片和控制图片
        input_image = self._load_image(params['input_image'])
        control_image = self._load_image(params['control_image'])
        
        # 调整尺寸
        input_image = self._resize_image(input_image, params['width'], params['height'])
        control_image = self._resize_image(control_image, params['width'], params['height'])
        
        return pipe(
            prompt=prompt,
            image=input_image,
            control_image=control_image,
            height=params['height'],
            width=params['width'],
            guidance_scale=params['cfg'],
            num_inference_steps=params['num_inference_steps'],
            max_sequence_length=512,
            generator=generator
        )
    
    def load(self) -> bool:
        """加载Flux模型 - 只加载text2img pipeline"""
        try:
            # 检查模型路径是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                return False
            
            logger.info(f"正在加载模型: {self.model_name}，物理GPU: {self.physical_gpu_id}")
            
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
                    
                    # 只加载text2img pipeline
                    if self._load_pipeline("text2img"):
                        self.is_loaded = True
                        logger.info(f"FluxPipeline加载成功，使用物理GPU {self.physical_gpu_id}")
                        return True
                    else:
                        return False
                    
                else:
                    # CPU模式
                    if self._load_pipeline("text2img"):
                        self.is_loaded = True
                        logger.info(f"FluxPipeline加载到CPU成功")
                        return True
                    else:
                        return False
                
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
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "num_inference_steps": 50,
            "seed": 42,
            "cfg": 3.5,
            "height": 1024,
            "width": 1024,
            "save_disk_path": None,
            "mode": "text2img",  # 默认文本生成图片
            "strength": 0.8,  # img2img强度
            "input_image": None,  # 输入图片
            "mask_image": None,  # 蒙版图片（用于fill模式）
            "control_image": None,  # 控制图片（用于controlnet模式）
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
        
        # 验证模式特定参数
        mode = kwargs.get('mode', 'text2img')
        if mode == 'img2img' and not kwargs.get('input_image'):
            return False
        elif mode == 'fill' and (not kwargs.get('input_image') or not kwargs.get('mask_image')):
            return False
        elif mode == 'controlnet' and (not kwargs.get('input_image') or not kwargs.get('control_image')):
            return False
        
        return True
    
    def get_supported_features(self) -> list:
        """获取支持的功能列表"""
        return ["text-to-image", "image-to-image", "fill", "controlnet"]
    
    def get_optimization_kwargs(self) -> Dict[str, Any]:
        """获取优化参数 - Flux使用bfloat16"""
        kwargs = {}
        kwargs["torch_dtype"] = torch.bfloat16
        return kwargs

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
            for pipeline in self.pipelines.values():
                if pipeline is not None:
                    logger.info(f"开始清理pipeline组件 (实例: {self._instance_id})")
                    
                    # 获取pipeline中的所有组件
                    components_to_cleanup = []
                    
                    # 检查并收集需要清理的组件
                    for attr_name in ['transformer', 'vae', 'text_encoder', 'text_encoder_2', 
                                    'scheduler', 'tokenizer', 'tokenizer_2']:
                        if hasattr(pipeline, attr_name):
                            component = getattr(pipeline, attr_name)
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
                        if hasattr(pipeline, 'disable_model_cpu_offload'):
                            pipeline.disable_model_cpu_offload()
                            logger.debug("已禁用模型CPU offload")
                    except Exception as e:
                        logger.warning(f"禁用CPU offload时出错: {e}")
                    
                    # 尝试将整个pipeline移动到CPU
                    try:
                        pipeline.to('cpu')
                        logger.debug("已将pipeline移动到CPU")
                    except Exception as e:
                        logger.warning(f"移动pipeline到CPU时出错: {e}")
                    
                    # 删除pipeline引用
                    del pipeline
                    self.pipelines[name] = None
                    logger.info(f"Pipeline已删除 (实例: {self._instance_id})")
                
        except Exception as e:
            logger.error(f"清理pipeline时出错: {e}")
    

    def unload(self):
        """卸载模型"""
        logger.info(f"开始卸载模型 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        
        # 标记为未加载
        self.is_loaded = False
        
        # 清理pipeline
        for pipeline in self.pipelines.values():
            if pipeline is not None:
                try:
                    # 禁用CPU offload
                    if hasattr(pipeline, 'disable_model_cpu_offload'):
                        pipeline.disable_model_cpu_offload()
                    
                    # 删除pipeline
                    del pipeline
                    
                    # 清理GPU内存
                    if self.physical_gpu_id != "cpu":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                except Exception as e:
                    logger.error(f"卸载模型时出错: {e}")
        
        logger.info(f"模型 {self.model_id} 已卸载 (物理GPU: {self.physical_gpu_id})")