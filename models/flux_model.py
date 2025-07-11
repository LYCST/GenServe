import base64
import io
import time
import logging
import os
import threading
import gc
import random
import re
from typing import Dict, Any, Optional, Union, List, Callable

try:
    import torch
except ImportError:
    raise ImportError("未安装 torch，请先安装 torch")

try:
    from PIL import Image
except ImportError:
    raise ImportError("未安装 pillow，请先安装 pillow")

try:
    from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxFillPipeline, FluxControlPipeline, FluxPriorReduxPipeline
except ImportError:
    raise ImportError("未安装 diffusers，请先安装 diffusers")

from .base import BaseModel
from config import Config

logger = logging.getLogger(__name__)

def _safe_path(path: Optional[str], fallback: Optional[str] = None) -> str:
    if path and isinstance(path, str):
        return path
    if fallback:
        return fallback
    raise ValueError("模型路径未配置")

class FluxModel(BaseModel):
    """Flux模型实现 - 支持GPU隔离和多模式图生图"""
    
    def __init__(self, model_id: str = "flux1-dev", gpu_device: Optional[str] = None, physical_gpu_id: Optional[str] = None):
        model_info = self._get_model_info(model_id)
        super().__init__(
            model_id=model_id,
            model_name=model_info["name"],
            description=model_info["description"],
            gpu_device=gpu_device
        )
        self.model_path = _safe_path(Config.get_model_paths().get(model_id))
        self.physical_gpu_id = physical_gpu_id
        self._generation_lock = threading.Lock()
        self._instance_id = f"{model_id}_{gpu_device}_{id(self)}"
        self.pipelines: Dict[str, Union[None, dict, Callable[..., Any]]] = {
            "text2img": None,
            "img2img": None,
            "fill": None,
            "controlnet_depth": None,
            "controlnet_canny": None,
            "controlnet_openpose": None,
            "redux": None
        }
        logger.info(f"创建FluxModel实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id}")
    
    def _get_model_info(self, model_id: str) -> Dict[str, str]:
        model_info_map = {
            "flux1-dev": {"name": "FLUX.1-dev", "description": "Black Forest Labs FLUX.1-dev model for high-quality image generation"},
            "flux1-depth-dev": {"name": "FLUX.1-Depth-dev", "description": "FLUX.1 Depth ControlNet model for depth-guided generation"},
            "flux1-fill-dev": {"name": "FLUX.1-Fill-dev", "description": "FLUX.1 Fill model for inpainting and outpainting"},
            "flux1-canny-dev": {"name": "FLUX.1-Canny-dev", "description": "FLUX.1 Canny ControlNet model for edge-guided generation"},
            "flux1-openpose-dev": {"name": "FLUX.1-OpenPose-dev", "description": "FLUX.1 OpenPose ControlNet model for pose-guided generation"},
            "flux1-redux-dev": {"name": "FLUX.1-Redux-dev", "description": "FLUX.1 Redux model for high-quality image enhancement"}
        }
        return model_info_map.get(model_id, {"name": model_id, "description": f"FLUX model: {model_id}"})
    
    def _extract_gpu_id(self, device: Optional[str]) -> str:
        if not device:
            return "0"
        if device == "cpu":
            return "cpu"
        if device.startswith("cuda:"):
            return device.split(":")[1]
        return "0"
    
    def _load_pipeline(self, pipeline_type: str) -> bool:
        try:
            model_paths = Config.get_model_paths()
            if pipeline_type == "text2img":
                if self.model_id == "flux1-redux-dev":
                    base_model_path = _safe_path(model_paths.get("flux1-dev"), self.model_path)
                else:
                    base_model_path = self.model_path
                self.pipelines[pipeline_type] = None
                self.pipelines[pipeline_type] = FluxPipeline.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                # 启用CPU offload
                pipe_obj = self.pipelines[pipeline_type]
                if pipe_obj is not None and not isinstance(pipe_obj, dict) and hasattr(pipe_obj, "enable_model_cpu_offload"):
                    pipe_obj.enable_model_cpu_offload(device="cuda:0")
                    logger.info(f"text2img pipeline CPU offload已启用")
                
            elif pipeline_type == "img2img":
                base_model_path = _safe_path(model_paths.get("flux1-dev"), self.model_path)
                self.pipelines[pipeline_type] = None
                self.pipelines[pipeline_type] = FluxImg2ImgPipeline.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                # 启用CPU offload
                pipe_obj = self.pipelines[pipeline_type]
                if pipe_obj is not None and not isinstance(pipe_obj, dict) and hasattr(pipe_obj, "enable_model_cpu_offload"):
                    pipe_obj.enable_model_cpu_offload(device="cuda:0")
                    logger.info(f"img2img pipeline CPU offload已启用")
                
            elif pipeline_type == "fill":
                fill_model_path = _safe_path(model_paths.get("flux1-fill-dev"), self.model_path)
                if not os.path.exists(fill_model_path):
                    logger.warning(f"Fill模型路径不存在: {fill_model_path}，使用基础模型")
                    fill_model_path = _safe_path(self.model_path)
                self.pipelines[pipeline_type] = None
                self.pipelines[pipeline_type] = FluxFillPipeline.from_pretrained(
                    fill_model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                # 启用CPU offload
                pipe_obj = self.pipelines[pipeline_type]
                if pipe_obj is not None and not isinstance(pipe_obj, dict) and hasattr(pipe_obj, "enable_model_cpu_offload"):
                    pipe_obj.enable_model_cpu_offload(device="cuda:0")
                    logger.info(f"fill pipeline CPU offload已启用")
                
            elif pipeline_type.startswith("controlnet_"):
                controlnet_type = pipeline_type.split("_")[1]
                if controlnet_type == 'depth':
                    controlnet_model_path = _safe_path(model_paths.get("flux1-depth-dev"), self.model_path)
                elif controlnet_type == 'canny':
                    controlnet_model_path = _safe_path(model_paths.get("flux1-canny-dev"), self.model_path)
                elif controlnet_type == 'openpose':
                    controlnet_model_path = _safe_path(model_paths.get("flux1-openpose-dev"), self.model_path)
                else:
                    raise ValueError(f"不支持的controlnet类型: {controlnet_type}")
                if not os.path.exists(controlnet_model_path):
                    logger.warning(f"ControlNet模型路径不存在: {controlnet_model_path}，使用基础模型")
                    controlnet_model_path = _safe_path(self.model_path)
                self.pipelines[pipeline_type] = None
                self.pipelines[pipeline_type] = FluxControlPipeline.from_pretrained(
                    controlnet_model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                # 启用CPU offload
                pipe_obj = self.pipelines[pipeline_type]
                if pipe_obj is not None and not isinstance(pipe_obj, dict) and hasattr(pipe_obj, "enable_model_cpu_offload"):
                    pipe_obj.enable_model_cpu_offload(device="cuda:0")
                    logger.info(f"controlnet {controlnet_type} pipeline CPU offload已启用")
                
            elif pipeline_type == "redux":
                redux_model_path = _safe_path(model_paths.get("flux1-redux-dev"), self.model_path)
                base_model_path = _safe_path(model_paths.get("flux1-dev"), self.model_path)
                if not os.path.exists(redux_model_path):
                    logger.warning(f"Redux模型路径不存在: {redux_model_path}，使用基础模型")
                    redux_model_path = _safe_path(self.model_path)
                self.pipelines[pipeline_type] = None
                prior = FluxPriorReduxPipeline.from_pretrained(
                    redux_model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                base = FluxPipeline.from_pretrained(
                    base_model_path,
                    text_encoder=None,
                    text_encoder_2=None,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True
                )
                self.pipelines[pipeline_type] = {"prior": prior, "base": base}
                # 为redux的两个pipeline都启用CPU offload
                if self.pipelines[pipeline_type] is not None and isinstance(self.pipelines[pipeline_type], dict):
                    prior_obj = self.pipelines[pipeline_type].get("prior")
                    base_obj = self.pipelines[pipeline_type].get("base")
                    if prior_obj is not None and hasattr(prior_obj, "enable_model_cpu_offload"):
                        prior_obj.enable_model_cpu_offload(device="cuda:0")
                        logger.info(f"redux prior pipeline CPU offload已启用")
                    if base_obj is not None and hasattr(base_obj, "enable_model_cpu_offload"):
                        base_obj.enable_model_cpu_offload(device="cuda:0")
                        logger.info(f"redux base pipeline CPU offload已启用")
                logger.info(f"Redux pipeline加载完成")
                return True
                
            # 检查pipeline是否成功加载
            if self.pipelines[pipeline_type]:
                logger.info(f"成功加载 {pipeline_type} pipeline")
                return True
            else:
                logger.error(f"{pipeline_type} pipeline 加载失败")
                return False
                
        except Exception as e:
            logger.error(f"加载 {pipeline_type} pipeline失败: {e}")
            return False
    
    def _get_pipeline(self, pipeline_type: str):
        pipe = self.pipelines.get(pipeline_type)
        if pipe is None:
            if not self._load_pipeline(pipeline_type):
                raise RuntimeError(f"无法加载 {pipeline_type} pipeline")
            pipe = self.pipelines.get(pipeline_type)
        if pipe is None:
            raise RuntimeError(f"{pipeline_type} pipeline 加载失败")
        return pipe
    
    def _load_image_from_base64(self, image_data: Optional[str]) -> Image.Image:
        if not image_data:
            raise ValueError("图片数据为空")
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"加载图片失败: {str(e)}")
    
    def _load_image(self, image_data: Union[str, bytes, Image.Image, None]) -> Image.Image:
        if image_data is None:
            raise ValueError("图片数据不能为空")
        if isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, str):
            if image_data.startswith('data:image') or len(image_data) > 100:
                return self._load_image_from_base64(image_data)
            elif os.path.exists(image_data):
                image = Image.open(image_data)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return image
            else:
                return self._load_image_from_base64(image_data)
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        else:
            raise ValueError("不支持的图片数据格式")
    
    def _get_output_dimensions(self, input_image: Image.Image, mode: str, task_params: Dict[str, Any]) -> tuple:
        user_height = task_params.get('height')
        user_width = task_params.get('width')
        if user_height is None or user_width is None:
            raise ValueError("height/width 不能为空")
        original_width, original_height = input_image.size
        if user_height == 1024 and user_width == 1024 and mode != 'text2img':
            height = original_height
            width = original_width
            logger.info(f"使用图片原始尺寸: {width}x{height}")
        else:
            height = user_height
            width = user_width
            logger.info(f"使用用户指定尺寸: {width}x{height}")
        if mode == 'controlnet':
            target_size = max(width, height)
            if target_size <= 512:
                width = height = 512
            elif target_size <= 768:
                width = height = 768
            else:
                width = height = 1024
            logger.info(f"ControlNet使用标准尺寸: {width}x{height}")
        return width, height
    
    def _create_safe_adapter_name(self, lora_name: Optional[str], index: int, timestamp: int, random_suffix: int) -> str:
        if not lora_name:
            raise ValueError("LoRA name 不能为空")
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', lora_name)
        if safe_name and safe_name[0].isdigit():
            safe_name = 'lora_' + safe_name
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return f"lora_{index}_{safe_name}_{timestamp}_{random_suffix}"
    
    def _load_and_apply_loras(self, pipe, loras: List[Dict[str, Any]]) -> None:
        if not loras:
            return
        try:
            from peft import PeftModel
        except ImportError:
            raise ValueError("PEFT库未安装，无法使用LoRA功能。请安装: pip install peft")
        logger.info(f"开始加载 {len(loras)} 个LoRA...")
        adapter_names = []
        adapter_weights = []
        timestamp = int(time.time() * 1000)
        random_suffix = random.randint(1000, 9999)
        for i, lora in enumerate(loras):
            lora_name = lora.get('name')
            if not lora_name:
                raise ValueError("LoRA name 不能为空")
            lora_weight = lora.get('weight', 1.0)
            lora_path = Config.get_lora_path(lora_name)
            if not lora_path:
                raise ValueError(f"LoRA '{lora_name}' 不存在")
            adapter_name = self._create_safe_adapter_name(lora_name, i, timestamp, random_suffix)
            adapter_names.append(adapter_name)
            adapter_weights.append(lora_weight)
            try:
                logger.info(f"加载LoRA: {lora_name} (权重: {lora_weight}) -> adapter: {adapter_name}")
                if not os.path.exists(lora_path):
                    raise ValueError(f"LoRA文件不存在: {lora_path}")
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                logger.info(f"LoRA {lora_name} 加载成功")
            except Exception as e:
                error_msg = str(e)
                if "PEFT backend is required" in error_msg:
                    raise ValueError(f"PEFT后端未正确配置。请确保安装了正确的PEFT版本: pip install peft transformers")
                elif "not a valid LoRA" in error_msg:
                    raise ValueError(f"LoRA文件格式无效或与当前模型不兼容: {lora_name}")
                elif "already in use" in error_msg:
                    raise ValueError(f"LoRA适配器名称冲突，请重试: {lora_name}")
                else:
                    raise ValueError(f"加载LoRA {lora_name} 失败: {error_msg}")
        if adapter_names:
            try:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info(f"设置LoRA适配器: {adapter_names} (权重: {adapter_weights})")
            except Exception as e:
                logger.error(f"设置LoRA适配器失败: {e}")
                raise ValueError(f"设置LoRA适配器失败: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        with self._generation_lock:
            return self._generate_internal(prompt, **kwargs)
    
    def _generate_internal(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        if not self.validate_prompt(prompt):
            raise ValueError("提示词验证失败")
        params = self.get_default_params()
        params.update(kwargs)
        if not self.validate_params(**params):
            raise ValueError("参数验证失败")
        start_time = time.time()
        original_device = None
        if torch.cuda.is_available():
            original_device = torch.cuda.current_device()
        try:
            generator = torch.Generator("cpu").manual_seed(params['seed'])
            if self.physical_gpu_id != "cpu":
                # 在GPU工作进程中，由于CUDA_VISIBLE_DEVICES限制，只能看到GPU 0
                # 所以使用当前可见的GPU 0，而不是原始的physical_gpu_id
                if torch.cuda.device_count() > 0:
                    torch.cuda.set_device(0)  # 使用可见的GPU 0
                    device = "cuda:0"
                    logger.info(f"线程 {threading.current_thread().name} 设置GPU设备: {device} (物理GPU: {self.physical_gpu_id})")
                else:
                    device = "cpu"
                    logger.warning(f"没有可用的GPU，使用CPU (物理GPU: {self.physical_gpu_id})")
            else:
                device = "cpu"
            generation_mode = params.get('mode', 'text2img')
            logger.info(f"开始{generation_mode}生成，设备: {device}")
            with torch.no_grad():
                if generation_mode == 'text2img':
                    result = self._generate_text2img(prompt, params, generator)
                elif generation_mode == 'img2img':
                    result = self._generate_img2img(prompt, params, generator)
                elif generation_mode == 'fill':
                    result = self._generate_fill(prompt, params, generator)
                elif generation_mode == 'controlnet':
                    result = self._generate_controlnet(prompt, params, generator)
                elif generation_mode == 'redux':
                    result = self._generate_redux(prompt, params, generator)
                else:
                    raise ValueError(f"不支持的生成模式: {generation_mode}")
            if not hasattr(result, 'images') or not result.images:
                raise RuntimeError(f"{generation_mode} 生成失败，未返回图片")
            image = result.images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            elapsed_time = time.time() - start_time
            if self.physical_gpu_id != "cpu":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
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
                "image_base64": img_base64,
                "elapsed_time": elapsed_time,
                "save_to_disk": save_to_disk,
                "params": params,
                "device": device,
                "physical_gpu_id": self.physical_gpu_id,
                "instance_id": self._instance_id,
                "thread_name": threading.current_thread().name,
                "mode": generation_mode,
                "controlnet_type": params.get('controlnet_type') if generation_mode == 'controlnet' else None
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
            if original_device is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(original_device)
                except Exception as e:
                    logger.warning(f"恢复GPU设备时出错: {e}")
    
    def _generate_text2img(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        pipe = self._get_pipeline("text2img")
        loras = params.get('loras', [])
        if loras:
            self._load_and_apply_loras(pipe, loras)
        if not callable(pipe):
            raise RuntimeError("text2img pipeline 未加载")
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
        pipe = self._get_pipeline("img2img")
        input_image = self._load_image(params.get('input_image'))
        width, height = self._get_output_dimensions(input_image, 'img2img', params)
        loras = params.get('loras', [])
        if loras:
            self._load_and_apply_loras(pipe, loras)
        if not callable(pipe):
            raise RuntimeError("img2img pipeline 未加载")
        return pipe(
            prompt=prompt,
            image=input_image,
            height=height,
            width=width,
            strength=params.get('strength', 0.8),
            guidance_scale=params['cfg'],
            num_inference_steps=params['num_inference_steps'],
            max_sequence_length=512,
            generator=generator
        )
    
    def _generate_fill(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        pipe = self._get_pipeline("fill")
        input_image = self._load_image(params.get('input_image'))
        mask_image = self._load_image(params.get('mask_image'))
        width, height = self._get_output_dimensions(input_image, 'fill', params)
        loras = params.get('loras', [])
        if loras:
            self._load_and_apply_loras(pipe, loras)
        if not callable(pipe):
            raise RuntimeError("fill pipeline 未加载")
        return pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            height=height,
            width=width,
            guidance_scale=params['cfg'],
            num_inference_steps=params['num_inference_steps'],
            max_sequence_length=512,
            generator=generator
        )
    
    def _generate_controlnet(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        controlnet_type = params.get('controlnet_type', 'depth').lower()
        pipeline_key = f"controlnet_{controlnet_type}"
        pipe = self._get_pipeline(pipeline_key)
        control_image = self._load_image(params.get('control_image'))
        width, height = self._get_output_dimensions(control_image, 'controlnet', params)
        if params.get('input_image'):
            logger.debug(f"ControlNet模式检测到input_image，但主要使用control_image")
        controlnet_kwargs = {
            'prompt': prompt,
            'control_image': control_image,
            'height': height,
            'width': width,
            'guidance_scale': params['cfg'],
            'num_inference_steps': params['num_inference_steps'],
            'max_sequence_length': 512,
            'generator': generator
        }
        if params.get('controlnet_conditioning_scale') is not None:
            logger.debug(f"用户提供了controlnet_conditioning_scale参数，但FluxControlPipeline不支持此参数")
        if params.get('control_guidance_start') is not None:
            logger.debug(f"用户提供了control_guidance_start参数，但FluxControlPipeline不支持此参数")
        if params.get('control_guidance_end') is not None:
            logger.debug(f"用户提供了control_guidance_end参数，但FluxControlPipeline不支持此参数")
        if not callable(pipe):
            raise RuntimeError("controlnet pipeline 未加载")
        return pipe(**controlnet_kwargs)
    
    def _generate_redux(self, prompt: str, params: Dict[str, Any], generator) -> Any:
        pipe = self._get_pipeline("redux")
        input_image = self._load_image(params.get('input_image'))
        width, height = self._get_output_dimensions(input_image, 'redux', params)
        logger.debug(f"开始Redux处理，输入图片尺寸: {input_image.size}")
        if not isinstance(pipe, dict) or 'prior' not in pipe or 'base' not in pipe:
            raise RuntimeError("redux pipeline 未正确加载")
        prior_obj = pipe.get("prior")
        base_obj = pipe.get("base")
        if not callable(prior_obj) or not callable(base_obj):
            raise RuntimeError("redux pipeline 的 prior/base 未正确加载")
        try:
            logger.debug(f"运行Redux prior pipeline...")
            prior_output = prior_obj(input_image)
            logger.debug(f"运行Redux base pipeline...")
            result = base_obj(
                guidance_scale=params['cfg'],
                num_inference_steps=params['num_inference_steps'],
                generator=generator,
                **prior_output
            )
            logger.debug(f"Redux处理完成")
            return result
        except Exception as e:
            logger.error(f"Redux处理失败: {e}")
            raise ValueError(f"Redux处理失败: {str(e)}")
    
    def load(self) -> bool:
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                return False
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 在GPU工作进程中，CUDA_VISIBLE_DEVICES已经设置，不需要再次设置
            if self.physical_gpu_id != "cpu":
                if not torch.cuda.is_available():
                    logger.error(f"GPU不可用")
                    return False
                if self._load_pipeline("text2img"):
                    self.is_loaded = True
                    logger.info(f"FluxPipeline加载成功")
                    return True
                else:
                    return False
            else:
                if self._load_pipeline("text2img"):
                    self.is_loaded = True
                    logger.info(f"FluxPipeline加载到CPU成功")
                    return True
                else:
                    return False
        except Exception as e:
            logger.error(f"模型 {self.model_name} 加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            "num_inference_steps": 50,
            "seed": 42,
            "cfg": 3.5,
            "height": 1024,
            "width": 1024,
            "save_disk_path": None,
            "mode": "text2img",
            "strength": 0.8,
            "input_image": None,
            "mask_image": None,
            "control_image": None,
            "controlnet_type": "depth",
            "loras": [],
        }
    
    def validate_params(self, **kwargs) -> bool:
        required_params = ['num_inference_steps', 'seed', 'cfg', 'height', 'width']
        for param in required_params:
            if param not in kwargs or kwargs[param] is None:
                return False
        if kwargs['num_inference_steps'] < 1 or kwargs['num_inference_steps'] > 100:
            return False
        if kwargs['cfg'] < 0 or kwargs['cfg'] > 50:
            return False
        if not self.validate_image_size(kwargs['height'], kwargs['width']):
            return False
        mode = kwargs.get('mode', 'text2img')
        if mode == 'img2img' and not kwargs.get('input_image'):
            return False
        elif mode == 'fill' and (not kwargs.get('input_image') or not kwargs.get('mask_image')):
            return False
        elif mode == 'controlnet' and not kwargs.get('control_image'):
            return False
        elif mode == 'redux' and not kwargs.get('input_image'):
            return False
        return True
    
    def get_supported_features(self) -> list:
        features = ["text-to-image"]
        if self.model_id in ["flux1-dev", "flux1-fill-dev"]:
            features.append("image-to-image")
        if self.model_id == "flux1-fill-dev":
            features.append("fill")
        if self.model_id in ["flux1-depth-dev", "flux1-canny-dev", "flux1-openpose-dev"]:
            features.append("controlnet")
        if self.model_id == "flux1-redux-dev":
            features.append("redux")
        return features
    
    def get_optimization_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        kwargs["torch_dtype"] = torch.bfloat16
        return kwargs
    
    def _emergency_cleanup(self):
        logger.warning(f"执行紧急清理 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        try:
            if self.physical_gpu_id != "cpu":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self._safe_pipeline_cleanup()
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("紧急清理完成")
        except Exception as e:
            logger.error(f"紧急清理时出错: {e}")
    
    def _safe_pipeline_cleanup(self):
        try:
            for name, pipeline in self.pipelines.items():
                if pipeline is not None:
                    logger.debug(f"开始清理pipeline组件 (实例: {self._instance_id})")
                    if isinstance(pipeline, dict):
                        for sub_name in list(pipeline.keys()):
                            sub_pipeline = pipeline[sub_name]
                            if sub_pipeline is not None:
                                self._cleanup_single_pipeline(sub_pipeline, f"{name}_{sub_name}")
                    else:
                        self._cleanup_single_pipeline(pipeline, name)
                    self.pipelines[name] = None
                    logger.debug(f"Pipeline {name} 已删除 (实例: {self._instance_id})")
        except Exception as e:
            logger.error(f"清理pipeline时出错: {e}")
    
    def _cleanup_single_pipeline(self, pipeline, pipeline_name: str):
        try:
            components_to_cleanup = []
            for attr_name in ['transformer', 'vae', 'scheduler', 'tokenizer', 'tokenizer_2']:
                if hasattr(pipeline, attr_name):
                    component = getattr(pipeline, attr_name)
                    if component is not None:
                        components_to_cleanup.append((attr_name, component))
            for name, component in components_to_cleanup:
                try:
                    if hasattr(component, 'to') and hasattr(component, 'parameters'):
                        component.to('cpu')
                        logger.debug(f"已将 {name} 移动到CPU")
                    if hasattr(component, 'cuda'):
                        try:
                            component.cpu()
                            logger.debug(f"已将 {name} 移动到CPU")
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"清理组件 {name} 时出错: {e}")
            try:
                if hasattr(pipeline, 'disable_model_cpu_offload'):
                    pipeline.disable_model_cpu_offload()
                    logger.debug("已禁用模型CPU offload")
            except Exception as e:
                logger.warning(f"禁用CPU offload时出错: {e}")
            try:
                pipeline.to('cpu')
                logger.debug("已将pipeline移动到CPU")
            except Exception as e:
                logger.warning(f"移动pipeline到CPU时出错: {e}")
        except Exception as e:
            logger.error(f"清理pipeline {pipeline_name} 时出错: {e}")
    
    def unload(self):
        logger.info(f"开始卸载模型 (实例: {self._instance_id}, 物理GPU: {self.physical_gpu_id})")
        self.is_loaded = False
        for name, pipeline in self.pipelines.items():
            if pipeline is not None:
                try:
                    if isinstance(pipeline, dict):
                        for sub_pipeline in list(pipeline.values()):
                            if sub_pipeline is not None and hasattr(sub_pipeline, 'disable_model_cpu_offload'):
                                sub_pipeline.disable_model_cpu_offload()
                            del sub_pipeline
                    else:
                        if hasattr(pipeline, 'disable_model_cpu_offload'):
                            pipeline.disable_model_cpu_offload()
                        del pipeline
                    if self.physical_gpu_id != "cpu":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.error(f"卸载模型时出错: {e}")
        self.pipelines = {key: None for key in self.pipelines}
        logger.info(f"模型 {self.model_id} 已卸载") 