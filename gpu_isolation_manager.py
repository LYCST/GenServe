import os
import subprocess
import multiprocessing as mp
from typing import Dict, Any, Optional, Callable
import logging
import time
import signal
import sys
import traceback
import queue
import gc
import psutil

# mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)

def gpu_worker_process(gpu_id: str, model_path: str, model_id: str, task_queue, result_queue):
    """GPU工作进程 - 在隔离环境中运行（顶层函数，支持spawn）"""
    import torch
    from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxFillPipeline, FluxControlPipeline
    import base64
    import io
    from PIL import Image
    import traceback
    import queue
    import logging
    import time
    import gc
    import psutil
    import os
    
    # 尝试导入PEFT相关库
    try:
        from peft import PeftModel
        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False
        logging.warning(f"PEFT库未安装，LoRA功能将不可用")
    
    # 注意：PeftConfig在新版本transformers中可能不可用，我们只需要PEFT库本身
    TRANSFORMERS_PEFT_AVAILABLE = True  # 简化检查，主要依赖PEFT库
    
    logger = logging.getLogger(f"gpu_worker_{gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    def load_image_from_base64(image_data: str):
        """从base64加载图片"""
        if not image_data:
            raise ValueError("图片数据为空")
        
        try:
            # 处理data URL格式
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # 解码base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            raise ValueError(f"加载图片失败: {str(e)}")
    
    # 设置进程优先级和内存限制
    try:
        process = psutil.Process()
        process.nice(10)  # 降低进程优先级，减少被OOM Killer杀死的概率
        logger.info(f"🚀 GPU {gpu_id} 工作进程启动 (PID: {os.getpid()}, 优先级: {process.nice()})")
    except Exception as e:
        logger.warning(f"无法设置进程优先级: {e}")
        logger.info(f"🚀 GPU {gpu_id} 工作进程启动 (PID: {os.getpid()})")
    
    try:
        logger.info(f"正在加载模型到GPU {gpu_id}...")
        
        # 加载前清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 根据模型ID确定实际的模型路径
        from config import Config
        model_paths = Config.get_model_paths()
        actual_model_path = model_paths.get(model_id, model_path)
        
        logger.info(f"GPU {gpu_id} 使用模型路径: {actual_model_path}")
        
        # 初始化pipeline字典 - 支持多种controlnet类型
        pipelines = {
            "text2img": None,
            "img2img": None,
            "fill": None,
            "controlnet_depth": None,
            "controlnet_canny": None,
            "controlnet_openpose": None
        }
        
        # 加载基础text2img pipeline
        pipelines["text2img"] = FluxPipeline.from_pretrained(
            actual_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=True
        )
        pipelines["text2img"].enable_model_cpu_offload(device="cuda:0")
        logger.info(f"✅ 基础模型已加载到GPU {gpu_id}")
        
        task_count = 0
        consecutive_failures = 0  # 连续失败计数
        max_consecutive_failures = 3  # 最大连续失败次数
        last_cleanup_time = time.time()  # 上次清理时间
        
        while True:
            try:
                # 定期内存清理
                current_time = time.time()
                if current_time - last_cleanup_time > Config.GPU_MEMORY_CLEANUP_INTERVAL:
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        if allocated > Config.GPU_MEMORY_THRESHOLD_MB:
                            logger.info(f"GPU {gpu_id} 定期清理内存 (已分配: {allocated:.1f}MB)")
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            gc.collect()
                    last_cleanup_time = current_time
                
                task = task_queue.get(timeout=1.0)
                if task is None:
                    logger.info(f"GPU {gpu_id} 收到退出信号")
                    break
                
                task_count += 1
                logger.info(f"GPU {gpu_id} 开始处理任务 #{task_count}: {task.get('task_id', 'unknown')}")
                
                # 任务开始前检查内存状态
                if torch.cuda.is_available():
                    initial_allocated = torch.cuda.memory_allocated() / 1024**2
                    initial_cached = torch.cuda.memory_reserved() / 1024**2
                    logger.info(f"GPU {gpu_id} 任务开始前内存: 已分配 {initial_allocated:.1f}MB, 缓存 {initial_cached:.1f}MB")
                    
                    # 如果内存使用过高，强制清理
                    if initial_allocated > Config.GPU_MEMORY_THRESHOLD_MB:
                        logger.warning(f"GPU {gpu_id} 内存使用过高 ({initial_allocated:.1f}MB > {Config.GPU_MEMORY_THRESHOLD_MB}MB)，强制清理")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                
                # 处理任务
                result = process_generation_task(pipelines, task, gpu_id, load_image_from_base64, actual_model_path)
                result_queue.put(result)
                
                success = result.get('success', False)
                logger.info(f"GPU {gpu_id} 任务 #{task_count} 处理完成: {success}")
                
                # 更新连续失败计数
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
                # 任务完成后进行清理和等待
                if success:
                    logger.info(f"GPU {gpu_id} 开始清理资源...")
                    
                    # 更激进的内存清理
                    if torch.cuda.is_available():
                        # 多次清理确保彻底
                        for i in range(3):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            time.sleep(0.1)
                        
                        torch.cuda.reset_peak_memory_stats()
                        
                        # 如果启用激进清理，进行额外清理
                        if Config.ENABLE_AGGRESSIVE_CLEANUP:
                            # 强制垃圾回收多次
                            for i in range(2):
                                gc.collect()
                                time.sleep(0.05)
                    
                    # 强制垃圾回收
                    gc.collect()
                    
                    # 等待一段时间确保清理完成
                    cleanup_wait_time = Config.GPU_TASK_CLEANUP_WAIT_TIME
                    logger.info(f"GPU {gpu_id} 等待 {cleanup_wait_time} 秒确保清理完成...")
                    time.sleep(cleanup_wait_time)
                    
                    # 记录清理后的内存状态
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        cached = torch.cuda.memory_reserved() / 1024**2
                        logger.info(f"GPU {gpu_id} 清理后内存: 已分配 {allocated:.1f}MB, 缓存 {cached:.1f}MB")
                        
                        # 如果内存仍然过高，进行额外清理
                        if allocated > Config.GPU_MEMORY_THRESHOLD_MB:
                            logger.warning(f"GPU {gpu_id} 清理后内存仍然过高 ({allocated:.1f}MB)，进行额外清理")
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            gc.collect()
                    
                    logger.info(f"GPU {gpu_id} 清理完成，准备接收下一个任务")
                else:
                    logger.warning(f"GPU {gpu_id} 任务失败，跳过清理等待")
                    
                    # 失败后也要清理内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                
                # 检查连续失败次数
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"GPU {gpu_id} 连续失败 {consecutive_failures} 次，准备重启进程")
                    break
                
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"GPU {gpu_id} 处理任务时出错: {str(e)}"
                logger.error(error_msg)
                logger.error(f"错误详情: {traceback.format_exc()}")
                
                consecutive_failures += 1
                
                try:
                    result_queue.put({
                        "success": False,
                        "error": str(e),
                        "gpu_id": gpu_id,
                        "task_id": task.get('task_id') if 'task' in locals() else 'unknown'
                    })
                except Exception as put_error:
                    logger.error(f"GPU {gpu_id} 无法返回错误结果: {put_error}")
                
                # 异常后清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
        
        logger.info(f"GPU {gpu_id} 开始最终清理资源...")
        # 清理所有pipeline
        for pipeline in pipelines.values():
            if pipeline is not None:
                del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info(f"GPU {gpu_id} 工作进程退出")
    except Exception as e:
        error_msg = f"GPU {gpu_id} 工作进程启动失败: {str(e)}"
        logger.error(error_msg)
        logger.error(f"启动错误详情: {traceback.format_exc()}")
        try:
            result_queue.put({
                "success": False,
                "error": f"进程启动失败: {str(e)}",
                "gpu_id": gpu_id
            })
        except Exception as put_error:
            logger.error(f"GPU {gpu_id} 无法返回启动失败结果: {put_error}")

def process_generation_task(pipelines, task, gpu_id: str, load_image_func, model_path: str):
    import torch
    import io
    import base64
    import time
    import traceback
    from PIL import Image
    import os
    from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxFillPipeline, FluxControlPipeline
    
    # 尝试导入PEFT相关库
    try:
        from peft import PeftModel
        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False
        logging.warning(f"PEFT库未安装，LoRA功能将不可用")
    
    # 注意：PeftConfig在新版本transformers中可能不可用，我们只需要PEFT库本身
    TRANSFORMERS_PEFT_AVAILABLE = True  # 简化检查，主要依赖PEFT库
    
    logger = logging.getLogger(f"gpu_worker_{gpu_id}")
    start_time = time.time()
    
    def create_safe_adapter_name(lora_name: str, index: int, timestamp: int, random_suffix: int) -> str:
        """创建安全的adapter名称"""
        # 移除所有特殊字符，只保留字母、数字和下划线
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', lora_name)
        # 确保不以数字开头
        if safe_name and safe_name[0].isdigit():
            safe_name = 'lora_' + safe_name
        # 限制长度
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return f"lora_{index}_{safe_name}_{timestamp}_{random_suffix}"
    
    # 记录任务开始时的内存状态
    if torch.cuda.is_available():
        initial_allocated = torch.cuda.memory_allocated() / 1024**2
        initial_cached = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU {gpu_id} 任务开始前内存: 已分配 {initial_allocated:.1f}MB, 缓存 {initial_cached:.1f}MB")
    
    try:
        logger.info(f"GPU {gpu_id} 开始生成任务: {task.get('task_id', 'unknown')}")
        generator = torch.Generator("cpu").manual_seed(task.get('seed', 42))
        
        # 获取生成模式和模型ID
        mode = task.get('mode', 'text2img')
        model_id = task.get('model_id', 'flux1-dev')
        logger.info(f"GPU {gpu_id} 使用模式: {mode}, 模型: {model_id}")
        
        # 根据模型ID确定实际的模型路径
        from config import Config
        model_paths = Config.get_model_paths()
        actual_model_path = model_paths.get(model_id, model_path)
        
        # 通用尺寸处理逻辑
        def get_output_dimensions(input_image, mode):
            """获取输出尺寸：优先使用用户指定，否则使用图片原始尺寸，并确保是16的倍数"""
            user_height = task.get('height')
            user_width = task.get('width')
            original_width, original_height = input_image.size  # PIL返回(width, height)
            
            # 检查是否用户明确指定了尺寸（不是默认值）
            if user_height == 1024 and user_width == 1024 and mode != 'text2img':
                height = original_height
                width = original_width
                logger.info(f"GPU {gpu_id} 使用图片原始尺寸: {width}x{height}")
            else:
                height = user_height
                width = user_width
                logger.info(f"GPU {gpu_id} 使用用户指定尺寸: {width}x{height}")
            
            # 确保尺寸是16的倍数（ControlNet要求）
            if mode == 'controlnet':
                # Flux ControlNet对尺寸有特殊要求
                # 使用标准尺寸：512x512, 768x768, 1024x1024
                # 避免使用非标准尺寸，可能导致latent packing错误
                
                # 计算最接近的标准尺寸
                target_size = max(width, height)
                if target_size <= 512:
                    width = height = 512
                elif target_size <= 768:
                    width = height = 768
                else:
                    width = height = 1024
                
                logger.info(f"GPU {gpu_id} ControlNet使用标准尺寸: {width}x{height}")
            
            return width, height
        
        # 根据模式选择或加载pipeline
        if mode == 'text2img':
            pipe = pipelines["text2img"]
        elif mode == 'img2img':
            if pipelines["img2img"] is None:
                logger.info(f"GPU {gpu_id} 加载img2img pipeline...")
                
                # 使用专门的模型路径（如果有的话）
                model_paths = Config.get_model_paths()
                img2img_model_path = model_paths.get("flux1-dev", actual_model_path)  # img2img使用基础模型
                
                if not os.path.exists(img2img_model_path):
                    # 如果路径不存在，使用传入的路径
                    img2img_model_path = actual_model_path
                    logger.warning(f"GPU {gpu_id} img2img模型路径不存在，使用传入路径: {img2img_model_path}")
                else:
                    logger.info(f"GPU {gpu_id} 使用img2img模型路径: {img2img_model_path}")
                
                try:
                    pipelines["img2img"] = FluxImg2ImgPipeline.from_pretrained(
                        img2img_model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    pipelines["img2img"].enable_model_cpu_offload(device="cuda:0")
                    logger.info(f"GPU {gpu_id} img2img pipeline加载完成")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 加载img2img pipeline失败: {e}")
                    # 如果img2img加载失败，尝试使用基础pipeline
                    logger.info(f"GPU {gpu_id} 尝试使用基础pipeline进行img2img操作")
                    pipelines["img2img"] = pipelines["text2img"]
            pipe = pipelines["img2img"]
        elif mode == 'fill':
            if pipelines["fill"] is None:
                logger.info(f"GPU {gpu_id} 加载fill pipeline...")
                
                # 使用专门的Fill模型路径
                model_paths = Config.get_model_paths()
                fill_model_path = model_paths.get("flux1-fill-dev", actual_model_path)
                
                if not os.path.exists(fill_model_path):
                    # 如果专用Fill模型路径不存在，使用基础路径
                    fill_model_path = actual_model_path
                    logger.warning(f"GPU {gpu_id} 专用Fill模型路径不存在，使用基础模型: {fill_model_path}")
                else:
                    logger.info(f"GPU {gpu_id} 使用Fill模型路径: {fill_model_path}")
                
                try:
                    pipelines["fill"] = FluxFillPipeline.from_pretrained(
                        fill_model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    pipelines["fill"].enable_model_cpu_offload(device="cuda:0")
                    logger.info(f"GPU {gpu_id} fill pipeline加载完成")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 加载fill pipeline失败: {e}")
                    # 如果fill加载失败，尝试使用基础pipeline
                    logger.info(f"GPU {gpu_id} 尝试使用基础pipeline进行fill操作")
                    pipelines["fill"] = pipelines["text2img"]
            pipe = pipelines["fill"]
        elif mode == 'controlnet':
            # 获取controlnet类型
            controlnet_type = task.get('controlnet_type', 'depth').lower()
            pipeline_key = f"controlnet_{controlnet_type}"
            
            if pipeline_key not in pipelines:
                raise ValueError(f"不支持的controlnet类型: {controlnet_type}")
            
            if pipelines[pipeline_key] is None:
                logger.info(f"GPU {gpu_id} 加载{controlnet_type} controlnet pipeline...")
                
                # 根据类型选择模型路径
                model_paths = Config.get_model_paths()
                if controlnet_type == 'depth':
                    controlnet_model_path = model_paths.get("flux1-depth-dev", actual_model_path)
                elif controlnet_type == 'canny':
                    controlnet_model_path = model_paths.get("flux1-canny-dev", actual_model_path)
                elif controlnet_type == 'openpose':
                    controlnet_model_path = model_paths.get("flux1-openpose-dev", actual_model_path)
                else:
                    raise ValueError(f"不支持的controlnet类型: {controlnet_type}")
                
                if not os.path.exists(controlnet_model_path):
                    # 如果专用路径不存在，使用基础路径
                    controlnet_model_path = actual_model_path
                    logger.warning(f"GPU {gpu_id} 专用{controlnet_type} controlnet模型路径不存在，使用基础模型: {controlnet_model_path}")
                else:
                    logger.info(f"GPU {gpu_id} 使用{controlnet_type} controlnet模型路径: {controlnet_model_path}")
                
                try:
                    pipelines[pipeline_key] = FluxControlPipeline.from_pretrained(
                        controlnet_model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    pipelines[pipeline_key].enable_model_cpu_offload(device="cuda:0")
                    logger.info(f"GPU {gpu_id} {controlnet_type} controlnet pipeline加载完成")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 加载{controlnet_type} controlnet pipeline失败: {e}")
                    # 如果controlnet加载失败，回退到基础pipeline
                    logger.info(f"GPU {gpu_id} 回退到基础pipeline")
                    pipelines[pipeline_key] = pipelines["text2img"]
            
            pipe = pipelines[pipeline_key]
        else:
            raise ValueError(f"不支持的生成模式: {mode}")
        
        # 加载和应用LoRA
        loras = task.get('loras', [])
        if loras:
            if not PEFT_AVAILABLE:
                raise ValueError("PEFT库未安装，无法使用LoRA功能。请安装: pip install peft")
            
            logger.info(f"GPU {gpu_id} 开始加载 {len(loras)} 个LoRA...")
            
            adapter_names = []
            adapter_weights = []
            
            # 添加时间戳和随机数确保adapter_name唯一性
            import time
            import random
            timestamp = int(time.time() * 1000)  # 毫秒时间戳
            random_suffix = random.randint(1000, 9999)  # 4位随机数
            
            for i, lora in enumerate(loras):
                lora_name = lora.get('name')
                lora_weight = lora.get('weight', 1.0)
                
                # 获取LoRA路径
                lora_path = Config.get_lora_path(lora_name)
                if not lora_path:
                    raise ValueError(f"LoRA '{lora_name}' 不存在")
                
                # 生成唯一的adapter_name
                adapter_name = create_safe_adapter_name(lora_name, i, timestamp, random_suffix)
                adapter_names.append(adapter_name)
                adapter_weights.append(lora_weight)
                
                try:
                    logger.info(f"GPU {gpu_id} 加载LoRA: {lora_name} (权重: {lora_weight}) -> adapter: {adapter_name}")
                    
                    # 检查LoRA文件是否存在
                    if not os.path.exists(lora_path):
                        raise ValueError(f"LoRA文件不存在: {lora_path}")
                    
                    # 尝试加载LoRA
                    pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                    logger.info(f"GPU {gpu_id} LoRA {lora_name} 加载成功")
                    
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
            
            # 设置LoRA适配器
            if adapter_names:
                try:
                    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    logger.info(f"GPU {gpu_id} 设置LoRA适配器: {adapter_names} (权重: {adapter_weights})")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 设置LoRA适配器失败: {e}")
                    raise ValueError(f"设置LoRA适配器失败: {str(e)}")
        
        with torch.no_grad():
            if mode == 'text2img':
                result = pipe(
                    prompt=task['prompt'],
                    height=task.get('height', 1024),
                    width=task.get('width', 1024),
                    guidance_scale=task.get('cfg', 3.5),
                    num_inference_steps=task.get('num_inference_steps', 50),
                    max_sequence_length=512,
                    generator=generator
                )
            elif mode == 'img2img':
                # 加载输入图片
                input_image = load_image_func(task.get('input_image'))
                
                # 获取尺寸：优先使用用户指定，否则使用图片原始尺寸
                width, height = get_output_dimensions(input_image, mode)
                
                result = pipe(
                    prompt=task['prompt'],
                    image=input_image,
                    height=height,
                    width=width,
                    strength=task.get('strength', 0.8),
                    guidance_scale=task.get('cfg', 3.5),
                    num_inference_steps=task.get('num_inference_steps', 50),
                    max_sequence_length=512,
                    generator=generator
                )
            elif mode == 'fill':
                # 加载输入图片和蒙版
                input_image = load_image_func(task.get('input_image'))
                mask_image = load_image_func(task.get('mask_image'))
                
                # 获取尺寸：优先使用用户指定，否则使用图片原始尺寸
                width, height = get_output_dimensions(input_image, mode)
                
                # 根据官方示例，使用原始图片尺寸
                result = pipe(
                    prompt=task['prompt'],
                    image=input_image,
                    mask_image=mask_image,
                    height=height,
                    width=width,
                    guidance_scale=task.get('cfg', 30.0),  # Fill模式推荐使用30
                    num_inference_steps=task.get('num_inference_steps', 50),
                    max_sequence_length=512,
                    generator=generator
                )
            elif mode == 'controlnet':
                # 加载控制图片
                control_image = load_image_func(task.get('control_image'))
                controlnet_type = task.get('controlnet_type', 'depth')
                
                # 获取尺寸：优先使用用户指定，否则使用控制图片原始尺寸
                width, height = get_output_dimensions(control_image, mode)
                
                # 如果同时提供了input_image，记录但不使用（ControlNet模式主要使用control_image）
                if task.get('input_image'):
                    logger.info(f"GPU {gpu_id} ControlNet模式检测到input_image，但主要使用control_image")
                
                # 构建ControlNet调用参数 - FluxControlPipeline只支持基本参数
                controlnet_kwargs = {
                    'prompt': task['prompt'],
                    'control_image': control_image,
                    'height': height,
                    'width': width,
                    'guidance_scale': task.get('cfg', 10.0),  # controlnet推荐使用10.0
                    'num_inference_steps': task.get('num_inference_steps', 30),  # controlnet推荐使用30步
                    'max_sequence_length': 512,
                    'generator': generator
                }
                
                # 记录用户提供的ControlNet强度控制参数（但不使用，因为FluxControlPipeline不支持）
                if task.get('controlnet_conditioning_scale') is not None:
                    logger.info(f"GPU {gpu_id} 用户提供了controlnet_conditioning_scale: {task.get('controlnet_conditioning_scale')}，但FluxControlPipeline不支持此参数")
                
                if task.get('control_guidance_start') is not None:
                    logger.info(f"GPU {gpu_id} 用户提供了control_guidance_start: {task.get('control_guidance_start')}，但FluxControlPipeline不支持此参数")
                
                if task.get('control_guidance_end') is not None:
                    logger.info(f"GPU {gpu_id} 用户提供了control_guidance_end: {task.get('control_guidance_end')}，但FluxControlPipeline不支持此参数")
                
                result = pipe(**controlnet_kwargs)
            else:
                raise ValueError(f"不支持的生成模式: {mode}")
        
        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        elapsed_time = time.time() - start_time
        
        # 记录任务完成时的内存状态
        if torch.cuda.is_available():
            final_allocated = torch.cuda.memory_allocated() / 1024**2
            final_cached = torch.cuda.memory_reserved() / 1024**2
            memory_increase = final_allocated - initial_allocated
            logger.info(f"GPU {gpu_id} 任务完成后内存: 已分配 {final_allocated:.1f}MB (+{memory_increase:.1f}MB), 缓存 {final_cached:.1f}MB")
        
        logger.info(f"GPU {gpu_id} {mode}生成成功，耗时: {elapsed_time:.2f}秒")
        
        # 处理保存到磁盘
        save_to_disk = False
        save_path = task.get('save_disk_path')
        if save_path:
            try:
                image.save(save_path)
                save_to_disk = True
                logger.info(f"GPU {gpu_id} 图片已保存到: {save_path}")
            except Exception as e:
                logger.warning(f"GPU {gpu_id} 保存图片失败: {e}")
        
        return {
            "success": True,
            "image_base64": img_base64,
            "elapsed_time": elapsed_time,
            "gpu_id": gpu_id,
            "task_id": task.get('task_id'),
            "save_to_disk": save_to_disk,
            "params": task,
            "mode": mode,
            "controlnet_type": task.get('controlnet_type') if mode == 'controlnet' else None
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"GPU {gpu_id} {mode}生成失败: {str(e)}"
        logger.error(error_msg)
        logger.error(f"生成错误详情: {traceback.format_exc()}")
        
        # 记录错误时的内存状态
        if torch.cuda.is_available():
            error_allocated = torch.cuda.memory_allocated() / 1024**2
            error_cached = torch.cuda.memory_reserved() / 1024**2
            logger.error(f"GPU {gpu_id} 错误时内存: 已分配 {error_allocated:.1f}MB, 缓存 {error_cached:.1f}MB")
        
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
            "gpu_id": gpu_id,
            "task_id": task.get('task_id'),
            "mode": mode,
            "controlnet_type": task.get('controlnet_type') if mode == 'controlnet' else None
        }

class GPUIsolationManager:
    """GPU隔离管理器 - 使用子进程实现真正的GPU隔离"""
    
    def __init__(self):
        self.processes: Dict[str, mp.Process] = {}
        self.result_queues: Dict[str, mp.Queue] = {}
        self.task_queues: Dict[str, mp.Queue] = {}
        self.process_configs: Dict[str, Dict[str, Any]] = {}  # 存储进程配置用于重启
        self.is_running = True
        self.restart_attempts: Dict[str, int] = {}  # 记录重启次数
        self.max_restart_attempts = 3  # 最大重启次数
    
    def create_gpu_process(self, gpu_id: str, model_path: str, model_id: str) -> bool:
        """为指定GPU创建隔离进程"""
        try:
            task_queue = mp.Queue()
            result_queue = mp.Queue()
            process = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, model_path, model_id, task_queue, result_queue),
                name=f"gpu-worker-{gpu_id}"
            )
            process.start()
            process_key = f"{model_id}_{gpu_id}"
            self.processes[process_key] = process
            self.task_queues[process_key] = task_queue
            self.result_queues[process_key] = result_queue
            
            # 保存进程配置用于重启
            self.process_configs[process_key] = {
                "gpu_id": gpu_id,
                "model_path": model_path,
                "model_id": model_id
            }
            
            # 初始化重启计数
            self.restart_attempts[process_key] = 0
            
            logger.info(f"✅ GPU {gpu_id} 隔离进程已创建 (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"❌ 创建GPU {gpu_id} 隔离进程失败: {e}")
            return False
    
    def restart_gpu_process(self, process_key: str) -> bool:
        """重启指定的GPU进程"""
        if process_key not in self.process_configs:
            logger.error(f"无法重启进程 {process_key}：配置不存在")
            return False
        
        # 检查重启次数
        if self.restart_attempts[process_key] >= self.max_restart_attempts:
            logger.error(f"进程 {process_key} 重启次数已达上限 ({self.max_restart_attempts})，停止重启")
            return False
        
        config = self.process_configs[process_key]
        gpu_id = config["gpu_id"]
        
        logger.warning(f"🔄 尝试重启GPU {gpu_id} 进程 (第 {self.restart_attempts[process_key] + 1} 次)")
        
        try:
            # 清理旧进程
            if process_key in self.processes:
                old_process = self.processes[process_key]
                if old_process.is_alive():
                    old_process.terminate()
                    old_process.join(timeout=5.0)
                    if old_process.is_alive():
                        old_process.kill()
            
            # 清理旧队列
            if process_key in self.task_queues:
                del self.task_queues[process_key]
            if process_key in self.result_queues:
                del self.result_queues[process_key]
            
            # 创建新进程
            task_queue = mp.Queue()
            result_queue = mp.Queue()
            process = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, config["model_path"], config["model_id"], task_queue, result_queue),
                name=f"gpu-worker-{gpu_id}-restart"
            )
            process.start()
            
            # 更新进程记录
            self.processes[process_key] = process
            self.task_queues[process_key] = task_queue
            self.result_queues[process_key] = result_queue
            self.restart_attempts[process_key] += 1
            
            logger.info(f"✅ GPU {gpu_id} 进程重启成功 (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 重启GPU {gpu_id} 进程失败: {e}")
            return False
    
    def check_and_restart_dead_processes(self) -> Dict[str, bool]:
        """检查并重启死亡的进程"""
        restart_results = {}
        
        for process_key, process in self.processes.items():
            try:
                if not process.is_alive():
                    logger.warning(f"检测到死亡进程 {process_key} (PID: {process.pid}, exitcode: {process.exitcode})")
                    
                    # 尝试重启
                    success = self.restart_gpu_process(process_key)
                    restart_results[process_key] = success
                    
                    if success:
                        logger.info(f"✅ 进程 {process_key} 重启成功")
                    else:
                        logger.error(f"❌ 进程 {process_key} 重启失败")
                        
            except Exception as e:
                logger.error(f"检查进程 {process_key} 状态时出错: {e}")
                restart_results[process_key] = False
        
        return restart_results
    
    def submit_task(self, gpu_id: str, model_id: str, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提交任务到指定GPU"""
        process_key = f"{model_id}_{gpu_id}"
        
        if process_key not in self.task_queues:
            logger.error(f"GPU {gpu_id} 进程不存在")
            return None
        
        # 检查进程是否还活着
        if process_key not in self.processes:
            logger.error(f"GPU {gpu_id} 进程记录不存在")
            return None
            
        process = self.processes[process_key]
        if not process.is_alive():
            logger.error(f"GPU {gpu_id} 进程已死亡 (PID: {process.pid}, exitcode: {process.exitcode})")
            
            # 尝试重启进程
            restart_success = self.restart_gpu_process(process_key)
            if restart_success:
                logger.info(f"GPU {gpu_id} 进程已重启，重新提交任务")
                # 重新获取进程和队列
                process = self.processes[process_key]
                task_queue = self.task_queues[process_key]
                result_queue = self.result_queues[process_key]
            else:
                return {
                    "success": False,
                    "error": f"GPU进程已死亡且重启失败 (exitcode: {process.exitcode})",
                    "gpu_id": gpu_id
                }
        else:
            task_queue = self.task_queues[process_key]
            result_queue = self.result_queues[process_key]
        
        try:
            # 提交任务
            logger.info(f"提交任务到GPU {gpu_id} (PID: {process.pid}): {task.get('task_id', 'unknown')}")
            task_queue.put(task)
            
            # 等待结果
            result = result_queue.get(timeout=300)  # 5分钟超时
            logger.info(f"GPU {gpu_id} 任务完成: {result.get('success', False)}")
            return result
            
        except queue.Empty:
            error_msg = f"GPU {gpu_id} 任务超时 (5分钟)"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "gpu_id": gpu_id
            }
        except Exception as e:
            error_msg = f"提交任务到GPU {gpu_id} 失败: {str(e)}"
            logger.error(error_msg)
            logger.error(f"错误详情: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "gpu_id": gpu_id
            }
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """获取所有GPU进程状态"""
        status = {}
        
        for process_key, process in self.processes.items():
            try:
                is_alive = process.is_alive()
                status[process_key] = {
                    "pid": process.pid,
                    "alive": is_alive,
                    "exitcode": process.exitcode,
                    "name": process.name,
                    "daemon": process.daemon,
                    "restart_attempts": self.restart_attempts.get(process_key, 0),
                    "max_restart_attempts": self.max_restart_attempts
                }
                
                # 如果进程死亡，记录详细信息
                if not is_alive:
                    logger.warning(f"进程 {process_key} 已死亡 (PID: {process.pid}, exitcode: {process.exitcode})")
                    
            except Exception as e:
                logger.error(f"检查进程 {process_key} 状态时出错: {e}")
                status[process_key] = {
                    "pid": "unknown",
                    "alive": False,
                    "exitcode": "unknown",
                    "error": str(e),
                    "restart_attempts": self.restart_attempts.get(process_key, 0),
                    "max_restart_attempts": self.max_restart_attempts
                }
        
        return status
    
    def shutdown(self):
        """关闭所有GPU进程"""
        logger.info("正在关闭GPU隔离管理器...")
        
        # 发送退出信号
        for process_key, task_queue in self.task_queues.items():
            try:
                task_queue.put(None)  # 退出信号
            except:
                pass
        
        # 等待进程结束
        for process_key, process in self.processes.items():
            try:
                process.join(timeout=10.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
                    if process.is_alive():
                        process.kill()
            except Exception as e:
                logger.warning(f"关闭进程 {process_key} 时出错: {e}")
        
        logger.info("GPU隔离管理器已关闭")

# 使用示例
if __name__ == "__main__":
    # 测试GPU隔离
    manager = GPUIsolationManager()
    
    # 创建GPU进程
    gpu_ids = ["0", "1", "2", "3"]
    from config import Config
    model_paths = Config.get_model_paths()
    model_path = model_paths.get("flux1-dev", "/path/to/flux1-dev")
    
    for gpu_id in gpu_ids:
        manager.create_gpu_process(gpu_id, model_path, "flux1-dev")
    
    # 提交测试任务
    test_task = {
        "task_id": "test_001",
        "prompt": "A beautiful landscape",
        "height": 1024,
        "width": 1024,
        "seed": 42
    }
    
    result = manager.submit_task("0", "flux1-dev", test_task)
    print(f"测试结果: {result}")
    
    # 关闭管理器
    manager.shutdown() 