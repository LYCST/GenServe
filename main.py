import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import asyncio
import logging
import signal
import sys
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
from contextlib import asynccontextmanager
import base64
import io
import json

# 导入进程级并发管理器
from models.process_concurrent_manager import ProcessConcurrentModelManager
from config import Config

# 配置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
concurrent_manager: Optional[ProcessConcurrentModelManager] = None

# 请求模型 - 用于JSON请求
class GenerateRequest(BaseModel):
    prompt: str
    model_id: str = "flux1-dev"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    cfg: float = 3.5
    seed: int = 42
    priority: int = 0
    mode: str = "text2img"  # text2img, img2img, fill, controlnet, redux
    strength: float = 0.8  # 用于img2img模式
    input_image: Optional[str] = None  # base64编码的输入图片
    mask_image: Optional[str] = None  # base64编码的蒙版图片（用于fill模式）
    control_image: Optional[str] = None  # base64编码的控制图片（用于controlnet模式）
    controlnet_type: str = "depth"  # controlnet类型：depth, canny, openpose
    controlnet_conditioning_scale: Optional[float] = None  # ControlNet条件强度，控制深度图影响程度
    control_guidance_start: Optional[float] = None  # ControlNet开始作用点（0-1），控制何时开始应用深度图
    control_guidance_end: Optional[float] = None  # ControlNet结束作用点（0-1），控制何时停止应用深度图
    loras: Optional[List[Dict[str, Any]]] = None  # LoRA配置列表

class GenerateResponse(BaseModel):
    success: bool
    task_id: str
    image_base64: Optional[str] = None
    error: Optional[str] = None
    elapsed_time: Optional[float] = None
    gpu_id: Optional[str] = None
    model_id: Optional[str] = None
    mode: Optional[str] = None
    controlnet_type: Optional[str] = None  # 添加controlnet类型到响应

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global concurrent_manager
    
    # 启动时初始化
    logger.info("🚀 启动GenServe服务...")
    
    try:
        # 初始化进程级并发管理器
        concurrent_manager = ProcessConcurrentModelManager()
        logger.info("✅ 进程级并发管理器初始化完成")
        
        # 打印配置摘要
        Config.print_config_summary()
        
        yield
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    finally:
        # 关闭时清理
        logger.info("🛑 正在关闭GenServe服务...")
        if concurrent_manager:
            concurrent_manager.shutdown()
        logger.info("✅ 服务已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="GenServe - 多GPU图片生成服务",
    description="基于Flux模型的多GPU并发图片生成服务，支持JSON(base64)和Form-data(文件上传)两种图片上传方式",
    version="2.0.0",
    lifespan=lifespan
)

# 信号处理
def signal_handler(signum, frame):
    """处理退出信号"""
    logger.info(f"收到信号 {signum}，正在优雅关闭...")
    if concurrent_manager:
        concurrent_manager.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def file_to_base64(file: UploadFile) -> str:
    """将上传的文件转换为base64"""
    try:
        content = await file.read()
        return base64.b64encode(content).decode()
    except Exception as e:
        logger.error(f"文件转base64失败: {e}")
        raise HTTPException(status_code=400, detail=f"文件处理失败: {str(e)}")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "GenServe - 多GPU图片生成服务",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "json_base64": "POST /generate",
            "form_data": "POST /generate/upload"
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """生成图片 - JSON格式，支持base64编码的图片"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        # 验证模型是否支持
        supported_models_data = concurrent_manager.get_model_list()
        supported_model_ids = [model['model_id'] for model in supported_models_data]
        if request.model_id not in supported_model_ids:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的模型: {request.model_id}，支持的模型: {supported_model_ids}"
            )
        
        # 验证参数
        if not Config.validate_prompt(request.prompt):
            raise HTTPException(status_code=400, detail="提示词长度超出限制")
        
        if not Config.validate_image_size(request.height, request.width):
            raise HTTPException(status_code=400, detail="图片尺寸超出限制")
        
        # 验证模式特定参数
        if request.mode == "img2img" and not request.input_image:
            raise HTTPException(status_code=400, detail="img2img模式需要提供input_image")
        elif request.mode == "fill" and (not request.input_image or not request.mask_image):
            raise HTTPException(status_code=400, detail="fill模式需要提供input_image和mask_image")
        elif request.mode == "controlnet" and not request.control_image:
            raise HTTPException(status_code=400, detail="controlnet模式需要提供control_image")
        elif request.mode == "redux" and not request.input_image:
            raise HTTPException(status_code=400, detail="redux模式需要提供input_image")
        
        # 验证controlnet类型
        if request.mode == "controlnet":
            valid_controlnet_types = ["depth", "canny", "openpose"]
            if request.controlnet_type.lower() not in valid_controlnet_types:
                raise HTTPException(status_code=400, detail=f"不支持的controlnet类型: {request.controlnet_type}，支持的类型: {valid_controlnet_types}")
        
        # 验证LoRA参数
        if request.loras:
            for lora in request.loras:
                if not isinstance(lora, dict):
                    raise HTTPException(status_code=400, detail="每个LoRA必须是字典格式")
                if 'name' not in lora:
                    raise HTTPException(status_code=400, detail="每个LoRA必须包含name字段")
                if 'weight' not in lora:
                    raise HTTPException(status_code=400, detail="每个LoRA必须包含weight字段")
                
                # 验证LoRA是否存在
                lora_path = Config.get_lora_path(lora['name'])
                if not lora_path:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"LoRA '{lora['name']}' 不存在，请检查 /loras 接口获取可用LoRA列表"
                    )
                
                # 验证权重范围
                weight = lora.get('weight', 1.0)
                if not isinstance(weight, (int, float)) or weight < 0 or weight > 2:
                    raise HTTPException(status_code=400, detail="LoRA权重必须在0-2之间")
        
        # 调用进程级并发管理器
        result = await concurrent_manager.generate_image_async(
            model_id=request.model_id,
            prompt=request.prompt,
            priority=request.priority,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            cfg=request.cfg,
            seed=request.seed,
            mode=request.mode,
            strength=request.strength,
            input_image=request.input_image,
            mask_image=request.mask_image,
            control_image=request.control_image,
            controlnet_type=request.controlnet_type,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            control_guidance_start=request.control_guidance_start,
            control_guidance_end=request.control_guidance_end,
            loras=request.loras
        )
        
        # 构建响应
        response = GenerateResponse(
            success=result.get("success", False),
            task_id=result.get("task_id", ""),
            image_base64=result.get("image_base64"),
            error=result.get("error"),
            elapsed_time=result.get("elapsed_time"),
            gpu_id=result.get("gpu_id"),
            model_id=result.get("model_id"),
            mode=request.mode,
            controlnet_type=request.controlnet_type
        )
        
        return response
        
    except Exception as e:
        logger.error(f"生成图片时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/upload", response_model=GenerateResponse)
async def generate_image_upload_general(
    prompt: str = Form(...),
    model_id: str = Form("flux1-dev"),
    height: int = Form(1024),
    width: int = Form(1024),
    num_inference_steps: int = Form(50),
    cfg: float = Form(3.5),
    seed: int = Form(42),
    priority: int = Form(0),
    mode: str = Form("text2img"),  # text2img, img2img, fill, controlnet, redux
    strength: float = Form(0.8),
    input_image: Optional[UploadFile] = File(None),
    mask_image: Optional[UploadFile] = File(None),
    control_image: Optional[UploadFile] = File(None),
    controlnet_type: str = Form("depth"),
    controlnet_conditioning_scale: Optional[float] = Form(None),  # ControlNet条件强度
    control_guidance_start: Optional[float] = Form(None),  # ControlNet开始作用点
    control_guidance_end: Optional[float] = Form(None),  # ControlNet结束作用点
    loras: Optional[str] = Form(None)  # JSON格式的LoRA配置
):
    """生成图片 - 通用Form-data格式，支持所有模式的文件上传"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        # 验证模型是否支持
        supported_models_data = concurrent_manager.get_model_list()
        supported_model_ids = [model['model_id'] for model in supported_models_data]
        if model_id not in supported_model_ids:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的模型: {model_id}，支持的模型: {supported_model_ids}"
            )
        
        # 验证参数
        if not Config.validate_prompt(prompt):
            raise HTTPException(status_code=400, detail="提示词长度超出限制")
        
        if not Config.validate_image_size(height, width):
            raise HTTPException(status_code=400, detail="图片尺寸超出限制")
        
        # 验证controlnet类型
        if mode == "controlnet":
            valid_controlnet_types = ["depth", "canny", "openpose"]
            if controlnet_type.lower() not in valid_controlnet_types:
                raise HTTPException(status_code=400, detail=f"不支持的controlnet类型: {controlnet_type}，支持的类型: {valid_controlnet_types}")
        
        # 处理上传的文件
        input_image_base64 = None
        mask_image_base64 = None
        control_image_base64 = None
        
        if input_image:
            input_image_base64 = await file_to_base64(input_image)
        
        if mask_image:
            mask_image_base64 = await file_to_base64(mask_image)
        
        if control_image:
            control_image_base64 = await file_to_base64(control_image)
        
        # 处理LoRA参数
        loras_list = None
        if loras:
            try:
                loras_list = json.loads(loras)
                # 验证LoRA格式
                if not isinstance(loras_list, list):
                    raise ValueError("loras参数必须是列表格式")
                
                for lora in loras_list:
                    if not isinstance(lora, dict):
                        raise ValueError("每个LoRA必须是字典格式")
                    if 'name' not in lora:
                        raise ValueError("每个LoRA必须包含name字段")
                    if 'weight' not in lora:
                        raise ValueError("每个LoRA必须包含weight字段")
                    
                    # 验证LoRA是否存在
                    lora_path = Config.get_lora_path(lora['name'])
                    if not lora_path:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"LoRA '{lora['name']}' 不存在，请检查 /loras 接口获取可用LoRA列表"
                        )
                    
                    # 验证权重范围
                    weight = lora.get('weight', 1.0)
                    if not isinstance(weight, (int, float)) or weight < 0 or weight > 2:
                        raise ValueError("LoRA权重必须在0-2之间")
                        
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="loras参数必须是有效的JSON格式")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # 验证模式特定参数
        if mode == "img2img" and not input_image_base64:
            raise HTTPException(status_code=400, detail="img2img模式需要提供input_image文件")
        elif mode == "fill" and (not input_image_base64 or not mask_image_base64):
            raise HTTPException(status_code=400, detail="fill模式需要提供input_image和mask_image文件")
        elif mode == "controlnet" and not control_image_base64:
            raise HTTPException(status_code=400, detail="controlnet模式需要提供control_image文件")
        elif mode == "redux" and not input_image_base64:
            raise HTTPException(status_code=400, detail="redux模式需要提供input_image文件")
        
        # 调用进程级并发管理器
        result = await concurrent_manager.generate_image_async(
            model_id=model_id,
            prompt=prompt,
            priority=priority,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg=cfg,
            seed=seed,
            mode=mode,
            strength=strength,
            input_image=input_image_base64,
            mask_image=mask_image_base64,
            control_image=control_image_base64,
            controlnet_type=controlnet_type,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            loras=loras_list
        )
        
        # 构建响应
        response = GenerateResponse(
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
        
        return response
        
    except Exception as e:
        logger.error(f"生成图片时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """获取服务状态 - 进程级版本"""
    if not concurrent_manager:
        return {"error": "服务未就绪"}
    
    try:
        # 获取并发管理器状态
        concurrent_status = concurrent_manager.get_status()
        
        return {
            "concurrent_manager": concurrent_status
        }
        
    except Exception as e:
        logger.error(f"获取状态时出错: {e}")
        return {"error": str(e)}

@app.get("/models")
async def get_models():
    """获取模型列表 - 进程级版本"""
    if not concurrent_manager:
        return {"error": "服务未就绪"}
    
    try:
        models = concurrent_manager.get_model_list()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"获取模型列表时出错: {e}")
        return {"error": str(e)}

@app.get("/loras")
async def get_loras():
    """获取LoRA模型列表"""
    try:
        loras = Config.get_lora_list()
        return {
            "success": True,
            "loras": loras,
            "total_count": len(loras),
            "base_path": Config.LORA_BASE_PATH
        }
        
    except Exception as e:
        logger.error(f"获取LoRA列表时出错: {e}")
        return {
            "success": False,
            "error": str(e),
            "loras": [],
            "total_count": 0,
            "base_path": Config.LORA_BASE_PATH
        }

@app.get("/health")
async def health_check():
    """健康检查"""
    if not concurrent_manager:
        return {"status": "unhealthy", "error": "并发管理器未初始化"}
    
    try:
        status = concurrent_manager.get_status()
        alive_processes = status.get("alive_processes", 0)
        total_processes = status.get("total_processes", 0)
        
        if alive_processes == total_processes and total_processes > 0:
            return {"status": "healthy", "alive_processes": alive_processes}
        else:
            return {
                "status": "degraded", 
                "alive_processes": alive_processes,
                "total_processes": total_processes
            }
            
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        log_level=Config.LOG_LEVEL.lower()
    ) 