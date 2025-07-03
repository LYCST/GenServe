import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import asyncio
import logging
import signal
import sys
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager
import base64
import io

# 导入进程级并发管理器
from models.process_concurrent_manager import ProcessConcurrentModelManager
from device_manager import DeviceManager
from config import Config

# 配置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
concurrent_manager: Optional[ProcessConcurrentModelManager] = None
device_manager = DeviceManager()

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
    mode: str = "text2img"  # text2img, img2img, fill, controlnet
    strength: float = 0.8  # 用于img2img模式
    input_image: Optional[str] = None  # base64编码的输入图片
    mask_image: Optional[str] = None  # base64编码的蒙版图片（用于fill模式）
    control_image: Optional[str] = None  # base64编码的控制图片（用于controlnet模式）

class GenerateResponse(BaseModel):
    success: bool
    task_id: str = ""
    image_base64: Optional[str] = None
    error: Optional[str] = None
    elapsed_time: Optional[float] = None
    gpu_id: Optional[str] = None
    model_id: Optional[str] = None
    mode: Optional[str] = None  # 生成模式

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
            "form_data": "POST /generate/img2img"
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """生成图片 - JSON格式，支持base64编码的图片"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
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
        elif request.mode == "controlnet" and (not request.input_image or not request.control_image):
            raise HTTPException(status_code=400, detail="controlnet模式需要提供input_image和control_image")
        
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
            control_image=request.control_image
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
            mode=request.mode
        )
        
        return response
        
    except Exception as e:
        logger.error(f"生成图片时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/img2img", response_model=GenerateResponse)
async def generate_image_upload(
    prompt: str = Form(...),
    model_id: str = Form("flux1-dev"),
    height: int = Form(1024),
    width: int = Form(1024),
    num_inference_steps: int = Form(50),
    cfg: float = Form(3.5),
    seed: int = Form(42),
    priority: int = Form(0),
    mode: str = Form("text2img"),
    strength: float = Form(0.8),
    input_image: Optional[UploadFile] = File(None),
    mask_image: Optional[UploadFile] = File(None),
    control_image: Optional[UploadFile] = File(None)
):
    """生成图片 - Form-data格式，支持文件上传"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        # 验证参数
        if not Config.validate_prompt(prompt):
            raise HTTPException(status_code=400, detail="提示词长度超出限制")
        
        if not Config.validate_image_size(height, width):
            raise HTTPException(status_code=400, detail="图片尺寸超出限制")
        
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
        
        # 验证模式特定参数
        if mode == "img2img" and not input_image_base64:
            raise HTTPException(status_code=400, detail="img2img模式需要提供input_image文件")
        elif mode == "fill" and (not input_image_base64 or not mask_image_base64):
            raise HTTPException(status_code=400, detail="fill模式需要提供input_image和mask_image文件")
        elif mode == "controlnet" and (not input_image_base64 or not control_image_base64):
            raise HTTPException(status_code=400, detail="controlnet模式需要提供input_image和control_image文件")
        
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
            control_image=control_image_base64
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
            mode=mode
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
        
        # 获取设备信息
        device_info = device_manager.get_available_devices()
        gpu_load = device_manager.get_device_usage()
        
        return {
            "concurrent_manager": concurrent_status,
            "device_info": device_info,
            "gpu_load": gpu_load
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