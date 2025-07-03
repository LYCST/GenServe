import logging
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from models.concurrent_manager import ConcurrentModelManager
from device_manager import DeviceManager
from config import Config
import os

# 设置PyTorch内存管理配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = Config.PYTORCH_CUDA_ALLOC_CONF

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局管理器变量
device_manager = None
concurrent_model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global device_manager, concurrent_model_manager
    
    logger.info("正在启动GenServe并发服务...")
    
    # 初始化管理器
    device_manager = DeviceManager()
    concurrent_model_manager = ConcurrentModelManager()
    
    # 显示配置信息
    config = Config.get_config()
    logger.info(f"服务配置: {config['service']}")
    logger.info(f"设备配置: {config['device']}")
    logger.info(f"模型GPU配置: {config['model_management']['model_gpu_config']}")
    
    # 显示可用设备
    devices = device_manager.get_available_devices()
    logger.info(f"可用设备: {devices}")
    
    # 显示并发管理器状态
    status = concurrent_model_manager.get_status()
    logger.info(f"并发管理器状态: {status}")
    
    logger.info("GenServe并发服务启动完成")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭GenServe并发服务...")
    if concurrent_model_manager:
        concurrent_model_manager.shutdown()
    logger.info("GenServe并发服务已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="GenServe",
    description="基于多种模型的图片生成API服务，支持多GPU并发处理",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
config = Config.get_config()
if config["service"]["enable_cors"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["service"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Pydantic模型
class GenerateRequest(BaseModel):
    prompt: str
    model: str = "flux1-dev"
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    cfg: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None
    save_disk_path: Optional[str] = None

class GenerateResponse(BaseModel):
    message: str
    model: str
    elapsed_time: str
    output: str  # base64字符串
    save_to_disk: bool
    device: str
    task_id: Optional[str] = None
    worker: Optional[str] = None

class DeviceRequest(BaseModel):
    device: str

@app.get("/models")
async def get_models():
    """获取支持的模型列表"""
    return {
        "models": concurrent_model_manager.get_model_list()
    }

@app.get("/health")
async def health_check():
    """健康状态检查"""
    status = concurrent_model_manager.get_status()
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device_config": Config.get_config()["device"],
        "concurrent_status": status
    }

@app.get("/status")
async def get_detailed_status():
    """获取详细状态信息"""
    return {
        "concurrent_manager": concurrent_model_manager.get_status(),
        "device_info": device_manager.get_available_devices(),
        "gpu_load": device_manager.get_gpu_load_info()
    }

@app.get("/config")
async def get_config():
    """获取服务配置"""
    return {
        "service_info": {
            "name": "GenServe",
            "version": "1.0.0",
            "description": "基于多种模型的图片生成API服务，支持多GPU并发处理"
        },
        "config": Config.get_config(),
        "models": concurrent_model_manager.get_model_list()
    }

@app.get("/devices")
async def get_devices():
    """获取可用设备信息"""
    return {
        "available_devices": device_manager.get_available_devices(),
        "device_usage": device_manager.get_device_usage(),
        "gpu_load_info": device_manager.get_gpu_load_info()
    }

@app.get("/gpu/load")
async def get_gpu_load():
    """获取GPU负载信息"""
    return {
        "gpu_load_info": device_manager.get_gpu_load_info(),
        "model_gpu_config": Config.get_config()["model_management"]["model_gpu_config"]
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """生成图片 - 支持并发处理"""
    try:
        # 简化参数处理
        generation_params = {}
        
        # 只更新非None的参数
        for key, value in request.model_dump().items():
            if key not in ['prompt', 'model'] and value is not None:
                generation_params[key] = value
        
        # 使用并发管理器生成图片
        result = await concurrent_model_manager.generate_image_async(
            request.model, request.prompt, **generation_params
        )
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('error', '生成失败'))
        
        return GenerateResponse(
            message="图片生成成功",
            model=request.model,
            elapsed_time=f"{result['elapsed_time']:.2f}s",
            output=result['base64'],
            save_to_disk=result['save_to_disk'],
            device=result.get('device', 'unknown'),
            task_id=result.get('task_id'),
            worker=result.get('worker')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图片生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"图片生成失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    config = Config.get_config()
    uvicorn.run(
        app, 
        host=config["service"]["host"], 
        port=config["service"]["port"]
    ) 