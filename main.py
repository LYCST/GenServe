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
from fastapi import Depends

# 导入进程级并发管理器
from models.process_concurrent_manager import ProcessConcurrentModelManager
from config import Config
from utils import ValidationUtils, ResponseUtils, GenerateResponse, TaskUtils
from auth import get_api_key, require_generation, require_readonly, require_admin, AuthUtils

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
        base64_content = base64.b64encode(content).decode()
        return base64_content
    except Exception as e:
        logger.error(f"文件转base64失败: {e}")
        logger.error(f"文件信息: filename={file.filename}, content_type={file.content_type}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"文件处理失败: {str(e)}")

@app.get("/")
async def root(key_info: Dict = Depends(get_api_key)):
    """根路径"""
    return {
        "message": "GenServe - 多GPU图片生成服务",
        "version": "2.0.0",
        "status": "running",
        "auth_info": {
            "user": key_info["name"],
            "permissions": key_info["permissions"]
        },
        "endpoints": {
            "json_base64": "POST /generate",
            "form_data": "POST /generate/upload"
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest, key_info: Dict = Depends(require_generation)):
    """生成图片 - JSON格式，支持base64编码的图片"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        logger.info(f"收到请求: {request.model_id} (用户: {key_info['name']})")
        # 获取支持的模型列表
        supported_models_data = concurrent_manager.get_model_list()
        supported_model_ids = [model['model_id'] for model in supported_models_data]
        
        # 使用统一的验证工具
        ValidationUtils.validate_generation_request(
            model_id=request.model_id,
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            mode=request.mode,
            controlnet_type=request.controlnet_type,
            input_image=request.input_image,
            mask_image=request.mask_image,
            control_image=request.control_image,
            loras=request.loras,
            supported_models=supported_model_ids
        )
        logger.info(f"构建任务参数")
        # 构建任务参数
        task_params = TaskUtils.build_task_params(
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
        logger.info(f"调用进程级并发管理器")
        # 调用进程级并发管理器
        result = await concurrent_manager.generate_image_async(
            model_id=request.model_id,
            prompt=request.prompt,
            priority=request.priority,
            **task_params
        )
        logger.info(f"构建响应")
        # 使用统一的响应构建工具
        return ResponseUtils.build_generation_response(
            result=result,
            mode=request.mode,
            controlnet_type=request.controlnet_type
        )
        
    except HTTPException:
        raise
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
    loras: Optional[str] = Form(None),  # JSON格式的LoRA配置
    key_info: Dict = Depends(require_generation)
):
    """生成图片 - 通用Form-data格式，支持所有模式的文件上传"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        logger.info(f"收到文件上传请求: {model_id} (用户: {key_info['name']})")
        # 获取支持的模型列表
        supported_models_data = concurrent_manager.get_model_list()
        supported_model_ids = [model['model_id'] for model in supported_models_data]
        
        # 处理上传的文件
        input_image_base64 = None
        mask_image_base64 = None
        control_image_base64 = None
        
        # 添加文件上传状态调试
        logger.info(f"文件上传状态: input_image={input_image is not None}, mask_image={mask_image is not None}, control_image={control_image is not None}")
        
        if input_image:
            input_image_base64 = await file_to_base64(input_image)
            logger.info(f"input_image处理完成，大小: {len(input_image_base64)}")
        
        if mask_image:
            mask_image_base64 = await file_to_base64(mask_image)
            logger.info(f"mask_image处理完成，大小: {len(mask_image_base64)}")
        
        if control_image:
            control_image_base64 = await file_to_base64(control_image)
            logger.info(f"control_image处理完成，大小: {len(control_image_base64)}")
        
        # 解析LoRA配置
        loras_config = None
        if loras:
            try:
                loras_config = json.loads(loras)
                logger.info(f"LoRA配置解析成功: {loras_config}")
            except json.JSONDecodeError as e:
                logger.error(f"LoRA配置JSON解析失败: {e}")
                raise HTTPException(status_code=400, detail=f"LoRA配置格式错误: {str(e)}")
        
        # 使用统一的验证工具
        ValidationUtils.validate_generation_request(
            model_id=model_id,
            prompt=prompt,
            height=height,
            width=width,
            mode=mode,
            controlnet_type=controlnet_type,
            input_image=input_image_base64,
            mask_image=mask_image_base64,
            control_image=control_image_base64,
            loras=loras_config,
            supported_models=supported_model_ids
        )
        
        # 构建任务参数
        task_params = TaskUtils.build_task_params(
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
            loras=loras_config
        )
        
        # 调用进程级并发管理器
        result = await concurrent_manager.generate_image_async(
            model_id=model_id,
            prompt=prompt,
            priority=priority,
            **task_params
        )
        
        # 使用统一的响应构建工具
        return ResponseUtils.build_generation_response(
            result=result,
            mode=mode,
            controlnet_type=controlnet_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成图片时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status(key_info: Dict = Depends(require_readonly)):
    """获取服务状态"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    status = concurrent_manager.get_status()
    status["auth_info"] = {
        "user": key_info["name"],
        "permissions": key_info["permissions"]
    }
    return status

@app.get("/task/{task_id}")
async def get_task_result(task_id: str, key_info: Dict = Depends(require_readonly)):
    """根据task_id查询任务结果"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = concurrent_manager.get_task_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        
        # 添加认证信息
        if isinstance(result, dict):
            result["auth_info"] = {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务 {task_id} 结果时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models(key_info: Dict = Depends(require_readonly)):
    """获取支持的模型列表"""
    if not concurrent_manager:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    models = concurrent_manager.get_model_list()
    return {
        "models": models,
        "auth_info": {
            "user": key_info["name"],
            "permissions": key_info["permissions"]
        }
    }

@app.get("/loras")
async def get_loras(key_info: Dict = Depends(require_readonly)):
    """获取可用的LoRA列表"""
    try:
        loras = Config.get_lora_list()
        return {
            "success": True,
            "loras": loras,
            "total": len(loras),
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    except Exception as e:
        logger.error(f"获取LoRA列表失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "loras": [],
            "total": 0,
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

@app.get("/health")
async def health_check(key_info: Dict = Depends(get_api_key)):
    """健康检查"""
    if not concurrent_manager:
        return {
            "status": "unhealthy", 
            "reason": "concurrent_manager_not_initialized",
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    
    status = concurrent_manager.get_status()
    if status.get("is_running", False):
        return {
            "status": "healthy",
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    else:
        return {
            "status": "unhealthy", 
            "reason": "concurrent_manager_not_running",
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

@app.get("/auth/keys")
async def get_api_keys(key_info: Dict = Depends(require_admin)):
    """获取API密钥列表（仅管理员）"""
    try:
        keys_info = AuthUtils.list_api_keys()
        return {
            "success": True,
            "keys": keys_info,
            "total": len(keys_info),
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    except Exception as e:
        logger.error(f"获取API密钥列表失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "keys": [],
            "total": 0,
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

@app.post("/auth/generate-key")
async def generate_new_api_key(
    name: str = Form(...),
    permissions: str = Form("generation"),  # 默认只有生成权限
    key_info: Dict = Depends(require_admin)
):
    """生成新的API密钥（仅管理员）"""
    try:
        # 解析权限
        permission_list = [p.strip() for p in permissions.split(",") if p.strip()]
        
        # 生成新密钥
        new_key = AuthUtils.generate_api_key(name, permission_list)
        
        # 动态添加到认证管理器
        from auth import auth_manager
        success = auth_manager.add_api_key(new_key, name, permission_list)
        
        if not success:
            return {
                "success": False,
                "error": "API密钥已存在或添加失败",
                "auth_info": {
                    "user": key_info["name"],
                    "permissions": key_info["permissions"]
                }
            }
        
        # 生成配置字符串
        config_string = f"{new_key}:{name}:{permissions}"
        
        return {
            "success": True,
            "api_key": new_key,
            "name": name,
            "permissions": permission_list,
            "config_string": config_string,
            "message": "API密钥已生成并添加到系统中，重启服务后仍可使用",
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    except Exception as e:
        logger.error(f"生成API密钥失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

@app.post("/auth/delete-key")
async def delete_api_key(
    api_key: str = Form(...),
    key_info: Dict = Depends(require_admin)
):
    """删除API密钥（仅管理员）"""
    try:
        success = AuthUtils.remove_api_key(api_key)
        
        if success:
            return {
                "success": True,
                "message": "API密钥已删除",
                "auth_info": {
                    "user": key_info["name"],
                    "permissions": key_info["permissions"]
                }
            }
        else:
            return {
                "success": False,
                "error": "API密钥不存在或无法删除（环境变量中的密钥无法删除）",
                "auth_info": {
                    "user": key_info["name"],
                    "permissions": key_info["permissions"]
                }
            }
    except Exception as e:
        logger.error(f"删除API密钥失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

@app.post("/auth/delete-key-by-name")
async def delete_api_key_by_name(
    name: str = Form(...),
    key_info: Dict = Depends(require_admin)
):
    """通过用户名删除API密钥（仅管理员）"""
    try:
        result = AuthUtils.remove_api_key_by_name(name)
        
        return {
            "success": result["success"],
            "message": result.get("message", ""),
            "error": result.get("error", ""),
            "deleted_keys": result.get("deleted_keys", []),
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    except Exception as e:
        logger.error(f"通过用户名删除API密钥失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "deleted_keys": [],
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

@app.post("/auth/delete-key-by-id")
async def delete_api_key_by_id(
    key_id: str = Form(...),
    key_info: Dict = Depends(require_admin)
):
    """通过key_id删除API密钥（仅管理员）"""
    try:
        result = AuthUtils.remove_api_key_by_id(key_id)
        
        return {
            "success": result["success"],
            "message": result.get("message", ""),
            "error": result.get("error", ""),
            "deleted_keys": result.get("deleted_keys", []),
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    except Exception as e:
        logger.error(f"通过key_id删除API密钥失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "deleted_keys": [],
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

@app.post("/auth/reload-keys")
async def reload_api_keys(key_info: Dict = Depends(require_admin)):
    """重新加载API密钥（仅管理员）"""
    try:
        AuthUtils.reload_api_keys()
        return {
            "success": True,
            "message": "API密钥已重新加载",
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }
    except Exception as e:
        logger.error(f"重新加载API密钥失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "auth_info": {
                "user": key_info["name"],
                "permissions": key_info["permissions"]
            }
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        log_level=Config.LOG_LEVEL.lower()
    ) 