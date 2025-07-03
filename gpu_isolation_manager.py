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

# mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)

def gpu_worker_process(gpu_id: str, model_path: str, model_id: str, task_queue, result_queue):
    """GPU工作进程 - 在隔离环境中运行（顶层函数，支持spawn）"""
    import torch
    from diffusers import FluxPipeline
    import base64
    import io
    from PIL import Image
    import traceback
    import queue
    import logging
    import time

    logger = logging.getLogger(f"gpu_worker_{gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info(f"🚀 GPU {gpu_id} 工作进程启动 (PID: {os.getpid()})")
    try:
        logger.info(f"正在加载模型到GPU {gpu_id}...")
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=True
        )
        pipe.enable_model_cpu_offload(device="cuda:0")
        logger.info(f"✅ 模型已加载到GPU {gpu_id}")
        while True:
            try:
                task = task_queue.get(timeout=1.0)
                if task is None:
                    logger.info(f"GPU {gpu_id} 收到退出信号")
                    break
                logger.info(f"GPU {gpu_id} 开始处理任务: {task.get('task_id', 'unknown')}")
                result = process_generation_task(pipe, task, gpu_id)
                result_queue.put(result)
                logger.info(f"GPU {gpu_id} 任务处理完成: {result.get('success', False)}")
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"GPU {gpu_id} 处理任务时出错: {str(e)}"
                logger.error(error_msg)
                logger.error(f"错误详情: {traceback.format_exc()}")
                try:
                    result_queue.put({
                        "success": False,
                        "error": str(e),
                        "gpu_id": gpu_id,
                        "task_id": task.get('task_id') if 'task' in locals() else 'unknown'
                    })
                except Exception as put_error:
                    logger.error(f"GPU {gpu_id} 无法返回错误结果: {put_error}")
        logger.info(f"GPU {gpu_id} 开始清理资源...")
        del pipe
        torch.cuda.empty_cache()
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

def process_generation_task(pipe, task, gpu_id: str):
    import torch
    import io
    import base64
    import time
    import traceback
    logger = logging.getLogger(f"gpu_worker_{gpu_id}")
    start_time = time.time()
    try:
        logger.info(f"GPU {gpu_id} 开始生成任务: {task.get('task_id', 'unknown')}")
        generator = torch.Generator("cpu").manual_seed(task.get('seed', 42))
        with torch.no_grad():
            result = pipe(
                prompt=task['prompt'],
                height=task.get('height', 1024),
                width=task.get('width', 1024),
                guidance_scale=task.get('cfg', 3.5),
                num_inference_steps=task.get('num_inference_steps', 50),
                max_sequence_length=512,
                generator=generator
            )
        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        elapsed_time = time.time() - start_time
        logger.info(f"GPU {gpu_id} 生成成功，耗时: {elapsed_time:.2f}秒")
        return {
            "success": True,
            "image_base64": img_base64,
            "elapsed_time": elapsed_time,
            "gpu_id": gpu_id,
            "task_id": task.get('task_id'),
            "params": task
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"GPU {gpu_id} 生成失败: {str(e)}"
        logger.error(error_msg)
        logger.error(f"生成错误详情: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
            "gpu_id": gpu_id,
            "task_id": task.get('task_id')
        }

class GPUIsolationManager:
    """GPU隔离管理器 - 使用子进程实现真正的GPU隔离"""
    
    def __init__(self):
        self.processes: Dict[str, mp.Process] = {}
        self.result_queues: Dict[str, mp.Queue] = {}
        self.task_queues: Dict[str, mp.Queue] = {}
        self.is_running = True
    
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
            logger.info(f"✅ GPU {gpu_id} 隔离进程已创建 (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"❌ 创建GPU {gpu_id} 隔离进程失败: {e}")
            return False
    
    def submit_task(self, gpu_id: str, model_id: str, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提交任务到指定GPU"""
        process_key = f"{model_id}_{gpu_id}"
        
        if process_key not in self.task_queues:
            logger.error(f"GPU {gpu_id} 进程不存在")
            return None
        
        try:
            # 提交任务
            self.task_queues[process_key].put(task)
            
            # 等待结果
            result = self.result_queues[process_key].get(timeout=300)  # 5分钟超时
            return result
            
        except Exception as e:
            logger.error(f"提交任务到GPU {gpu_id} 失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "gpu_id": gpu_id
            }
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """获取所有GPU进程状态"""
        status = {}
        
        for process_key, process in self.processes.items():
            status[process_key] = {
                "pid": process.pid,
                "alive": process.is_alive(),
                "exitcode": process.exitcode
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
    model_path = "/home/shuzuan/prj/models/flux1-dev"
    
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