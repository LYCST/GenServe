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
    import logging
    import traceback
    import queue
    from models.flux_model import FluxModel
    import time
    import os
    import gc

    logger = logging.getLogger(f"gpu_worker_{gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    model = None
    try:
        # 在GPU工作进程中，由于设置了CUDA_VISIBLE_DEVICES，只能看到GPU 0
        # 所以使用cuda:0而不是cuda:{gpu_id}
        model = FluxModel(model_id=model_id, gpu_device="cuda:0", physical_gpu_id=gpu_id)
        if not model.load():
            logger.error(f"模型加载失败: {model_id} on GPU {gpu_id}")
            result_queue.put({"success": False, "error": f"模型加载失败: {model_id}", "gpu_id": gpu_id})
            return
        logger.info(f"模型 {model_id} 已加载到GPU {gpu_id}")
        while True:
            try:
                task = task_queue.get(timeout=1.0)
                if task is None:
                    logger.info(f"GPU {gpu_id} 收到退出信号")
                    break
                
                task_id = task.get('task_id', 'unknown')
                logger.info(f"🎯 GPU {gpu_id} 进程接收到任务: {task_id[:8]}")
                logger.info(f"📋 任务详情: 提示词='{task.get('prompt', '')[:50]}{'...' if len(task.get('prompt', '')) > 50 else ''}', 模式={task.get('mode', 'unknown')}")
                
                try:
                    logger.info(f"🚀 GPU {gpu_id} 开始处理任务: {task_id[:8]}")
                    result = model.generate(**task)
                    result['task_id'] = task_id
                    result['gpu_id'] = gpu_id
                    logger.info(f"✅ GPU {gpu_id} 任务 {task_id[:8]} 处理完成，准备发送结果")
                    result_queue.put(result)
                    logger.info(f"📤 GPU {gpu_id} 任务 {task_id[:8]} 结果已发送到结果队列")
                except Exception as e:
                    logger.error(f"❌ GPU {gpu_id} 模型推理异常: {e}")
                    logger.error(traceback.format_exc())
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "task_id": task_id,
                        "gpu_id": gpu_id
                    }
                    result_queue.put(error_result)
                    logger.error(f"📤 GPU {gpu_id} 任务 {task_id[:8]} 错误结果已发送")
                
                # 推理后可选清理
                gc.collect()
                logger.info(f"🧹 GPU {gpu_id} 任务 {task_id[:8]} 内存清理完成")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ GPU {gpu_id} 进程异常: {e}")
                logger.error(traceback.format_exc())
                break
    finally:
        if model is not None:
            model.unload()
        logger.info(f"GPU {gpu_id} 工作进程退出")

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
    
    def submit_task(self, gpu_id: str, model_id: str, task: Dict[str, Any]) -> bool:
        """提交任务到指定GPU - 异步提交，不等待结果"""
        process_key = f"{model_id}_{gpu_id}"
        
        if process_key not in self.task_queues:
            logger.error(f"GPU {gpu_id} 进程不存在")
            return False
        
        # 检查进程是否还活着
        if process_key not in self.processes:
            logger.error(f"GPU {gpu_id} 进程记录不存在")
            return False
            
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
            else:
                return False
        else:
            task_queue = self.task_queues[process_key]
        
        try:
            # 异步提交任务，不等待结果
            task_id = task.get('task_id', 'unknown')
            logger.info(f"📤 异步提交任务到GPU {gpu_id} (PID: {process.pid}): {task_id[:8]}")
            logger.info(f"📋 任务详情: 提示词='{task.get('prompt', '')[:50]}{'...' if len(task.get('prompt', '')) > 50 else ''}', 模式={task.get('mode', 'unknown')}")
            
            task_queue.put(task, block=False)  # 非阻塞提交
            logger.info(f"✅ 任务 {task_id[:8]} 已成功提交到GPU {gpu_id} 任务队列")
            return True
            
        except queue.Full:
            error_msg = f"GPU {gpu_id} 任务队列已满"
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"提交任务到GPU {gpu_id} 失败: {str(e)}"
            logger.error(error_msg)
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    def get_task_result(self, gpu_id: str, model_id: str, timeout: float = 300) -> Optional[Dict[str, Any]]:
        """从指定GPU获取任务结果 - 可选使用"""
        process_key = f"{model_id}_{gpu_id}"
        
        if process_key not in self.result_queues:
            logger.error(f"GPU {gpu_id} 结果队列不存在")
            return None
        
        result_queue = self.result_queues[process_key]
        
        try:
            # 等待结果
            result = result_queue.get(timeout=timeout)
            logger.info(f"GPU {gpu_id} 获取到结果: {result.get('success', False)}")
            return result
            
        except queue.Empty:
            error_msg = f"GPU {gpu_id} 获取结果超时 ({timeout}秒)"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "gpu_id": gpu_id
            }
        except Exception as e:
            error_msg = f"从GPU {gpu_id} 获取结果失败: {str(e)}"
            logger.error(error_msg)
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