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
    from diffusers import FluxPipeline
    import base64
    import io
    from PIL import Image
    import traceback
    import queue
    import logging
    import time
    import gc
    import psutil

    logger = logging.getLogger(f"gpu_worker_{gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
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
        
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=True
        )
        pipe.enable_model_cpu_offload(device="cuda:0")
        logger.info(f"✅ 模型已加载到GPU {gpu_id}")
        
        task_count = 0
        consecutive_failures = 0  # 连续失败计数
        max_consecutive_failures = 3  # 最大连续失败次数
        last_cleanup_time = time.time()  # 上次清理时间
        
        while True:
            try:
                # 定期内存清理
                current_time = time.time()
                from config import Config
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
                result = process_generation_task(pipe, task, gpu_id)
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
        del pipe
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

def process_generation_task(pipe, task, gpu_id: str):
    import torch
    import io
    import base64
    import time
    import traceback
    logger = logging.getLogger(f"gpu_worker_{gpu_id}")
    start_time = time.time()
    
    # 记录任务开始时的内存状态
    if torch.cuda.is_available():
        initial_allocated = torch.cuda.memory_allocated() / 1024**2
        initial_cached = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU {gpu_id} 任务开始前内存: 已分配 {initial_allocated:.1f}MB, 缓存 {initial_cached:.1f}MB")
    
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
        
        # 记录任务完成时的内存状态
        if torch.cuda.is_available():
            final_allocated = torch.cuda.memory_allocated() / 1024**2
            final_cached = torch.cuda.memory_reserved() / 1024**2
            memory_increase = final_allocated - initial_allocated
            logger.info(f"GPU {gpu_id} 任务完成后内存: 已分配 {final_allocated:.1f}MB (+{memory_increase:.1f}MB), 缓存 {final_cached:.1f}MB")
        
        logger.info(f"GPU {gpu_id} 生成成功，耗时: {elapsed_time:.2f}秒")
        
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
            "params": task
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"GPU {gpu_id} 生成失败: {str(e)}"
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
            "task_id": task.get('task_id')
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