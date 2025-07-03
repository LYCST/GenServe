import asyncio
import threading
import time
import os
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple
import logging
from queue import Queue, Empty, PriorityQueue
from dataclasses import dataclass, field
import uuid
import torch
from gpu_isolation_manager import GPUIsolationManager
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class GenerationTask:
    """生成任务"""
    task_id: str
    model_id: str
    prompt: str
    params: Dict[str, Any]
    result_queue: Queue
    created_at: float
    priority: int = 0  # 优先级，数字越小优先级越高
    
    def __lt__(self, other):
        """支持优先级队列排序"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at

class ProcessConcurrentModelManager:
    """进程级并发模型管理器 - 使用子进程实现真正的GPU隔离"""
    
    def __init__(self):
        self.gpu_manager = GPUIsolationManager()
        self.model_instances: Dict[str, List[str]] = {}  # model_id -> [gpu_ids]
        
        # 全局任务队列和调度
        self.global_task_queue = PriorityQueue()
        self.task_results: Dict[str, Queue] = {}  # task_id -> result_queue
        
        # 工作线程管理
        self.scheduler_thread = None
        self.is_running = False
        
        # 负载均衡计数器
        self.gpu_round_robin_counters: Dict[str, int] = {}  # model_id -> counter
        
        # 从配置获取参数
        config = Config.get_config()
        self.max_global_queue_size = config["concurrent"]["max_global_queue_size"]
        self.task_timeout = config["concurrent"]["task_timeout"]
        self.scheduler_sleep_time = config["concurrent"]["scheduler_sleep_time"]
        self.load_balance_strategy = config["concurrent"]["load_balance_strategy"]
        
        # 统计信息
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "queue_full_rejections": 0
        }
        
        self._initialize_models()
        self._start_manager()
    
    def _initialize_models(self):
        """初始化模型进程"""
        model_gpu_config = Config.get_config()["model_management"]["model_gpu_config"]
        model_paths = Config.get_config()["model_management"]["model_paths"]
        
        for model_id, gpu_list in model_gpu_config.items():
            if model_id not in self.model_instances:
                self.model_instances[model_id] = []
            
            model_path = model_paths.get(model_id, "")
            if not model_path or not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                continue
            
            # 为每个GPU创建隔离进程
            for gpu_device in gpu_list:
                if gpu_device.startswith("cuda:"):
                    gpu_id = gpu_device.split(":")[1]
                    
                    # 创建GPU隔离进程
                    if self.gpu_manager.create_gpu_process(gpu_id, model_path, model_id):
                        self.model_instances[model_id].append(gpu_id)
                        logger.info(f"✅ GPU {gpu_id} 进程创建成功")
                    else:
                        logger.error(f"❌ GPU {gpu_id} 进程创建失败")
        
        # 打印初始化结果
        total_processes = sum(len(gpus) for gpus in self.model_instances.values())
        logger.info(f"🎉 模型进程初始化完成，总共创建了 {total_processes} 个GPU进程")
        for model_id, gpus in self.model_instances.items():
            logger.info(f"  {model_id}: {len(gpus)} 个GPU进程 ({gpus})")
    
    def _start_manager(self):
        """启动管理器"""
        self.is_running = True
        
        # 启动全局调度器
        self.scheduler_thread = threading.Thread(
            target=self._global_scheduler_loop, 
            name="process-scheduler"
        )
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("进程级全局调度器已启动")
        
        # 启动进程监控线程
        self.monitor_thread = threading.Thread(
            target=self._process_monitor_loop,
            name="process-monitor"
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("进程监控线程已启动")
    
    def _process_monitor_loop(self):
        """进程监控循环 - 定期检查死亡进程并重启"""
        logger.info("进程监控线程开始运行")
        
        while self.is_running:
            try:
                # 每30秒检查一次进程状态
                time.sleep(30)
                
                # 检查并重启死亡进程
                restart_results = self.gpu_manager.check_and_restart_dead_processes()
                
                if restart_results:
                    logger.info(f"进程监控: 检查了 {len(restart_results)} 个进程")
                    for process_key, success in restart_results.items():
                        if success:
                            logger.info(f"✅ 进程 {process_key} 重启成功")
                        else:
                            logger.warning(f"⚠️ 进程 {process_key} 重启失败")
                
            except Exception as e:
                logger.error(f"进程监控循环错误: {e}")
    
    def _global_scheduler_loop(self):
        """全局调度器循环 - 智能任务分配"""
        logger.info("进程级全局调度器开始运行")
        
        while self.is_running:
            try:
                # 获取全局任务（带超时）
                task = self.global_task_queue.get(timeout=self.scheduler_sleep_time)
                
                # 找到最佳的GPU进程
                best_gpu_id = self._find_best_gpu(task.model_id)
                
                if best_gpu_id:
                    # 异步分配任务给GPU进程，不等待完成
                    self._submit_task_async(task, best_gpu_id)
                    logger.info(f"任务 {task.task_id} 已分配给GPU {best_gpu_id}")
                else:
                    # 所有GPU都忙碌，重新放回队列
                    self.global_task_queue.put(task)
                    # 更长的等待时间，避免CPU占用过高
                    time.sleep(self.scheduler_sleep_time * 5)
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"全局调度器错误: {e}")
    
    def _find_best_gpu(self, model_id: str) -> Optional[str]:
        """找到最佳的GPU进程 - 实现轮询负载均衡"""
        if model_id not in self.model_instances:
            logger.warning(f"⚠️ 模型 {model_id} 不存在")
            return None
        
        available_gpus = self.model_instances[model_id]
        
        if not available_gpus:
            logger.debug(f"⚠️ 模型 {model_id} 没有可用GPU")
            return None
        
        # 初始化轮询计数器
        if model_id not in self.gpu_round_robin_counters:
            self.gpu_round_robin_counters[model_id] = 0
        
        # 轮询策略：依次选择下一个GPU
        gpu_index = self.gpu_round_robin_counters[model_id] % len(available_gpus)
        selected_gpu = available_gpus[gpu_index]
        
        # 更新计数器
        self.gpu_round_robin_counters[model_id] += 1
        
        logger.debug(f"负载均衡: 模型 {model_id} 选择GPU {selected_gpu} (索引: {gpu_index}/{len(available_gpus)})")
        return selected_gpu
    
    def _submit_task_async(self, task: GenerationTask, gpu_id: str):
        """异步提交任务到GPU进程"""
        # 创建后台线程处理任务，不阻塞调度器
        thread = threading.Thread(
            target=self._process_task_on_gpu_process,
            args=(task, gpu_id),
            name=f"task-{task.task_id[:8]}"
        )
        thread.daemon = True
        thread.start()
    
    def _process_task_on_gpu_process(self, task: GenerationTask, gpu_id: str):
        """在GPU进程中处理任务"""
        logger.info(f"开始处理任务 {task.task_id[:8]} (GPU: {gpu_id})")
        
        try:
            # 准备任务数据
            task_data = {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "height": task.params.get('height', 1024),
                "width": task.params.get('width', 1024),
                "cfg": task.params.get('cfg', 3.5),
                "num_inference_steps": task.params.get('num_inference_steps', 50),
                "seed": task.params.get('seed', 42),
                "mode": task.params.get('mode', 'text2img'),
                "strength": task.params.get('strength', 0.8),
                "input_image": task.params.get('input_image'),
                "mask_image": task.params.get('mask_image'),
                "control_image": task.params.get('control_image')
            }
            
            # 提交任务到GPU进程
            result = self.gpu_manager.submit_task(gpu_id, task.model_id, task_data)
            
            if result:
                # 添加任务信息
                result.update({
                    "task_id": task.task_id,
                    "gpu_id": gpu_id,
                    "model_id": task.model_id,
                    "thread_name": threading.current_thread().name
                })
                
                # 发送结果
                task.result_queue.put(result)
                
                # 更新统计
                if result.get("success", False):
                    self.stats["completed_tasks"] += 1
                    logger.info(f"✅ 任务 {task.task_id[:8]} 完成 (GPU: {gpu_id})")
                else:
                    self.stats["failed_tasks"] += 1
                    logger.error(f"❌ 任务 {task.task_id[:8]} 失败: {result.get('error', '未知错误')}")
            else:
                # 任务提交失败
                error_result = {
                    "success": False,
                    "error": "GPU进程不可用",
                    "task_id": task.task_id,
                    "gpu_id": gpu_id,
                    "model_id": task.model_id
                }
                task.result_queue.put(error_result)
                self.stats["failed_tasks"] += 1
                
        except Exception as e:
            logger.error(f"❌ 处理任务 {task.task_id[:8]} 时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_result = {
                "success": False,
                "error": str(e),
                "task_id": task.task_id,
                "gpu_id": gpu_id,
                "model_id": task.model_id
            }
            task.result_queue.put(error_result)
            self.stats["failed_tasks"] += 1
    
    async def generate_image_async(
        self, 
        model_id: str, 
        prompt: str, 
        priority: int = 0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        cfg: float = 3.5,
        seed: int = 42,
        mode: str = "text2img",
        strength: float = 0.8,
        input_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        control_image: Optional[str] = None
    ) -> Dict[str, Any]:
        """异步生成图片 - 支持图生图"""
        if not self.is_running:
            return {
                "success": False,
                "error": "并发管理器未运行",
                "task_id": "",
                "gpu_id": None,
                "model_id": model_id
            }
        
        # 验证模型是否支持
        if model_id not in self.model_instances:
            return {
                "success": False,
                "error": f"模型 {model_id} 未加载",
                "task_id": "",
                "gpu_id": None,
                "model_id": model_id
            }
        
        # 创建任务
        task_id = str(uuid.uuid4())
        result_queue = Queue()
        
        # 构建任务参数
        task_params = {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "cfg": cfg,
            "seed": seed,
            "mode": mode,
            "strength": strength,
            "input_image": input_image,
            "mask_image": mask_image,
            "control_image": control_image
        }
        
        task = GenerationTask(
            task_id=task_id,
            model_id=model_id,
            prompt=prompt,
            params=task_params,
            result_queue=result_queue,
            created_at=time.time(),
            priority=priority
        )
        
        logger.info(f"提交{mode}任务: {task_id}, 模型: {model_id}, 优先级: {priority}")
        
        # 提交到全局队列
        try:
            self.global_task_queue.put(task)
            logger.info(f"任务 {task_id} 已提交到全局队列")
        except Exception as e:
            logger.error(f"提交任务 {task_id} 失败: {e}")
            return {
                "success": False,
                "error": f"提交任务失败: {str(e)}",
                "task_id": task_id,
                "gpu_id": None,
                "model_id": model_id
            }
        
        # 等待结果
        try:
            result = result_queue.get(timeout=Config.TASK_TIMEOUT)
            logger.info(f"任务 {task_id} 完成: {result.get('success', False)}")
            
            # 添加任务ID到结果
            result["task_id"] = task_id
            result["model_id"] = model_id
            result["mode"] = mode
            
            return result
            
        except Empty:
            logger.error(f"任务 {task_id} 超时")
            return {
                "success": False,
                "error": f"任务超时 ({Config.TASK_TIMEOUT}秒)",
                "task_id": task_id,
                "gpu_id": None,
                "model_id": model_id
            }
        except Exception as e:
            logger.error(f"等待任务 {task_id} 结果时出错: {e}")
            return {
                "success": False,
                "error": f"等待结果失败: {str(e)}",
                "task_id": task_id,
                "gpu_id": None,
                "model_id": model_id
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取详细状态"""
        # 获取GPU进程状态
        gpu_status = self.gpu_manager.get_gpu_status()
        
        # 计算总体统计
        total_processes = sum(len(gpus) for gpus in self.model_instances.values())
        alive_processes = sum(1 for status in gpu_status.values() if status["alive"])
        dead_processes = total_processes - alive_processes
        
        # 统计重启信息
        total_restarts = sum(status.get("restart_attempts", 0) for status in gpu_status.values())
        max_restarts_reached = sum(1 for status in gpu_status.values() 
                                 if status.get("restart_attempts", 0) >= status.get("max_restart_attempts", 3))
        
        return {
            "is_running": self.is_running,
            "global_queue_size": self.global_task_queue.qsize(),
            "max_global_queue_size": self.max_global_queue_size,
            "total_processes": total_processes,
            "alive_processes": alive_processes,
            "dead_processes": dead_processes,
            "total_restarts": total_restarts,
            "max_restarts_reached": max_restarts_reached,
            "load_balance_strategy": self.load_balance_strategy,
            "task_timeout": self.task_timeout,
            "stats": self.stats.copy(),
            "model_instances": {
                model_id: {
                    "gpu_count": len(gpus),
                    "gpu_ids": gpus
                }
                for model_id, gpus in self.model_instances.items()
            },
            "gpu_processes": gpu_status
        }
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        models = []
        for model_id, gpus in self.model_instances.items():
            model_info = {
                "model_id": model_id,
                "model_name": "FLUX.1-dev" if model_id == "flux1-dev" else model_id,
                "description": "Black Forest Labs FLUX.1-dev model for high-quality image generation",
                "total_gpu_processes": len(gpus),
                "available_gpu_processes": len(gpus),  # 简化，实际应该检查进程状态
                "supported_features": ["text-to-image"]
            }
            models.append(model_info)
        return models
    
    def shutdown(self):
        """关闭管理器"""
        logger.info("正在关闭进程级并发模型管理器...")
        
        self.is_running = False
        
        # 等待调度器线程结束
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        # 等待监控线程结束
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # 关闭GPU隔离管理器
        self.gpu_manager.shutdown()
        
        logger.info("进程级并发模型管理器已关闭") 