import asyncio
import threading
import time
import os
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple
import logging
from queue import Queue, Empty, PriorityQueue
import queue
from dataclasses import dataclass, field
import uuid
import torch
from gpu_isolation_manager import GPUIsolationManager
from config import Config
from utils import ValidationUtils, TaskUtils

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
            # 检查模型路径是否配置
            model_path = model_paths.get(model_id)
            if not model_path:
                logger.warning(f"⚠️ 模型 {model_id} 未配置路径，跳过")
                continue
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ 模型 {model_id} 路径不存在: {model_path}，跳过")
                continue
            
            if model_id not in self.model_instances:
                self.model_instances[model_id] = []
            
            # 为每个GPU创建隔离进程
            for gpu_device in gpu_list:
                if gpu_device.startswith("cuda:"):
                    gpu_id = gpu_device.split(":")[1]
                    
                    # 创建GPU隔离进程
                    if self.gpu_manager.create_gpu_process(gpu_id, model_path, model_id):
                        self.model_instances[model_id].append(gpu_id)
                        logger.info(f"✅ GPU {gpu_id} 进程创建成功 (模型: {model_id})")
                    else:
                        logger.error(f"❌ GPU {gpu_id} 进程创建失败 (模型: {model_id})")
        
        # 打印初始化结果
        total_processes = sum(len(gpus) for gpus in self.model_instances.values())
        supported_models = list(self.model_instances.keys())
        
        logger.info(f"🎉 模型进程初始化完成，总共创建了 {total_processes} 个GPU进程")
        logger.info(f"📋 支持的模型: {supported_models}")
        
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
        
        # 启动结果监听器
        self.result_listener_thread = threading.Thread(
            target=self._result_listener_loop,
            name="result-listener"
        )
        self.result_listener_thread.daemon = True
        self.result_listener_thread.start()
        logger.info("结果监听器已启动")
    
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
    
    def _result_listener_loop(self):
        """结果监听器循环 - 高效监听所有GPU进程的结果队列"""
        logger.info("📡 结果监听器开始运行")
        
        while self.is_running:
            try:
                # 检查所有GPU进程的结果队列
                for process_key, result_queue in self.gpu_manager.result_queues.items():
                    try:
                        # 非阻塞方式检查结果
                        result = result_queue.get_nowait()
                        
                        if result and 'task_id' in result:
                            task_id = result['task_id']
                            logger.info(f"📨 结果监听器收到任务 {task_id[:8]} 的结果 (GPU: {result.get('gpu_id', 'unknown')})")
                            
                            # 查找对应的任务结果队列
                            if task_id in self.task_results:
                                task_result_queue = self.task_results[task_id]
                                
                                # 添加线程信息
                                result['thread_name'] = threading.current_thread().name
                                
                                # 发送结果
                                task_result_queue.put(result)
                                logger.info(f"📤 任务 {task_id[:8]} 结果已发送到结果队列")
                                
                                # 更新统计
                                if result.get("success", False):
                                    self.stats["completed_tasks"] += 1
                                    logger.info(f"✅ 任务 {task_id[:8]} 完成 (GPU: {result.get('gpu_id', 'unknown')})")
                                else:
                                    self.stats["failed_tasks"] += 1
                                    logger.error(f"❌ 任务 {task_id[:8]} 失败: {result.get('error', '未知错误')}")
                                
                                # 清理任务记录
                                del self.task_results[task_id]
                                logger.debug(f"🧹 任务 {task_id[:8]} 记录已清理 (剩余任务数: {len(self.task_results)})")
                            else:
                                logger.warning(f"⚠️ 未找到任务 {task_id[:8]} 的结果队列")
                        
                    except queue.Empty:
                        # 队列为空，继续检查下一个
                        continue
                    except Exception as e:
                        logger.error(f"❌ 处理GPU {process_key} 结果时出错: {e}")
                
                # 短暂休眠，避免CPU占用过高，但保持响应性
                time.sleep(0.05)  # 50ms，比原来的100ms更快
                
            except Exception as e:
                logger.error(f"❌ 结果监听器循环错误: {e}")
                import traceback
                logger.error(f"错误详情: {traceback.format_exc()}")
                time.sleep(1.0)  # 出错时等待更长时间
    
    def _global_scheduler_loop(self):
        """全局调度器循环 - 智能任务分配，高并发处理"""
        logger.info("🚀 进程级全局调度器开始运行")
        
        while self.is_running:
            try:
                # 获取全局任务（带超时，避免无限阻塞）
                try:
                    task = self.global_task_queue.get(timeout=self.scheduler_sleep_time)
                except Empty:
                    # 队列为空，继续循环
                    continue
                
                logger.info(f"🎯 调度器获取到任务: {task.task_id[:8]}, 模型: {task.model_id}, 优先级: {task.priority}")
                
                # 找到最佳的GPU进程
                best_gpu_id = self._find_best_gpu(task.model_id)
                
                if best_gpu_id:
                    # 立即分配任务给GPU进程，不等待完成
                    logger.info(f"🚀 调度器分配任务 {task.task_id[:8]} 到GPU {best_gpu_id}")
                    self._submit_task_immediately(task, best_gpu_id)
                    logger.info(f"✅ 任务 {task.task_id[:8]} 已分配给GPU {best_gpu_id}")
                else:
                    # 所有GPU都忙碌，重新放回队列（优先级保持不变）
                    logger.warning(f"⚠️ 所有GPU都忙碌，任务 {task.task_id[:8]} 重新放回队列")
                    self.global_task_queue.put(task)
                    # 短暂等待，避免CPU占用过高
                    time.sleep(self.scheduler_sleep_time * 2)
                    
            except Exception as e:
                logger.error(f"❌ 全局调度器错误: {e}")
                import traceback
                logger.error(f"错误详情: {traceback.format_exc()}")
                # 出错时短暂等待，避免无限循环
                time.sleep(1.0)
    
    def _find_best_gpu(self, model_id: str) -> Optional[str]:
        """找到最佳的GPU进程 - 实现轮询负载均衡，检查可用性"""
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
        
        logger.info(f"🔍 为模型 {model_id} 寻找可用GPU，可用GPU列表: {available_gpus}")
        
        # 尝试找到可用的GPU，最多检查所有GPU一次
        checked_count = 0
        while checked_count < len(available_gpus):
            # 轮询策略：依次选择下一个GPU
            gpu_index = self.gpu_round_robin_counters[model_id] % len(available_gpus)
            selected_gpu = available_gpus[gpu_index]
            
            logger.info(f"🎲 轮询选择GPU {selected_gpu} (索引: {gpu_index}/{len(available_gpus)})")
            
            # 检查GPU是否可用
            if self._is_gpu_available(selected_gpu, model_id):
                # 更新计数器
                self.gpu_round_robin_counters[model_id] += 1
                logger.info(f"✅ 负载均衡: 模型 {model_id} 选择GPU {selected_gpu} (索引: {gpu_index}/{len(available_gpus)})")
                return selected_gpu
            else:
                # GPU不可用，尝试下一个
                self.gpu_round_robin_counters[model_id] += 1
                checked_count += 1
                logger.info(f"❌ GPU {selected_gpu} 不可用，尝试下一个 (已检查: {checked_count}/{len(available_gpus)})")
        
        # 所有GPU都不可用
        logger.warning(f"⚠️ 模型 {model_id} 的所有GPU都不可用")
        return None
    
    def _is_gpu_available(self, gpu_id: str, model_id: str) -> bool:
        """检查GPU是否可用 - 高效检查"""
        try:
            process_key = f"{model_id}_{gpu_id}"
            
            # 检查进程是否存在
            if process_key not in self.gpu_manager.processes:
                logger.debug(f"❌ GPU {gpu_id} 进程不存在 (key: {process_key})")
                return False
            
            # 检查进程是否活着
            process = self.gpu_manager.processes[process_key]
            if not process.is_alive():
                logger.debug(f"❌ GPU {gpu_id} 进程已死亡 (PID: {process.pid}, exitcode: {process.exitcode})")
                return False
            
            # 检查任务队列是否已满
            if process_key in self.gpu_manager.task_queues:
                task_queue = self.gpu_manager.task_queues[process_key]
                try:
                    # 非阻塞方式获取队列大小
                    queue_size = task_queue.qsize()
                    max_queue_size = 5  # 每个GPU队列最大大小
                    
                    if queue_size >= max_queue_size:
                        logger.debug(f"❌ GPU {gpu_id} 任务队列已满 (大小: {queue_size}/{max_queue_size})")
                        return False
                    else:
                        logger.debug(f"✅ GPU {gpu_id} 可用 (队列大小: {queue_size}/{max_queue_size})")
                        return True
                        
                except Exception as e:
                    # 如果无法获取队列大小，假设可用
                    logger.debug(f"✅ GPU {gpu_id} 可用 (无法获取队列大小: {e})")
                    return True
            else:
                logger.debug(f"❌ GPU {gpu_id} 任务队列不存在")
                return False
            
        except Exception as e:
            logger.debug(f"❌ 检查GPU {gpu_id} 可用性时出错: {e}")
            return False
    
    def _submit_task_immediately(self, task: GenerationTask, gpu_id: str):
        """异步提交任务到GPU进程，不阻塞调度器"""
        logger.info(f"📤 开始提交任务 {task.task_id[:8]} 到GPU {gpu_id}")
        
        # 存储任务结果队列，供结果监听器使用
        self.task_results[task.task_id] = task.result_queue
        logger.info(f"💾 任务 {task.task_id[:8]} 结果队列已存储 (当前任务数: {len(self.task_results)})")
        
        # 创建后台线程处理任务，不阻塞调度器
        thread = threading.Thread(
            target=self._process_task_on_gpu_process,
            args=(task, gpu_id),
            name=f"task-{task.task_id[:8]}"
        )
        thread.daemon = True
        thread.start()
        logger.info(f"🧵 任务 {task.task_id[:8]} 后台线程已启动 (线程名: {thread.name})")
        logger.info(f"✅ 任务 {task.task_id[:8]} 已异步提交到GPU {gpu_id}")
    
    def _process_task_on_gpu_process(self, task: GenerationTask, gpu_id: str):
        """在后台线程中处理GPU任务 - 真正的异步处理"""
        logger.info(f"⚙️ 后台线程开始处理任务 {task.task_id[:8]} (GPU: {gpu_id}, 线程: {threading.current_thread().name})")
        
        try:
            # 使用统一的任务数据构建工具
            task_data = TaskUtils.build_task_data(
                task_id=task.task_id,
                prompt=task.prompt,
                params=task.params
            )
            logger.info(f"📋 任务 {task.task_id[:8]} 数据已构建，准备提交到GPU进程")
            
            # 提交任务到GPU进程，不等待结果
            submit_success = self.gpu_manager.submit_task(gpu_id, task.model_id, task_data)
            
            if submit_success:
                logger.info(f"🎯 任务 {task.task_id[:8]} 已成功提交到GPU {gpu_id} 进程，等待异步结果")
                # 不在这里等待结果，让GPU进程异步处理
                # 结果会通过GPU进程的结果队列返回
            else:
                # 任务提交失败，立即返回错误
                logger.error(f"❌ 任务 {task.task_id[:8]} 提交到GPU {gpu_id} 失败")
                error_result = {
                    "success": False,
                    "error": "GPU进程不可用或队列已满",
                    "task_id": task.task_id,
                    "gpu_id": gpu_id,
                    "model_id": task.model_id
                }
                task.result_queue.put(error_result)
                self.stats["failed_tasks"] += 1
                logger.error(f"❌ 任务 {task.task_id[:8]} 提交失败")
                
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
        
        logger.info(f"🏁 后台线程完成处理任务 {task.task_id[:8]} (GPU: {gpu_id})")
    
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
        control_image: Optional[str] = None,
        controlnet_type: str = "depth",
        controlnet_conditioning_scale: Optional[float] = None,
        control_guidance_start: Optional[float] = None,
        control_guidance_end: Optional[float] = None,
        loras: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """异步生成图片 - 真正的异步并行处理"""
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
        
        # 验证ControlNet类型
        if mode == "controlnet" and not ValidationUtils.validate_controlnet_type(controlnet_type):
            return {
                "success": False,
                "error": f"不支持的controlnet类型: {controlnet_type}，支持的类型: {ValidationUtils.get_supported_controlnet_types()}",
                "task_id": "",
                "gpu_id": None,
                "model_id": model_id
            }
        
        # 创建任务
        task_id = str(uuid.uuid4())
        result_queue = Queue()
        
        # 使用统一的任务参数构建工具
        task_params = TaskUtils.build_task_params(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg=cfg,
            seed=seed,
            mode=mode,
            strength=strength,
            input_image=input_image,
            mask_image=mask_image,
            control_image=control_image,
            controlnet_type=controlnet_type,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            loras=loras
        )
        
        task = GenerationTask(
            task_id=task_id,
            model_id=model_id,
            prompt=prompt,
            params=task_params,
            result_queue=result_queue,
            created_at=time.time(),
            priority=priority
        )
        
        logger.info(f"🔄 创建任务: {task_id[:8]}, 模型: {model_id}, 模式: {mode}, 优先级: {priority}")
        logger.info(f"📝 任务详情: 提示词='{prompt[:50]}{'...' if len(prompt) > 50 else ''}', 参数={task_params}")
        
        # 检查全局队列是否过载
        if self.global_task_queue.qsize() >= self.max_global_queue_size:
            self.stats["queue_full_rejections"] += 1
            return {
                "success": False,
                "error": "服务器过载，请稍后重试",
                "task_id": task_id,
                "gpu_id": None,
                "model_id": model_id
            }
        
        # 提交到全局队列
        try:
            self.global_task_queue.put(task)
            self.stats["total_tasks"] += 1
            logger.info(f"📥 任务 {task_id[:8]} 已提交到全局队列 (队列大小: {self.global_task_queue.qsize()})")
        except Exception as e:
            logger.error(f"❌ 提交任务 {task_id[:8]} 到全局队列失败: {e}")
            return {
                "success": False,
                "error": f"提交任务失败: {str(e)}",
                "task_id": task_id,
                "gpu_id": None,
                "model_id": model_id
            }
        
        # 异步等待结果 - 使用asyncio.get_event_loop().run_in_executor
        try:
            # 将同步的队列等待转换为异步操作
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: result_queue.get(timeout=self.task_timeout)
            )
            
            logger.info(f"✅ 任务 {task_id[:8]} 完成: {result.get('success', False)}")
            
            # 添加任务ID到结果
            result["task_id"] = task_id
            result["model_id"] = model_id
            result["mode"] = mode
            if mode == "controlnet":
                result["controlnet_type"] = controlnet_type
            
            return result
            
        except Empty:
            logger.error(f"❌ 任务 {task_id[:8]} 超时")
            return {
                "success": False,
                "error": f"任务超时 ({self.task_timeout}秒)",
                "task_id": task_id,
                "gpu_id": None,
                "model_id": model_id
            }
        except Exception as e:
            logger.error(f"❌ 等待任务 {task_id[:8]} 结果时出错: {e}")
            return {
                "success": False,
                "error": f"等待结果失败: {str(e)}",
                "task_id": task_id,
                "gpu_id": None,
                "model_id": model_id
            }
        finally:
            # 清理任务结果队列
            self.task_results.pop(task_id, None)
    
    def get_task_result(self, task_id: str, timeout: float = 300) -> Optional[Dict[str, Any]]:
        """根据task_id获取任务结果"""
        if task_id not in self.task_results:
            return {
                "success": False,
                "error": f"任务 {task_id} 不存在",
                "task_id": task_id
            }
        
        result_queue = self.task_results[task_id]
        
        try:
            result = result_queue.get(timeout=timeout)
            # 清理结果队列
            del self.task_results[task_id]
            return result
        except Empty:
            return {
                "success": False,
                "error": f"任务 {task_id} 超时 ({timeout}秒)",
                "task_id": task_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"获取任务 {task_id} 结果失败: {str(e)}",
                "task_id": task_id
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
        
        # 等待结果监听器线程结束
        if hasattr(self, 'result_listener_thread') and self.result_listener_thread:
            self.result_listener_thread.join(timeout=5.0)
        
        # 关闭GPU隔离管理器
        self.gpu_manager.shutdown()
        
        logger.info("进程级并发模型管理器已关闭") 