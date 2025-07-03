import asyncio
import threading
import time
import os
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue, Empty, PriorityQueue
from dataclasses import dataclass, field
from .base import BaseModel
from .flux_model import FluxModel
from device_manager import DeviceManager
from config import Config
import uuid
import torch

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

class ModelInstance:
    """模型实例，绑定到特定GPU"""
    def __init__(self, model: BaseModel, device: str, instance_id: str, physical_gpu_id: str):
        self.model = model
        self.device = device
        self.instance_id = instance_id
        self.physical_gpu_id = physical_gpu_id  # 物理GPU ID
        self.is_busy = False
        self.last_used = time.time()
        self.total_generations = 0
        self.current_task = None
        self.lock = threading.Lock()
        self.task_queue = PriorityQueue()  # 使用优先级队列
        
        # 从配置获取队列大小限制
        config = Config.get_config()
        self.max_queue_size = config["concurrent"]["max_gpu_queue_size"]
    
    def is_available(self) -> bool:
        """检查实例是否可用"""
        with self.lock:
            return not self.is_busy and self.model.is_loaded
    
    def can_accept_task(self) -> bool:
        """检查是否可以接受新任务（考虑队列大小）"""
        return self.task_queue.qsize() < self.max_queue_size
    
    def set_busy(self, busy: bool, task_id: Optional[str] = None):
        """设置忙碌状态"""
        with self.lock:
            self.is_busy = busy
            self.current_task = task_id if busy else None
            if not busy:
                self.last_used = time.time()
                self.total_generations += 1
    
    def get_status(self) -> Dict[str, Any]:
        """获取实例状态"""
        with self.lock:
            return {
                "instance_id": self.instance_id,
                "device": self.device,
                "physical_gpu_id": self.physical_gpu_id,
                "is_busy": self.is_busy,
                "is_loaded": self.model.is_loaded,
                "current_task": self.current_task,
                "queue_size": self.task_queue.qsize(),
                "max_queue_size": self.max_queue_size,
                "total_generations": self.total_generations,
                "last_used": self.last_used
            }

class ConcurrentModelManager:
    """改进的并发模型管理器 - 支持GPU环境隔离"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.model_instances: Dict[str, List[ModelInstance]] = {}  # model_id -> [instances]
        self.instance_lookup: Dict[str, ModelInstance] = {}  # instance_id -> instance
        
        # 全局任务队列和调度
        self.global_task_queue = PriorityQueue()
        self.task_results: Dict[str, Queue] = {}  # task_id -> result_queue
        
        # 工作线程管理
        self.worker_threads = []
        self.scheduler_thread = None
        self.is_running = False
        
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
    
    def _create_model_instance_for_gpu(self, model_id: str, gpu_device: str, instance_id: str) -> Optional[ModelInstance]:
        """为特定GPU创建模型实例（使用子进程隔离）"""
        try:
            # 提取物理GPU ID
            physical_gpu_id = gpu_device.split(":")[1] if gpu_device.startswith("cuda:") else "cpu"
            
            logger.info(f"正在为GPU {physical_gpu_id} 创建模型实例 {instance_id}")
            
            # 创建模型实例
            if model_id == "flux1-dev":
                # 创建模型时传入物理GPU ID，让模型内部处理GPU隔离
                model = FluxModel(gpu_device=gpu_device, physical_gpu_id=physical_gpu_id)
            else:
                logger.warning(f"未知模型类型: {model_id}")
                return None
            
            # 加载模型
            if model.load():
                instance = ModelInstance(model, gpu_device, instance_id, physical_gpu_id)
                logger.info(f"✅ 模型实例 {instance_id} 创建成功，物理GPU: {physical_gpu_id}")
                return instance
            else:
                logger.error(f"❌ 模型实例 {instance_id} 加载失败")
                return None
                
        except Exception as e:
            logger.error(f"❌ 创建模型实例失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _initialize_models(self):
        """初始化模型实例"""
        model_gpu_config = Config.get_config()["model_management"]["model_gpu_config"]
        
        for model_id, gpu_list in model_gpu_config.items():
            if model_id not in self.model_instances:
                self.model_instances[model_id] = []
            
            # 为每个GPU创建一个模型实例
            for gpu_device in gpu_list:
                if self.device_manager.validate_device(gpu_device):
                    try:
                        # 生成唯一实例ID
                        instance_id = f"{model_id}_{gpu_device.replace(':', '_')}"
                        
                        # 检查是否已经存在这个实例
                        if instance_id in self.instance_lookup:
                            logger.warning(f"实例 {instance_id} 已存在，跳过创建")
                            continue
                        
                        # 创建模型实例
                        instance = self._create_model_instance_for_gpu(model_id, gpu_device, instance_id)
                        
                        if instance:
                            self.model_instances[model_id].append(instance)
                            self.instance_lookup[instance_id] = instance
                        else:
                            logger.error(f"❌ 无法创建模型实例 {instance_id}")
                    except Exception as e:
                        logger.error(f"❌ 创建模型实例失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
        
        # 打印初始化结果
        total_instances = sum(len(instances) for instances in self.model_instances.values())
        logger.info(f"🎉 模型初始化完成，总共创建了 {total_instances} 个实例")
        for model_id, instances in self.model_instances.items():
            logger.info(f"  {model_id}: {len(instances)} 个实例")
            for inst in instances:
                logger.info(f"    - {inst.instance_id} (设备: {inst.device}, 物理GPU: {inst.physical_gpu_id})")
    
    def _start_manager(self):
        """启动管理器"""
        self.is_running = True
        
        # 启动全局调度器
        self.scheduler_thread = threading.Thread(
            target=self._global_scheduler_loop, 
            name="global-scheduler"
        )
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("全局调度器已启动")
        
        # 为每个GPU实例启动工作线程
        for model_id, instances in self.model_instances.items():
            for instance in instances:
                worker = threading.Thread(
                    target=self._gpu_worker_loop,
                    args=(instance,),
                    name=f"worker-{instance.instance_id}"
                )
                worker.daemon = True
                worker.start()
                self.worker_threads.append(worker)
                logger.info(f"GPU工作线程 {worker.name} 已启动")
    
    def _global_scheduler_loop(self):
        """全局调度器循环 - 智能任务分配"""
        logger.info("全局调度器开始运行")
        
        while self.is_running:
            try:
                # 获取全局任务（带超时）
                task = self.global_task_queue.get(timeout=self.scheduler_sleep_time)
                
                # 找到最佳的GPU实例
                best_instance = self._find_best_instance(task.model_id)
                
                if best_instance:
                    # 分配任务给GPU
                    best_instance.task_queue.put(task)
                    logger.info(f"任务 {task.task_id} 已分配给 {best_instance.instance_id} (设备: {best_instance.device})")
                else:
                    # 所有GPU都忙碌或队列已满，重新放回队列
                    self.global_task_queue.put(task)
                    # 更长的等待时间，避免CPU占用过高
                    time.sleep(self.scheduler_sleep_time * 5)
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"全局调度器错误: {e}")
    
    def _find_best_instance(self, model_id: str) -> Optional[ModelInstance]:
        """找到最佳的模型实例"""
        if model_id not in self.model_instances:
            logger.warning(f"⚠️ 模型 {model_id} 不存在")
            return None
        
        instances = self.model_instances[model_id]
        
        # 首先找空闲的实例
        available_instances = [
            inst for inst in instances 
            if inst.model.is_loaded and not inst.is_busy and inst.can_accept_task()
        ]
        
        if available_instances:
            # 选择队列最短的空闲实例
            best = min(available_instances, key=lambda x: x.task_queue.qsize())
            logger.debug(f"✅ 选择空闲实例 {best.instance_id}，队列大小: {best.task_queue.qsize()}")
            return best
        
        # 如果没有空闲的，找队列未满的实例
        queueable_instances = [
            inst for inst in instances 
            if inst.model.is_loaded and inst.can_accept_task()
        ]
        
        if queueable_instances:
            # 选择队列最短的实例
            best = min(queueable_instances, key=lambda x: x.task_queue.qsize())
            logger.debug(f"✅ 选择可排队实例 {best.instance_id}，队列大小: {best.task_queue.qsize()}")
            return best
        
        logger.debug(f"⚠️ 模型 {model_id} 没有可用实例")
        return None
    
    def _gpu_worker_loop(self, instance: ModelInstance):
        """GPU工作线程循环"""
        logger.info(f"GPU工作线程开始运行: {instance.instance_id}")
        
        while self.is_running:
            try:
                # 获取任务
                task = instance.task_queue.get(timeout=1.0)
                
                # 标记为忙碌
                instance.set_busy(True, task.task_id)
                
                # 处理任务
                self._process_task_on_gpu(task, instance)
                
                # 标记为空闲
                instance.set_busy(False)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"GPU工作线程 {instance.instance_id} 错误: {e}")
                instance.set_busy(False)
    
    def _process_task_on_gpu(self, task: GenerationTask, instance: ModelInstance):
        """在GPU上处理任务 - 线程级GPU隔离版本"""
        logger.info(f"开始处理任务 {task.task_id[:8]} (实例: {instance.instance_id})")
        
        try:
            # 更新模型设备配置
            if hasattr(instance.model, '_update_device_for_task'):
                instance.model._update_device_for_task(instance.device)
            
            # 执行生成任务 - 模型内部会使用torch.cuda.set_device()进行线程级隔离
            result = instance.model.generate(task.prompt, **task.params)
            
            # 添加任务信息
            result.update({
                "task_id": task.task_id,
                "device": instance.device,
                "instance_id": instance.instance_id,
                "physical_gpu": instance.physical_gpu_id,
                "thread_name": threading.current_thread().name
            })
            
            # 发送结果
            task.result_queue.put(result)
            
            # 更新统计
            if result.get("success", False):
                self.stats["completed_tasks"] += 1
                instance.total_generations += 1
                logger.info(f"✅ 任务 {task.task_id[:8]} 完成 (实例: {instance.instance_id})")
            else:
                self.stats["failed_tasks"] += 1
                logger.error(f"❌ 任务 {task.task_id[:8]} 失败: {result.get('error', '未知错误')}")
            
        except Exception as e:
            logger.error(f"❌ 处理任务 {task.task_id[:8]} 时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 失败时执行强制清理
            try:
                if instance.device.startswith("cuda:"):
                    logger.warning(f"任务失败，对GPU {instance.device} 执行强制清理")
                    self._check_and_cleanup_memory(instance, force_cleanup=True)
            except Exception as cleanup_error:
                logger.error(f"强制清理GPU显存时出错: {cleanup_error}")
            
            result = {
                "success": False,
                "error": str(e),
                "task_id": task.task_id,
                "device": instance.device,
                "instance_id": instance.instance_id,
                "physical_gpu": instance.physical_gpu_id,
                "thread_name": threading.current_thread().name
            }
            task.result_queue.put(result)
            
            # 更新统计
            self.stats["failed_tasks"] += 1
            
        finally:
            # 释放GPU
            instance.set_busy(False)
    
    async def generate_image_async(self, model_id: str, prompt: str, priority: int = 0, **kwargs) -> Dict[str, Any]:
        """异步生成图片 - 支持优先级"""
        task_id = str(uuid.uuid4())
        
        # 检查是否有可用实例
        if model_id not in self.model_instances or not self.model_instances[model_id]:
            return {
                "success": False,
                "error": f"模型 {model_id} 不可用",
                "task_id": task_id
            }
        
        # 检查全局队列是否过载
        if self.global_task_queue.qsize() >= self.max_global_queue_size:
            self.stats["queue_full_rejections"] += 1
            return {
                "success": False,
                "error": "服务器过载，请稍后重试",
                "task_id": task_id
            }
        
        # 创建结果队列
        result_queue = Queue()
        self.task_results[task_id] = result_queue
        
        # 创建任务
        task = GenerationTask(
            task_id=task_id,
            model_id=model_id,
            prompt=prompt,
            params=kwargs,
            result_queue=result_queue,
            created_at=time.time(),
            priority=priority
        )
        
        # 添加到全局任务队列
        self.global_task_queue.put(task)
        self.stats["total_tasks"] += 1
        
        logger.info(f"任务 {task_id} 已加入队列，优先级: {priority}")
        
        try:
            # 等待结果（使用配置的超时时间）
            result = await asyncio.get_event_loop().run_in_executor(
                None, result_queue.get, True, self.task_timeout
            )
            return result
            
        except Exception as e:
            logger.error(f"等待任务 {task_id} 结果时发生错误: {e}")
            return {
                "success": False,
                "error": f"任务超时或发生错误: {str(e)}",
                "task_id": task_id
            }
        finally:
            # 清理结果队列
            self.task_results.pop(task_id, None)
    
    def get_status(self) -> Dict[str, Any]:
        """获取详细状态"""
        # 计算总体统计
        total_queue_size = sum(
            inst.task_queue.qsize() 
            for instances in self.model_instances.values() 
            for inst in instances
        )
        
        busy_instances = sum(
            1 for instances in self.model_instances.values() 
            for inst in instances if inst.is_busy
        )
        
        total_instances = sum(
            len(instances) for instances in self.model_instances.values()
        )
        
        return {
            "is_running": self.is_running,
            "global_queue_size": self.global_task_queue.qsize(),
            "max_global_queue_size": self.max_global_queue_size,
            "total_queue_size": total_queue_size,
            "worker_threads": len(self.worker_threads),
            "busy_instances": busy_instances,
            "total_instances": total_instances,
            "load_balance_strategy": self.load_balance_strategy,
            "task_timeout": self.task_timeout,
            "stats": self.stats.copy(),
            "model_instances": {
                model_id: [inst.get_status() for inst in instances]
                for model_id, instances in self.model_instances.items()
            }
        }
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        models = []
        for model_id, instances in self.model_instances.items():
            if instances:
                first_instance = instances[0]
                model_info = first_instance.model.get_info()
                model_info.update({
                    "total_instances": len(instances),
                    "available_instances": len([inst for inst in instances if inst.is_available()]),
                    "busy_instances": len([inst for inst in instances if inst.is_busy]),
                    "total_queue_size": sum(inst.task_queue.qsize() for inst in instances)
                })
                models.append(model_info)
        return models
    
    def shutdown(self):
        """关闭管理器"""
        logger.info("正在关闭并发模型管理器...")
        
        self.is_running = False
        
        # 等待调度器线程结束
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        # 等待工作线程结束
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # 卸载所有模型
        for instances in self.model_instances.values():
            for instance in instances:
                try:
                    logger.info(f"卸载模型实例: {instance.instance_id}")
                    instance.model.unload()
                except Exception as e:
                    logger.error(f"卸载实例 {instance.instance_id} 时出错: {e}")
        
        logger.info("并发模型管理器已关闭")