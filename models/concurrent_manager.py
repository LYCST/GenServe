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
    def __init__(self, model: BaseModel, device: str, instance_id: str):
        self.model = model
        self.device = device
        self.instance_id = instance_id
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
                "is_busy": self.is_busy,
                "is_loaded": self.model.is_loaded,
                "current_task": self.current_task,
                "queue_size": self.task_queue.qsize(),
                "max_queue_size": self.max_queue_size,
                "total_generations": self.total_generations,
                "last_used": self.last_used,
                "cuda_visible_devices": self.device.split(":")[1]
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
    
    def _create_model_with_gpu_isolation(self, model_id: str, gpu_device: str, instance_id: str) -> Optional[BaseModel]:
        """创建具有GPU隔离的模型实例 - 修复版本"""
        try:
            # 创建模型实例
            if model_id == "flux1-dev":
                # 传递原始设备名，不在这里设置CUDA_VISIBLE_DEVICES
                model = FluxModel(gpu_device=gpu_device)
            else:
                logger.warning(f"未知模型类型: {model_id}")
                return None
            
            # 加载模型
            logger.info(f"开始加载模型 {model_id} (实例: {instance_id})")
            if model.load():
                logger.info(f"✅ 模型实例 {instance_id} 创建成功，目标设备: {gpu_device}")
                return model
            else:
                logger.error(f"❌ 模型实例 {model_id} 加载失败")
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
            
            # 为每个GPU创建一个模型实例（每个GPU只创建一个）
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
                        model = self._create_model_with_gpu_isolation(model_id, gpu_device, instance_id)
                        
                        if model:
                            instance = ModelInstance(model, gpu_device, instance_id)
                            self.model_instances[model_id].append(instance)
                            self.instance_lookup[instance_id] = instance
                            logger.info(f"✅ 模型实例 {instance_id} 创建成功")
                        else:
                            logger.error(f"❌ 模型实例 {model_id} 在 {gpu_device} 上加载失败")
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
                logger.info(f"    - {inst.instance_id} ({inst.device})")
    
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
                    # 立即标记GPU为忙碌状态，防止重复分配
                    best_instance.set_busy(True, task.task_id)
                    
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
        """找到最佳的模型实例 - 改进的负载均衡算法"""
        if model_id not in self.model_instances:
            logger.warning(f"⚠️ 模型 {model_id} 不存在")
            return None
        
        instances = self.model_instances[model_id]
        
        # 过滤可接受任务的实例
        available_instances = [
            inst for inst in instances 
            if inst.model.is_loaded and inst.can_accept_task() and not inst.is_busy
        ]
        
        if not available_instances:
            logger.debug(f"⚠️ 模型 {model_id} 没有可用实例")
            return None
        
        # 选择队列最短的空闲实例
        best = min(available_instances, key=lambda x: x.task_queue.qsize())
        logger.debug(f"✅ 选择空闲实例 {best.instance_id}，队列大小: {best.task_queue.qsize()}")
        return best
    
    def _gpu_worker_loop(self, instance: ModelInstance):
        """GPU工作线程循环"""
        logger.info(f"GPU工作线程开始运行: {instance.instance_id}")
        
        while self.is_running:
            try:
                # 获取任务
                task = instance.task_queue.get(timeout=1.0)
                
                # 处理任务
                self._process_task_on_gpu(task, instance)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"GPU工作线程 {instance.instance_id} 错误: {e}")
    
    def _check_and_cleanup_memory(self, instance: ModelInstance, force_cleanup: bool = False):
        """检查并清理内存 - GPU隔离版本"""
        if not instance.device.startswith("cuda:"):
            return
        
        try:
            gpu_id = instance.device.split(":")[1]
            old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            
            # 设置GPU隔离环境
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    # 在GPU隔离环境中，总是使用设备0
                    with torch.cuda.device(0):
                        allocated = torch.cuda.memory_allocated(0)
                        total = torch.cuda.get_device_properties(0).total_memory
                        usage_ratio = allocated / total
                        
                        logger.debug(f"GPU {gpu_id} (隔离环境) 内存使用率: {usage_ratio:.1%}")
                        
                        # 如果内存使用率超过95%或者强制清理，执行深度清理
                        if usage_ratio > 0.95 or force_cleanup:
                            logger.warning(f"GPU {gpu_id} 内存使用率过高 ({usage_ratio:.1%})，执行深度清理")
                            
                            # 先尝试模型的紧急清理
                            if hasattr(instance.model, '_emergency_cleanup'):
                                instance.model._emergency_cleanup()
                            else:
                                # 备用清理方法
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            # 检查清理效果
                            new_allocated = torch.cuda.memory_allocated(0)
                            new_usage_ratio = new_allocated / total
                            logger.info(f"GPU {gpu_id} 清理后内存使用率: {new_usage_ratio:.1%}")
                            
            finally:
                # 恢复环境变量
                if old_cuda_visible:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                
        except Exception as e:
            logger.warning(f"检查内存时出错: {e}")
    
    def _process_task_on_gpu(self, task: GenerationTask, instance: ModelInstance):
        """在GPU上处理任务 - 动态GPU隔离版本"""
        logger.info(f"开始处理任务 {task.task_id[:8]} (实例: {instance.instance_id})")
        
        # 设置GPU隔离环境变量
        old_cuda_visible = None
        gpu_id = None
        
        if instance.device.startswith("cuda:"):
            gpu_id = instance.device.split(":")[1]
            old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                logger.debug(f"设置GPU隔离环境: CUDA_VISIBLE_DEVICES={gpu_id}")
        
        try:
            # 更新模型设备配置
            if hasattr(instance.model, '_update_device_for_task'):
                instance.model._update_device_for_task(instance.device)
            
            # 执行生成任务
            result = instance.model.generate(task.prompt, **task.params)
            
            # 添加任务信息
            result.update({
                "task_id": task.task_id,
                "device": instance.device,
                "instance_id": instance.instance_id,
                "cuda_visible_devices": gpu_id if instance.device.startswith("cuda:") else "cpu",
                "physical_gpu": gpu_id if instance.device.startswith("cuda:") else "cpu"
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
                "cuda_visible_devices": gpu_id if instance.device.startswith("cuda:") else "cpu",
                "physical_gpu": gpu_id if instance.device.startswith("cuda:") else "cpu"
            }
            task.result_queue.put(result)
            
            # 更新统计
            self.stats["failed_tasks"] += 1
            
        finally:
            # 恢复GPU环境变量
            if old_cuda_visible is not None:
                if old_cuda_visible:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                logger.debug(f"恢复 CUDA_VISIBLE_DEVICES={old_cuda_visible}")
            
            # 释放GPU
            instance.set_busy(False)
            
            # 任务完成后的标准清理
            try:
                if instance.device.startswith("cuda:") and gpu_id is not None:
                    # 在GPU隔离环境中进行清理
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                    if torch.cuda.is_available():
                        with torch.cuda.device(0):  # 在隔离环境中总是使用设备0
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    # 恢复环境变量
                    if old_cuda_visible:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    logger.debug(f"已清理GPU {instance.device} 缓存")
            except Exception as e:
                logger.warning(f"清理GPU缓存时出错: {e}")
    
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
        if self.global_task_queue.qsize() > self.max_global_queue_size:
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
        """关闭管理器 - 增强版本"""
        logger.info("正在关闭并发模型管理器...")
        
        self.is_running = False
        
        # 等待调度器线程结束
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        # 等待工作线程结束
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # 彻底卸载所有模型
        for instances in self.model_instances.values():
            for instance in instances:
                try:
                    logger.info(f"卸载模型实例: {instance.instance_id}")
                    instance.model.unload()
                    
                    # 强制清理这个实例使用的GPU
                    if instance.device.startswith("cuda:"):
                        gpu_id = int(instance.device.split(":")[1])
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                        logger.info(f"已清理GPU {instance.device}")
                        
                except Exception as e:
                    logger.error(f"卸载实例 {instance.instance_id} 时出错: {e}")
        
        # 最终清理所有GPU
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                logger.info("已清理所有GPU")
        except Exception as e:
            logger.warning(f"最终GPU清理时出错: {e}")
        
        logger.info("并发模型管理器已关闭") 