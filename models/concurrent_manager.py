import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue, Empty
from dataclasses import dataclass
from .base import BaseModel
from .flux_model import FluxModel
from device_manager import DeviceManager
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

class ModelInstance:
    """模型实例，绑定到特定GPU"""
    def __init__(self, model: BaseModel, device: str):
        self.model = model
        self.device = device
        self.is_busy = False
        self.last_used = time.time()
        self.total_generations = 0
        self.lock = threading.Lock()
        self.task_queue = Queue()  # 每个GPU的任务队列
    
    def is_available(self) -> bool:
        """检查实例是否可用"""
        with self.lock:
            return not self.is_busy and self.model.is_loaded
    
    def set_busy(self, busy: bool):
        """设置忙碌状态"""
        with self.lock:
            self.is_busy = busy
            if not busy:
                self.last_used = time.time()
                self.total_generations += 1

class ConcurrentModelManager:
    """并发模型管理器 - 每个GPU一次只能执行一个任务"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.model_instances: Dict[str, List[ModelInstance]] = {}  # model_id -> [instances]
        self.global_task_queue = Queue()  # 全局任务队列
        self.worker_threads = []
        self.is_running = False
        self._initialize_models()
        self._start_workers()
    
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
                        # 创建模型实例
                        if model_id == "flux1-dev":
                            model = FluxModel(gpu_device=gpu_device)
                        else:
                            logger.warning(f"未知模型类型: {model_id}")
                            continue
                        
                        # 加载模型
                        if model.load():
                            instance = ModelInstance(model, gpu_device)
                            self.model_instances[model_id].append(instance)
                            logger.info(f"模型实例 {model_id} 在 {gpu_device} 上创建成功")
                        else:
                            logger.error(f"模型实例 {model_id} 在 {gpu_device} 上加载失败")
                    except Exception as e:
                        logger.error(f"创建模型实例 {model_id} 在 {gpu_device} 失败: {e}")
    
    def _start_workers(self):
        """启动工作线程"""
        self.is_running = True
        
        # 为每个GPU启动一个专门的工作线程
        for model_id, instances in self.model_instances.items():
            for i, instance in enumerate(instances):
                worker = threading.Thread(
                    target=self._gpu_worker_loop, 
                    args=(f"gpu-worker-{instance.device}", instance)
                )
                worker.daemon = True
                worker.start()
                self.worker_threads.append(worker)
                logger.info(f"GPU工作线程 {worker.name} 已启动，负责 {instance.device}")
        
        # 启动全局调度线程
        scheduler = threading.Thread(target=self._scheduler_loop, args=("scheduler",))
        scheduler.daemon = True
        scheduler.start()
        self.worker_threads.append(scheduler)
        logger.info("全局调度线程已启动")
    
    def _scheduler_loop(self, scheduler_name: str):
        """全局调度循环 - 将任务分配给可用的GPU"""
        logger.info(f"全局调度器 {scheduler_name} 开始运行")
        
        while self.is_running:
            try:
                # 获取全局任务
                task = self.global_task_queue.get(timeout=1.0)
                
                # 找到可用的GPU实例
                available_instance = self._find_available_instance(task.model_id)
                
                if available_instance:
                    # 将任务分配给GPU
                    available_instance.task_queue.put(task)
                    logger.info(f"任务 {task.task_id} 已分配给 {available_instance.device}")
                else:
                    # 没有可用GPU，重新放回全局队列
                    self.global_task_queue.put(task)
                    time.sleep(0.1)  # 短暂等待
                    
            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                logger.error(f"全局调度器处理任务时发生错误: {e}")
    
    def _gpu_worker_loop(self, worker_name: str, instance: ModelInstance):
        """GPU工作线程循环 - 每个GPU一个线程"""
        logger.info(f"GPU工作线程 {worker_name} 开始运行，负责 {instance.device}")
        
        while self.is_running:
            try:
                # 获取分配给这个GPU的任务
                task = instance.task_queue.get(timeout=1.0)
                
                # 处理任务
                self._process_task_on_gpu(task, instance, worker_name)
                
            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                logger.error(f"GPU工作线程 {worker_name} 处理任务时发生错误: {e}")
    
    def _process_task_on_gpu(self, task: GenerationTask, instance: ModelInstance, worker_name: str):
        """在指定GPU上处理任务"""
        logger.info(f"GPU工作线程 {worker_name} 开始处理任务 {task.task_id}")
        
        # 标记GPU为忙碌
        instance.set_busy(True)
        
        try:
            logger.info(f"任务 {task.task_id} 在 {instance.device} 上执行")
            
            # 设置CUDA设备
            if instance.device.startswith("cuda:"):
                import torch
                gpu_id = int(instance.device.split(":")[1])
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            
            # 执行生成
            result = instance.model.generate(task.prompt, **task.params)
            result["task_id"] = task.task_id
            result["device"] = instance.device
            result["worker"] = worker_name
            
            # 返回结果
            task.result_queue.put(result)
            
            logger.info(f"任务 {task.task_id} 完成，耗时: {result.get('elapsed_time', 0):.2f}秒")
            
        except Exception as e:
            logger.error(f"处理任务 {task.task_id} 时发生错误: {e}")
            result = {
                "success": False,
                "error": str(e),
                "task_id": task.task_id
            }
            task.result_queue.put(result)
        finally:
            # 释放GPU
            instance.set_busy(False)
    
    def _find_available_instance(self, model_id: str) -> Optional[ModelInstance]:
        """找到可用的模型实例"""
        if model_id not in self.model_instances:
            return None
        
        instances = self.model_instances[model_id]
        available_instances = [inst for inst in instances if inst.is_available()]
        
        if not available_instances:
            return None
        
        # 选择最近最少使用的实例
        best_instance = min(available_instances, key=lambda x: x.last_used)
        return best_instance
    
    async def generate_image_async(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """异步生成图片"""
        task_id = f"{model_id}_{int(time.time() * 1000)}_{id(prompt)}"
        
        # 创建结果队列
        result_queue = Queue()
        
        # 创建任务
        task = GenerationTask(
            task_id=task_id,
            model_id=model_id,
            prompt=prompt,
            params=kwargs,
            result_queue=result_queue,
            created_at=time.time()
        )
        
        # 添加到全局任务队列
        self.global_task_queue.put(task)
        logger.info(f"任务 {task_id} 已加入全局队列，当前队列大小: {self.global_task_queue.qsize()}")
        
        # 等待结果
        try:
            # 设置超时时间为5分钟
            result = await asyncio.get_event_loop().run_in_executor(
                None, result_queue.get, True, 300
            )
            return result
        except Exception as e:
            logger.error(f"等待任务 {task_id} 结果时发生错误: {e}")
            return {
                "success": False,
                "error": f"任务超时或发生错误: {str(e)}",
                "task_id": task_id
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        status = {
            "is_running": self.is_running,
            "global_queue_size": self.global_task_queue.qsize(),
            "worker_threads": len(self.worker_threads),
            "model_instances": {}
        }
        
        for model_id, instances in self.model_instances.items():
            status["model_instances"][model_id] = []
            for i, instance in enumerate(instances):
                status["model_instances"][model_id].append({
                    "index": i,
                    "device": instance.device,
                    "is_busy": instance.is_busy,
                    "is_loaded": instance.model.is_loaded,
                    "total_generations": instance.total_generations,
                    "last_used": instance.last_used,
                    "queue_size": instance.task_queue.qsize()
                })
        
        return status
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        models = []
        for model_id, instances in self.model_instances.items():
            if instances:
                # 使用第一个实例的信息
                first_instance = instances[0]
                model_info = first_instance.model.get_info()
                model_info["total_instances"] = len(instances)
                model_info["available_instances"] = len([inst for inst in instances if inst.is_available()])
                models.append(model_info)
        return models
    
    def shutdown(self):
        """关闭管理器"""
        logger.info("正在关闭并发模型管理器...")
        
        self.is_running = False
        
        # 等待工作线程结束
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        # 卸载所有模型
        for instances in self.model_instances.values():
            for instance in instances:
                instance.model.unload()
        
        logger.info("并发模型管理器已关闭") 