#!/usr/bin/env python3
"""
进程重启任务处理测试脚本
验证当GPU进程重启时，丢失的任务是否能正确处理
"""

import asyncio
import aiohttp
import time
import json
import random
import signal
import os
import psutil
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_genserve_processes():
    """查找GenServe相关进程"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('genserve' in cmd.lower() or 'main.py' in cmd.lower() for cmd in proc.info['cmdline']):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def kill_gpu_process(gpu_id: str = "1"):
    """强制杀死指定的GPU进程来模拟OOM"""
    logger.info(f"🔪 尝试杀死物理GPU {gpu_id} 进程来模拟OOM...")
    
    # 查找GPU工作进程
    target_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any(f'gpu-worker-{gpu_id}' in cmd for cmd in proc.info['cmdline']):
                target_processes.append(proc)
                logger.info(f"找到目标进程: PID {proc.info['pid']}, 名称: {proc.info['name']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if target_processes:
        for proc in target_processes:
            try:
                logger.info(f"杀死进程 PID {proc.info['pid']}")
                proc.kill()
                logger.info(f"✅ 进程 PID {proc.info['pid']} 已被杀死")
            except Exception as e:
                logger.error(f"❌ 杀死进程 PID {proc.info['pid']} 失败: {e}")
    else:
        logger.warning(f"⚠️ 未找到物理GPU {gpu_id} 的工作进程")

async def test_process_restart_scenario(base_url: str = "http://localhost:12411"):
    """测试进程重启场景"""
    logger.info("🧪 开始进程重启任务处理测试")
    logger.info(f"   服务地址: {base_url}")
    
    async with aiohttp.ClientSession() as session:
        # 第一步：发送一个长时间运行的任务
        logger.info(f"\n{'='*60}")
        logger.info("步骤1: 发送长时间运行的任务")
        logger.info(f"{'='*60}")
        
        long_task_data = {
            "prompt": "A very detailed landscape with mountains, rivers, and forests, high quality, 8k resolution",
            "model_id": "flux1-dev",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 50,  # 较长的推理步数
            "cfg": 3.5,
            "seed": random.randint(1, 1000000),
            "priority": 0,
            "mode": "text2img"
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{base_url}/generate", json=long_task_data) as response:
                if response.status == 200:
                    response_json = await response.json()
                    task_id = response_json.get("task_id", "")
                    gpu_id = response_json.get("gpu_id", "")
                    logger.info(f"✅ 长时间任务已提交")
                    logger.info(f"   任务ID: {task_id[:8]}")
                    logger.info(f"   物理GPU: {gpu_id}")
                    logger.info(f"   预计需要较长时间完成...")
                    
                    # 等待一段时间让任务开始执行
                    await asyncio.sleep(5)
                    
                    # 第二步：强制杀死GPU进程
                    logger.info(f"\n{'='*60}")
                    logger.info("步骤2: 强制杀死GPU进程模拟OOM")
                    logger.info(f"{'='*60}")
                    
                    kill_gpu_process(gpu_id)
                    
                    # 第三步：等待进程重启和任务处理
                    logger.info(f"\n{'='*60}")
                    logger.info("步骤3: 等待进程重启和任务处理")
                    logger.info(f"{'='*60}")
                    
                    logger.info("⏳ 等待30秒让进程重启和任务处理...")
                    await asyncio.sleep(30)
                    
                    # 第四步：检查任务状态
                    logger.info(f"\n{'='*60}")
                    logger.info("步骤4: 检查任务状态")
                    logger.info(f"{'='*60}")
                    
                    try:
                        async with session.get(f"{base_url}/task/{task_id}") as status_response:
                            if status_response.status == 200:
                                status_json = await status_response.json()
                                success = status_json.get("success", False)
                                error = status_json.get("error", "")
                                
                                if success:
                                    logger.info(f"✅ 任务 {task_id[:8]} 成功完成")
                                else:
                                    logger.info(f"❌ 任务 {task_id[:8]} 失败: {error}")
                                    if "进程重启" in error or "任务丢失" in error:
                                        logger.info(f"✅ 正确检测到进程重启导致的任务丢失")
                                    else:
                                        logger.warning(f"⚠️ 任务失败原因不是进程重启")
                            else:
                                logger.warning(f"⚠️ 无法获取任务状态，状态码: {status_response.status}")
                    except Exception as e:
                        logger.error(f"❌ 检查任务状态时出错: {e}")
                    
                    # 第五步：发送新任务验证系统恢复
                    logger.info(f"\n{'='*60}")
                    logger.info("步骤5: 发送新任务验证系统恢复")
                    logger.info(f"{'='*60}")
                    
                    new_task_data = {
                        "prompt": "A simple test image after restart",
                        "model_id": "flux1-dev",
                        "height": 512,
                        "width": 512,
                        "num_inference_steps": 10,  # 快速测试
                        "cfg": 3.5,
                        "seed": random.randint(1, 1000000),
                        "priority": 0,
                        "mode": "text2img"
                    }
                    
                    try:
                        async with session.post(f"{base_url}/generate", json=new_task_data) as new_response:
                            if new_response.status == 200:
                                new_response_json = await new_response.json()
                                new_task_id = new_response_json.get("task_id", "")
                                new_gpu_id = new_response_json.get("gpu_id", "")
                                new_success = new_response_json.get("success", False)
                                
                                logger.info(f"✅ 新任务提交成功")
                                logger.info(f"   任务ID: {new_task_id[:8]}")
                                logger.info(f"   物理GPU: {new_gpu_id}")
                                logger.info(f"   成功状态: {new_success}")
                                
                                if new_success:
                                    logger.info(f"✅ 系统已恢复正常，新任务可以正常处理")
                                else:
                                    logger.warning(f"⚠️ 新任务失败，系统可能未完全恢复")
                            else:
                                logger.error(f"❌ 新任务提交失败，状态码: {new_response.status}")
                    except Exception as e:
                        logger.error(f"❌ 发送新任务时出错: {e}")
                        
                else:
                    logger.error(f"❌ 长时间任务提交失败，状态码: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ 测试过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

async def test_concurrent_with_restart(base_url: str = "http://localhost:12411"):
    """测试并发任务在进程重启时的处理"""
    logger.info(f"\n{'='*60}")
    logger.info("🧪 开始并发任务进程重启测试")
    logger.info(f"{'='*60}")
    
    async def send_task(session: aiohttp.ClientSession, task_id: int, delay: float = 0):
        """发送单个任务"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        data = {
            "prompt": f"Test image {task_id}, detailed landscape",
            "model_id": "flux1-dev",
            "height": 768,
            "width": 768,
            "num_inference_steps": 30,
            "cfg": 3.5,
            "seed": random.randint(1, 1000000),
            "priority": 0,
            "mode": "text2img"
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{base_url}/generate", json=data) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    response_json = await response.json()
                    success = response_json.get("success", False)
                    gpu_id = response_json.get("gpu_id", "")
                    error = response_json.get("error", "")
                    
                    logger.info(f"任务 {task_id}: {'✅' if success else '❌'} (GPU: {gpu_id}, 耗时: {response_time:.2f}s)")
                    if not success:
                        logger.info(f"   错误: {error}")
                    
                    return {
                        "task_id": task_id,
                        "success": success,
                        "gpu_id": gpu_id,
                        "response_time": response_time,
                        "error": error
                    }
                else:
                    logger.error(f"任务 {task_id}: ❌ HTTP {response.status}")
                    return {
                        "task_id": task_id,
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            logger.error(f"任务 {task_id}: ❌ 异常 {e}")
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e)
            }
    
    async with aiohttp.ClientSession() as session:
        # 发送多个并发任务
        tasks = []
        for i in range(5):
            # 第3个任务延迟发送，在进程重启期间
            delay = 10 if i == 2 else 0
            task = send_task(session, i + 1, delay)
            tasks.append(task)
        
        logger.info("📤 发送5个并发任务...")
        logger.info("   任务3将在10秒后发送（预计在进程重启期间）")
        
        # 在任务3发送前杀死进程
        await asyncio.sleep(8)
        logger.info("🔪 杀死GPU进程...")
        kill_gpu_process("1")
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析结果
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
        
        logger.info(f"\n📊 并发测试结果:")
        logger.info(f"   总任务数: {len(results)}")
        logger.info(f"   成功任务: {len(successful_results)}")
        logger.info(f"   失败任务: {len(failed_results)}")
        
        # 分析失败原因
        restart_related_failures = 0
        for result in failed_results:
            error = result.get("error", "")
            if "进程重启" in error or "任务丢失" in error:
                restart_related_failures += 1
        
        logger.info(f"   进程重启相关失败: {restart_related_failures}")
        
        if restart_related_failures > 0:
            logger.info(f"✅ 正确检测到进程重启导致的任务丢失")
        else:
            logger.warning(f"⚠️ 未检测到进程重启相关的任务失败")

async def main():
    """主函数"""
    BASE_URL = "http://localhost:12411"
    
    try:
        # 检查服务是否运行
        logger.info("🔍 检查GenServe服务状态...")
        processes = find_genserve_processes()
        if processes:
            logger.info(f"✅ 找到 {len(processes)} 个GenServe进程")
            for proc in processes:
                logger.info(f"   PID: {proc.info['pid']}, 名称: {proc.info['name']}")
        else:
            logger.warning("⚠️ 未找到GenServe进程，请确保服务正在运行")
            return
        
        # 测试进程重启场景
        await test_process_restart_scenario(BASE_URL)
        
        # 等待一段时间
        logger.info("\n⏳ 等待20秒后进行并发测试...")
        await asyncio.sleep(20)
        
        # 测试并发任务在进程重启时的处理
        await test_concurrent_with_restart(BASE_URL)
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 所有测试完成")
        logger.info(f"{'='*60}")
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 运行测试
    asyncio.run(main()) 