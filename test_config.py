#!/usr/bin/env python3
"""
配置系统测试脚本
"""

import os
import sys
import torch
from config import Config

def test_basic_config():
    """测试基本配置"""
    print("=" * 50)
    print("测试基本配置")
    print("=" * 50)
    
    config = Config.get_config()
    
    print(f"服务配置: {config['service']}")
    print(f"设备配置: {config['device']}")
    print(f"模型管理配置: {config['model_management']}")
    print(f"并发配置: {config['concurrent']}")
    print(f"安全配置: {config['security']}")
    print(f"日志配置: {config['logging']}")

def test_gpu_config():
    """测试GPU配置"""
    print("\n" + "=" * 50)
    print("测试GPU配置")
    print("=" * 50)
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print(f"可用GPU列表: {Config.get_available_gpus()}")
    print(f"模型GPU配置: {Config.get_model_gpu_config()}")
    print(f"自动检测配置: {Config.auto_detect_gpu_config()}")

def test_environment_variables():
    """测试环境变量配置"""
    print("\n" + "=" * 50)
    print("测试环境变量配置")
    print("=" * 50)
    
    # 测试FLUX_GPUS环境变量
    test_gpus = "0,1,2"
    os.environ["FLUX_GPUS"] = test_gpus
    
    config = Config.get_model_gpu_config()
    print(f"设置FLUX_GPUS={test_gpus}")
    print(f"结果: {config}")
    
    # 清理环境变量
    if "FLUX_GPUS" in os.environ:
        del os.environ["FLUX_GPUS"]
    
    # 测试FLUX_MODEL_PATH环境变量
    test_path = "/test/path/to/model"
    os.environ["FLUX_MODEL_PATH"] = test_path
    
    model_path = Config.get_model_path("flux1-dev")
    print(f"设置FLUX_MODEL_PATH={test_path}")
    print(f"结果: {model_path}")
    
    # 清理环境变量
    if "FLUX_MODEL_PATH" in os.environ:
        del os.environ["FLUX_MODEL_PATH"]

def test_validation():
    """测试验证功能"""
    print("\n" + "=" * 50)
    print("测试验证功能")
    print("=" * 50)
    
    # 测试提示词验证
    short_prompt = "a cat"
    long_prompt = "a" * (Config.MAX_PROMPT_LENGTH + 100)
    empty_prompt = ""
    
    print(f"短提示词验证: {Config.validate_prompt(short_prompt)}")
    print(f"长提示词验证: {Config.validate_prompt(long_prompt)}")
    print(f"空提示词验证: {Config.validate_prompt(empty_prompt)}")
    
    # 测试图片尺寸验证
    valid_size = (512, 512)
    invalid_size = (3000, 3000)
    zero_size = (0, 512)
    
    print(f"有效尺寸验证: {Config.validate_image_size(*valid_size)}")
    print(f"无效尺寸验证: {Config.validate_image_size(*invalid_size)}")
    print(f"零尺寸验证: {Config.validate_image_size(*zero_size)}")
    
    # 测试GPU设备验证
    valid_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    invalid_gpu = "cuda:999"
    cpu_device = "cpu"
    
    print(f"有效GPU验证: {Config.validate_gpu_device(valid_gpu)}")
    print(f"无效GPU验证: {Config.validate_gpu_device(invalid_gpu)}")
    print(f"CPU设备验证: {Config.validate_gpu_device(cpu_device)}")

def test_torch_dtype():
    """测试PyTorch数据类型"""
    print("\n" + "=" * 50)
    print("测试PyTorch数据类型")
    print("=" * 50)
    
    # 测试默认数据类型
    default_dtype = Config.get_torch_dtype()
    print(f"默认数据类型: {default_dtype}")
    
    # 测试不同数据类型
    for dtype_name in ["float16", "float32", "bfloat16"]:
        os.environ["TORCH_DTYPE"] = dtype_name
        dtype = Config.get_torch_dtype()
        print(f"{dtype_name}: {dtype}")
    
    # 清理环境变量
    if "TORCH_DTYPE" in os.environ:
        del os.environ["TORCH_DTYPE"]

def test_config_summary():
    """测试配置摘要"""
    print("\n" + "=" * 50)
    print("测试配置摘要")
    print("=" * 50)
    
    Config.print_config_summary()

def test_concurrent_config():
    """测试并发配置"""
    print("\n" + "=" * 50)
    print("测试并发配置")
    print("=" * 50)
    
    config = Config.get_config()
    concurrent_config = config["concurrent"]
    
    print(f"全局队列大小: {concurrent_config['max_global_queue_size']}")
    print(f"GPU队列大小: {concurrent_config['max_gpu_queue_size']}")
    print(f"任务超时: {concurrent_config['task_timeout']}秒")
    print(f"调度器睡眠时间: {concurrent_config['scheduler_sleep_time']}秒")
    print(f"负载均衡策略: {concurrent_config['load_balance_strategy']}")
    
    # 测试不同负载均衡策略
    strategies = ["queue_length", "memory_usage", "round_robin"]
    for strategy in strategies:
        os.environ["GPU_LOAD_BALANCE_STRATEGY"] = strategy
        config = Config.get_config()
        print(f"策略 {strategy}: {config['concurrent']['load_balance_strategy']}")
    
    # 清理环境变量
    if "GPU_LOAD_BALANCE_STRATEGY" in os.environ:
        del os.environ["GPU_LOAD_BALANCE_STRATEGY"]

def main():
    """主测试函数"""
    print("开始配置系统测试...")
    
    try:
        test_basic_config()
        test_gpu_config()
        test_environment_variables()
        test_validation()
        test_torch_dtype()
        test_concurrent_config()
        test_config_summary()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 