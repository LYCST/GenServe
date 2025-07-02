#!/usr/bin/env python3
"""
模型加载诊断脚本
"""

import os
import sys
import torch
import logging
from config import Config
from models.flux_model import FluxModel
from device_manager import DeviceManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print(f"当前工作目录: {os.getcwd()}")

def check_config():
    """检查配置"""
    print("\n" + "=" * 50)
    print("配置检查")
    print("=" * 50)
    
    config = Config.get_config()
    print(f"模型GPU配置: {config['model_management']['model_gpu_config']}")
    print(f"模型路径: {config['model_management']['model_paths']}")
    
    # 检查模型路径
    model_path = Config.get_model_path("flux1-dev")
    print(f"Flux模型路径: {model_path}")
    print(f"路径存在: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print(f"路径内容:")
        try:
            files = os.listdir(model_path)
            for file in files[:10]:  # 只显示前10个文件
                print(f"  {file}")
            if len(files) > 10:
                print(f"  ... 还有 {len(files) - 10} 个文件")
        except Exception as e:
            print(f"  读取目录失败: {e}")

def check_device_manager():
    """检查设备管理器"""
    print("\n" + "=" * 50)
    print("设备管理器检查")
    print("=" * 50)
    
    device_manager = DeviceManager()
    devices = device_manager.get_available_devices()
    print(f"可用设备: {devices}")
    
    # 测试设备验证
    for device in ["cpu", "cuda:0", "cuda:1", "cuda:999"]:
        is_valid = device_manager.validate_device(device)
        print(f"设备 {device} 有效: {is_valid}")

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 50)
    print("模型加载测试")
    print("=" * 50)
    
    # 获取GPU配置
    gpu_config = Config.get_model_gpu_config()
    flux_gpus = gpu_config.get("flux1-dev", [])
    
    print(f"Flux GPU配置: {flux_gpus}")
    
    if not flux_gpus:
        print("❌ 没有配置GPU设备")
        return
    
    # 测试第一个GPU
    test_gpu = flux_gpus[0]
    print(f"测试GPU: {test_gpu}")
    
    try:
        print(f"正在创建FluxModel实例...")
        model = FluxModel(gpu_device=test_gpu)
        print(f"模型实例创建成功")
        
        print(f"正在加载模型...")
        success = model.load()
        print(f"模型加载结果: {success}")
        
        if success:
            print(f"模型信息: {model.get_info()}")
            print(f"模型已加载: {model.is_loaded}")
            
            # 测试简单生成
            print(f"测试简单生成...")
            try:
                result = model.generate("a simple test", num_inference_steps=1, height=64, width=64)
                print(f"生成测试成功: {result.get('success', False)}")
            except Exception as e:
                print(f"生成测试失败: {e}")
        else:
            print("❌ 模型加载失败")
            
    except Exception as e:
        print(f"❌ 创建模型实例失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("开始模型加载诊断...")
    
    try:
        check_environment()
        check_config()
        check_device_manager()
        test_model_loading()
        
        print("\n" + "=" * 50)
        print("诊断完成")
        print("=" * 50)
        
    except Exception as e:
        print(f"诊断过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 