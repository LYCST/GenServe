#!/usr/bin/env python3
"""
GenServe API测试脚本
调用图片生成API并保存返回的图片
"""

import requests
import base64
import json
from datetime import datetime
import os

def test_generate_image(prompt, save_path=None):
    """测试图片生成API"""
    
    # API端点
    url = "http://localhost:12411/generate"
    
    # 请求参数
    payload = {
        "prompt": prompt,
        "model": "flux1-dev",  # 需要指定模型
        "num_inference_steps": 20,  # 减少步数以加快生成
        "height": 1024,
        "width": 1024,
        "seed": 42
    }
    
    print(f"正在生成图片...")
    print(f"提示词: {prompt}")
    print(f"参数: {payload}")
    
    try:
        # 发送POST请求
        response = requests.post(url, json=payload, timeout=300)  # 5分钟超时
        
        if response.status_code == 200:
            result = response.json()
            
            # API返回的格式：{"message": "...", "model": "...", "elapsed_time": "...", "output": "base64...", "save_to_disk": bool, "device": "..."}
            print(f"✅ 图片生成成功！")
            print(f"消息: {result.get('message', '')}")
            print(f"模型: {result.get('model', '')}")
            print(f"耗时: {result.get('elapsed_time', '')}")
            print(f"使用设备: {result.get('device', 'unknown')}")
            
            # 获取base64图片数据
            base64_image = result.get("output")
            if base64_image:
                # 解码base64并保存图片
                image_data = base64.b64decode(base64_image)
                
                # 生成文件名
                if not save_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"generated_image_{timestamp}.png"
                
                # 保存图片
                with open(save_path, "wb") as f:
                    f.write(image_data)
                
                print(f"📸 图片已保存到: {os.path.abspath(save_path)}")
                return save_path
            else:
                print("❌ 响应中没有图片数据")
                print(f"完整响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ API请求失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时，图片生成可能需要更长时间")
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败，请确保服务正在运行")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    
    return None

def test_health_check():
    """测试健康检查API"""
    try:
        response = requests.get("http://localhost:12411/health")
        if response.status_code == 200:
            result = response.json()
            print("🟢 服务健康状态:")
            print(f"  状态: {result.get('status')}")
            print(f"  模型状态: {result.get('models')}")
            print(f"  设备信息: {result.get('device_info')}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("GenServe API 测试")
    print("=" * 50)
    
    # 首先检查服务状态
    print("\n1. 检查服务状态...")
    if not test_health_check():
        print("服务未正常运行，请先启动服务")
        exit(1)
    
    # 测试图片生成
    print("\n2. 测试图片生成...")
    
    # 单次测试
    prompt = "a beautiful landscape with mountains and lakes"
    print(f"\n--- 开始生成图片 ---")
    image_path = test_generate_image(prompt)
    if image_path:
        print(f"✅ 成功！可以使用图片查看器打开: {image_path}")
    else:
        print("❌ 生成失败")
    
    print("\n✨ 测试完成！") 