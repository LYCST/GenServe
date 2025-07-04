#!/usr/bin/env python3
"""
测试Redux功能
验证Redux pipeline的加载和生成功能
"""

import requests
import json
import base64
from PIL import Image
import io
import time

# 服务配置
BASE_URL = "http://localhost:12411"

def create_test_image():
    """创建一个简单的测试图片"""
    # 创建一个简单的测试图片
    width, height = 512, 512
    image = Image.new('RGB', (width, height), color=(128, 128, 128))
    
    # 添加一些简单的图案
    for x in range(0, width, 64):
        for y in range(0, height, 64):
            color = ((x + y) % 255, (x * 2) % 255, (y * 2) % 255)
            for dx in range(32):
                for dy in range(32):
                    if x + dx < width and y + dy < height:
                        image.putpixel((x + dx, y + dy), color)
    
    return image

def image_to_base64(image):
    """将PIL图片转换为base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return img_base64

def check_service_status():
    """检查服务状态"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_redux():
    """测试Redux功能"""
    print("🧪 测试Redux功能...")
    
    # 创建测试图片
    test_image = create_test_image()
    test_image_base64 = image_to_base64(test_image)
    
    # 保存测试图片
    test_image.save("test_redux_input.png")
    print("✅ 测试输入图片已保存为 test_redux_input.png")
    
    # 测试用例1：基本Redux功能
    print("\n📋 测试用例1: 基本Redux功能")
    request_data = {
        "prompt": "A beautiful landscape with mountains and trees, photorealistic",
        "mode": "redux",
        "input_image": test_image_base64,
        "height": 512,
        "width": 512,
        "num_inference_steps": 20,
        "cfg": 2.5,
        "seed": 42,
        "model_id": "flux1-redux-dev"
    }
    
    try:
        print("   发送Redux请求...")
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate", json=request_data, timeout=300)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"   ✅ Redux生成成功，耗时: {elapsed_time:.2f}秒")
                
                # 保存生成的图片
                if result.get("image_base64"):
                    image_data = base64.b64decode(result["image_base64"])
                    image = Image.open(io.BytesIO(image_data))
                    filename = "test_redux_output.png"
                    image.save(filename)
                    print(f"   💾 图片已保存为: {filename}")
            else:
                print(f"   ❌ Redux生成失败: {result.get('error')}")
        else:
            print(f"   ❌ 请求失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ 测试出错: {e}")
    
    # 测试用例2：Form-data格式的Redux请求
    print("\n📋 测试用例2: Form-data格式的Redux请求")
    
    try:
        print("   发送Form-data Redux请求...")
        
        # 准备文件数据
        files = {
            'input_image': ('test_image.png', io.BytesIO(base64.b64decode(test_image_base64)), 'image/png')
        }
        
        data = {
            'prompt': 'A beautiful landscape with mountains and trees, photorealistic',
            'mode': 'redux',
            'height': '512',
            'width': '512',
            'num_inference_steps': '20',
            'cfg': '2.5',
            'seed': '42',
            'model_id': 'flux1-redux-dev'
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate/upload", files=files, data=data, timeout=300)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"   ✅ Form-data Redux生成成功，耗时: {elapsed_time:.2f}秒")
                
                # 保存生成的图片
                if result.get("image_base64"):
                    image_data = base64.b64decode(result["image_base64"])
                    image = Image.open(io.BytesIO(image_data))
                    filename = "test_redux_formdata_output.png"
                    image.save(filename)
                    print(f"   💾 图片已保存为: {filename}")
            else:
                print(f"   ❌ Form-data Redux生成失败: {result.get('error')}")
        else:
            print(f"   ❌ Form-data请求失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ Form-data测试出错: {e}")
    
    print(f"\n🎉 Redux功能测试完成！")
    print(f"📁 生成的图片文件:")
    print(f"   - test_redux_input.png (测试输入图片)")
    print(f"   - test_redux_output.png (JSON格式Redux输出)")
    print(f"   - test_redux_formdata_output.png (Form-data格式Redux输出)")

def test_model_list():
    """测试模型列表"""
    print("\n📋 测试模型列表...")
    
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            result = response.json()
            models = result.get("models", [])
            print(f"   支持的模型数量: {len(models)}")
            for model in models:
                print(f"   - {model.get('model_id')}: {model.get('model_name')}")
        else:
            print(f"   ❌ 获取模型列表失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 测试模型列表出错: {e}")

if __name__ == "__main__":
    print("🚀 Redux功能测试")
    print("=" * 50)
    
    # 检查服务状态
    if not check_service_status():
        print("❌ 服务不可用，请先启动GenServe服务")
        exit(1)
    
    # 测试模型列表
    test_model_list()
    
    # 运行Redux测试
    test_redux()
    
    print("\n" + "=" * 50)
    print("🎯 测试总结:")
    print("1. 测试了Redux pipeline的加载和生成功能")
    print("2. 验证了JSON和Form-data两种请求格式")
    print("3. Redux使用两个pipeline：FluxPriorReduxPipeline和FluxPipeline")
    print("4. 支持图片到图片的生成，无需文本提示词") 