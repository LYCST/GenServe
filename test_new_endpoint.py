#!/usr/bin/env python3
"""
测试新的/generate/img2img端点
"""

import requests
import time
import base64
import io
from PIL import Image

def test_new_endpoint():
    """测试新的/generate/img2img端点"""
    base_url = "http://localhost:12411"
    
    print("🧪 测试新的 /generate/img2img 端点")
    print("=" * 50)
    
    # 1. 检查服务状态
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ 服务可用")
            print(f"  版本: {result.get('version')}")
            print(f"  端点: {result.get('endpoints')}")
        else:
            print("❌ 服务不可用")
            return
    except Exception as e:
        print(f"❌ 服务检查失败: {e}")
        return
    
    print()
    
    # 2. 创建测试图片
    print("📸 创建测试图片...")
    image = Image.new('RGB', (512, 512), (255, 0, 0))  # 红色图片
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    
    print(f"  图片大小: {len(image_data) / 1024:.1f}KB")
    
    # 3. 测试Form-data上传
    print("\n🔄 测试Form-data上传...")
    
    files = {
        'input_image': ('test_image.png', image_data, 'image/png')
    }
    
    data = {
        "prompt": "A beautiful landscape with mountains",
        "mode": "img2img",
        "strength": "0.7",
        "height": "512",
        "width": "512",
        "num_inference_steps": "20",
        "cfg": "3.5",
        "seed": "42"
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/generate/img2img",
            files=files,
            data=data,
            timeout=180
        )
        
        end_time = time.time()
        upload_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Form-data上传成功!")
                print(f"  上传时间: {upload_time:.2f}秒")
                print(f"  生成时间: {result.get('elapsed_time', 0):.2f}秒")
                print(f"  GPU: {result.get('gpu_id')}")
                print(f"  模式: {result.get('mode')}")
                
                # 保存生成的图片
                if result.get('image_base64'):
                    image_data = base64.b64decode(result['image_base64'])
                    generated_image = Image.open(io.BytesIO(image_data))
                    generated_image.save("test_result.png")
                    print("  结果图片已保存为: test_result.png")
                
                return True
            else:
                print(f"❌ 生成失败: {result.get('error')}")
                return False
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"  响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_json_endpoint():
    """测试JSON端点作为对比"""
    base_url = "http://localhost:12411"
    
    print("\n🔄 测试JSON端点作为对比...")
    
    # 创建测试图片
    image = Image.new('RGB', (512, 512), (0, 255, 0))  # 绿色图片
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    
    # 转换为base64
    image_base64 = base64.b64encode(image_data).decode()
    
    payload = {
        "prompt": "A beautiful landscape with mountains",
        "mode": "img2img",
        "input_image": image_base64,
        "strength": 0.7,
        "height": 512,
        "width": 512,
        "num_inference_steps": 20,
        "cfg": 3.5,
        "seed": 42
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=180
        )
        
        end_time = time.time()
        upload_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ JSON上传成功!")
                print(f"  上传时间: {upload_time:.2f}秒")
                print(f"  生成时间: {result.get('elapsed_time', 0):.2f}秒")
                print(f"  GPU: {result.get('gpu_id')}")
                print(f"  模式: {result.get('mode')}")
                return True
            else:
                print(f"❌ 生成失败: {result.get('error')}")
                return False
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def main():
    print("🚀 GenServe 新端点测试")
    print("=" * 60)
    
    # 测试新端点
    success1 = test_new_endpoint()
    
    # 测试JSON端点作为对比
    success2 = test_json_endpoint()
    
    print("\n📊 测试结果总结:")
    print(f"  Form-data端点 (/generate/img2img): {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"  JSON端点 (/generate): {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有端点测试通过!")
    else:
        print("\n⚠️ 部分端点测试失败，请检查服务状态")

if __name__ == "__main__":
    main() 