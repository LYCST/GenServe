#!/usr/bin/env python3
"""
图生图测试脚本
测试GenServe的各种图生图功能
"""

import requests
import time
import json
import base64
import io
from PIL import Image
import numpy as np

class Img2ImgTester:
    def __init__(self, base_url="http://localhost:12411"):
        self.base_url = base_url
        
    def create_test_image(self, width=512, height=512, color=(255, 0, 0)):
        """创建测试图片"""
        image = Image.new('RGB', (width, height), color)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def create_test_mask(self, width=512, height=512, mask_region=(100, 100, 300, 300)):
        """创建测试蒙版"""
        mask = Image.new('RGB', (width, height), (0, 0, 0))
        # 在指定区域创建白色蒙版
        mask_array = np.array(mask)
        x1, y1, x2, y2 = mask_region
        mask_array[y1:y2, x1:x2] = [255, 255, 255]
        mask = Image.fromarray(mask_array)
        
        buffer = io.BytesIO()
        mask.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def test_text2img(self):
        """测试文本生成图片"""
        print("🧪 测试文本生成图片...")
        
        payload = {
            "prompt": "A beautiful sunset over mountains, digital art",
            "mode": "text2img",
            "height": 512,
            "width": 512,
            "num_inference_steps": 20,
            "cfg": 3.5,
            "seed": 42
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ 文本生成图片成功")
                    print(f"  GPU: {result.get('gpu_id')}")
                    print(f"  耗时: {result.get('elapsed_time', 0):.2f}秒")
                    print(f"  模式: {result.get('mode')}")
                    return True
                else:
                    print(f"❌ 文本生成图片失败: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return False
    
    def test_img2img(self):
        """测试图片生成图片"""
        print("🧪 测试图片生成图片...")
        
        # 创建测试输入图片
        input_image = self.create_test_image(512, 512, (0, 255, 0))  # 绿色图片
        
        payload = {
            "prompt": "A beautiful landscape with mountains and trees",
            "mode": "img2img",
            "input_image": input_image,
            "strength": 0.7,
            "height": 512,
            "width": 512,
            "num_inference_steps": 20,
            "cfg": 3.5,
            "seed": 42
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ 图片生成图片成功")
                    print(f"  GPU: {result.get('gpu_id')}")
                    print(f"  耗时: {result.get('elapsed_time', 0):.2f}秒")
                    print(f"  模式: {result.get('mode')}")
                    return True
                else:
                    print(f"❌ 图片生成图片失败: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return False
    
    def test_fill(self):
        """测试填充/修复"""
        print("🧪 测试填充/修复...")
        
        # 创建测试输入图片和蒙版
        input_image = self.create_test_image(512, 512, (255, 255, 0))  # 黄色图片
        mask_image = self.create_test_mask(512, 512, (150, 150, 350, 350))  # 中心区域蒙版
        
        payload = {
            "prompt": "A beautiful flower garden in the center",
            "mode": "fill",
            "input_image": input_image,
            "mask_image": mask_image,
            "height": 512,
            "width": 512,
            "num_inference_steps": 20,
            "cfg": 3.5,
            "seed": 42
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ 填充/修复成功")
                    print(f"  GPU: {result.get('gpu_id')}")
                    print(f"  耗时: {result.get('elapsed_time', 0):.2f}秒")
                    print(f"  模式: {result.get('mode')}")
                    return True
                else:
                    print(f"❌ 填充/修复失败: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return False
    
    def test_controlnet(self):
        """测试ControlNet"""
        print("🧪 测试ControlNet...")
        
        # 创建测试输入图片和控制图片
        input_image = self.create_test_image(512, 512, (0, 0, 255))  # 蓝色图片
        control_image = self.create_test_image(512, 512, (128, 128, 128))  # 灰色控制图片
        
        payload = {
            "prompt": "A futuristic city with skyscrapers",
            "mode": "controlnet",
            "input_image": input_image,
            "control_image": control_image,
            "height": 512,
            "width": 512,
            "num_inference_steps": 20,
            "cfg": 3.5,
            "seed": 42
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ ControlNet成功")
                    print(f"  GPU: {result.get('gpu_id')}")
                    print(f"  耗时: {result.get('elapsed_time', 0):.2f}秒")
                    print(f"  模式: {result.get('mode')}")
                    return True
                else:
                    print(f"❌ ControlNet失败: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return False
    
    def test_parameter_validation(self):
        """测试参数验证"""
        print("🧪 测试参数验证...")
        
        # 测试缺少必需参数
        test_cases = [
            {
                "name": "缺少input_image",
                "payload": {
                    "prompt": "test",
                    "mode": "img2img"
                },
                "expected_error": "img2img模式需要提供input_image"
            },
            {
                "name": "缺少mask_image",
                "payload": {
                    "prompt": "test",
                    "mode": "fill",
                    "input_image": "dummy"
                },
                "expected_error": "fill模式需要提供input_image和mask_image"
            },
            {
                "name": "缺少control_image",
                "payload": {
                    "prompt": "test",
                    "mode": "controlnet",
                    "input_image": "dummy"
                },
                "expected_error": "controlnet模式需要提供input_image和control_image"
            }
        ]
        
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/generate",
                    json=test_case["payload"],
                    timeout=30
                )
                
                if response.status_code == 400:
                    result = response.json()
                    if test_case["expected_error"] in result.get("detail", ""):
                        print(f"✅ {test_case['name']} 验证通过")
                    else:
                        print(f"❌ {test_case['name']} 验证失败: 期望 '{test_case['expected_error']}', 实际 '{result.get('detail')}'")
                else:
                    print(f"❌ {test_case['name']} 验证失败: 期望400错误，实际 {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {test_case['name']} 请求异常: {e}")
    
    def run_full_test(self):
        """运行完整测试套件"""
        print("🚀 GenServe 图生图功能测试")
        print("=" * 60)
        
        # 检查服务状态
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code != 200:
                print("❌ 服务不可用")
                return
            print("✅ 服务可用")
        except Exception as e:
            print(f"❌ 服务检查失败: {e}")
            return
        
        print()
        
        # 测试各种模式
        test_results = []
        
        # 1. 文本生成图片
        test_results.append(("text2img", self.test_text2img()))
        print()
        
        # 2. 图片生成图片
        test_results.append(("img2img", self.test_img2img()))
        print()
        
        # 3. 填充/修复
        test_results.append(("fill", self.test_fill()))
        print()
        
        # 4. ControlNet
        test_results.append(("controlnet", self.test_controlnet()))
        print()
        
        # 5. 参数验证
        self.test_parameter_validation()
        print()
        
        # 总结
        print("📊 测试结果总结:")
        success_count = sum(1 for _, success in test_results if success)
        total_count = len(test_results)
        
        for mode, success in test_results:
            status = "✅" if success else "❌"
            print(f"  {status} {mode}: {'通过' if success else '失败'}")
        
        print(f"\n🎯 总体结果: {success_count}/{total_count} 通过")
        
        if success_count == total_count:
            print("🎉 所有图生图功能测试通过!")
        else:
            print("⚠️ 部分功能测试失败，请检查日志")

def main():
    import sys
    
    tester = Img2ImgTester()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "text2img":
            tester.test_text2img()
        elif sys.argv[1] == "img2img":
            tester.test_img2img()
        elif sys.argv[1] == "fill":
            tester.test_fill()
        elif sys.argv[1] == "controlnet":
            tester.test_controlnet()
        elif sys.argv[1] == "validation":
            tester.test_parameter_validation()
        else:
            print("用法: python test_img2img.py [text2img | img2img | fill | controlnet | validation]")
    else:
        tester.run_full_test()

if __name__ == "__main__":
    main() 