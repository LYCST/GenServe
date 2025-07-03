#!/usr/bin/env python3
"""
上传方式性能对比测试脚本
对比base64和form-data两种图片上传方式的性能差异
"""

import requests
import time
import json
import base64
import io
from PIL import Image
import os
import statistics

class UploadMethodTester:
    def __init__(self, base_url="http://localhost:12411"):
        self.base_url = base_url
        
    def create_test_image(self, width=1024, height=1024, color=(255, 0, 0)):
        """创建测试图片"""
        image = Image.new('RGB', (width, height), color)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_base64_upload(self, image_data, prompt="A beautiful landscape"):
        """测试base64上传方式"""
        start_time = time.time()
        
        # 转换为base64
        image_base64 = base64.b64encode(image_data).decode()
        
        payload = {
            "prompt": prompt,
            "mode": "img2img",
            "input_image": image_base64,
            "strength": 0.7,
            "height": 1024,
            "width": 1024,
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
            
            end_time = time.time()
            upload_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return {
                        "success": True,
                        "upload_time": upload_time,
                        "total_time": result.get('elapsed_time', 0),
                        "data_size": len(image_base64),
                        "original_size": len(image_data)
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get('error'),
                        "upload_time": upload_time
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "upload_time": upload_time
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "upload_time": time.time() - start_time
            }
    
    def test_form_data_upload(self, image_data, prompt="A beautiful landscape"):
        """测试form-data上传方式"""
        start_time = time.time()
        
        files = {
            'input_image': ('test_image.png', image_data, 'image/png')
        }
        
        data = {
            "prompt": prompt,
            "mode": "img2img",
            "strength": "0.7",
            "height": "1024",
            "width": "1024",
            "num_inference_steps": "20",
            "cfg": "3.5",
            "seed": "42"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate/img2img",
                files=files,
                data=data,
                timeout=180
            )
            
            end_time = time.time()
            upload_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return {
                        "success": True,
                        "upload_time": upload_time,
                        "total_time": result.get('elapsed_time', 0),
                        "data_size": len(image_data),
                        "original_size": len(image_data)
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get('error'),
                        "upload_time": upload_time
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "upload_time": upload_time
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "upload_time": time.time() - start_time
            }
    
    def run_performance_comparison(self, image_sizes=[(512, 512), (1024, 1024), (2048, 2048)]):
        """运行性能对比测试"""
        print("🚀 上传方式性能对比测试")
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
        
        results = {
            "base64": [],
            "form_data": []
        }
        
        for width, height in image_sizes:
            print(f"📊 测试图片尺寸: {width}x{height}")
            
            # 创建测试图片
            image_data = self.create_test_image(width, height)
            original_size = len(image_data)
            base64_size = len(base64.b64encode(image_data).decode())
            
            print(f"  原始大小: {original_size / 1024:.1f}KB")
            print(f"  Base64大小: {base64_size / 1024:.1f}KB")
            print(f"  大小增加: {((base64_size - original_size) / original_size * 100):.1f}%")
            
            # 测试base64上传
            print("  🔄 测试base64上传...")
            base64_result = self.test_base64_upload(image_data)
            if base64_result["success"]:
                results["base64"].append({
                    "size": original_size,
                    "upload_time": base64_result["upload_time"],
                    "total_time": base64_result["total_time"]
                })
                print(f"    ✅ 成功 - 上传时间: {base64_result['upload_time']:.2f}秒")
            else:
                print(f"    ❌ 失败: {base64_result['error']}")
            
            # 测试form-data上传
            print("  🔄 测试form-data上传...")
            form_result = self.test_form_data_upload(image_data)
            if form_result["success"]:
                results["form_data"].append({
                    "size": original_size,
                    "upload_time": form_result["upload_time"],
                    "total_time": form_result["total_time"]
                })
                print(f"    ✅ 成功 - 上传时间: {form_result['upload_time']:.2f}秒")
            else:
                print(f"    ❌ 失败: {form_result['error']}")
            
            print()
        
        # 分析结果
        self.analyze_results(results)
    
    def analyze_results(self, results):
        """分析测试结果"""
        print("📈 性能分析结果")
        print("=" * 60)
        
        for method, data in results.items():
            if not data:
                print(f"❌ {method} 没有成功数据")
                continue
            
            sizes = [d["size"] for d in data]
            upload_times = [d["upload_time"] for d in data]
            total_times = [d["total_time"] for d in data]
            
            print(f"\n🔍 {method.upper()} 方式:")
            print(f"  测试次数: {len(data)}")
            print(f"  平均图片大小: {statistics.mean(sizes) / 1024:.1f}KB")
            print(f"  平均上传时间: {statistics.mean(upload_times):.2f}秒")
            print(f"  平均总时间: {statistics.mean(total_times):.2f}秒")
            print(f"  上传时间标准差: {statistics.stdev(upload_times):.2f}秒")
        
        # 对比分析
        if results["base64"] and results["form_data"]:
            print(f"\n⚖️ 性能对比:")
            
            base64_avg_upload = statistics.mean([d["upload_time"] for d in results["base64"]])
            form_avg_upload = statistics.mean([d["upload_time"] for d in results["form_data"]])
            
            if base64_avg_upload > form_avg_upload:
                improvement = ((base64_avg_upload - form_avg_upload) / base64_avg_upload) * 100
                print(f"  Form-data上传速度比Base64快 {improvement:.1f}%")
            else:
                improvement = ((form_avg_upload - base64_avg_upload) / form_avg_upload) * 100
                print(f"  Base64上传速度比Form-data快 {improvement:.1f}%")
            
            print(f"\n💡 建议:")
            if base64_avg_upload > form_avg_upload:
                print("  ✅ 推荐使用Form-data方式上传图片")
                print("  📝 优势: 更快的上传速度，更少的内存占用")
            else:
                print("  ✅ 两种方式性能相近，可根据具体需求选择")
                print("  📝 Base64优势: 简单易用，适合小图片")
                print("  📝 Form-data优势: 性能更好，适合大图片")
    
    def test_simple_comparison(self):
        """简单对比测试"""
        print("🧪 简单性能对比测试")
        print("=" * 40)
        
        # 创建1024x1024测试图片
        image_data = self.create_test_image(1024, 1024)
        
        print("测试Base64上传...")
        base64_result = self.test_base64_upload(image_data)
        
        print("测试Form-data上传...")
        form_result = self.test_form_data_upload(image_data)
        
        print("\n📊 对比结果:")
        if base64_result["success"] and form_result["success"]:
            print(f"Base64上传时间: {base64_result['upload_time']:.2f}秒")
            print(f"Form-data上传时间: {form_result['upload_time']:.2f}秒")
            
            if base64_result['upload_time'] > form_result['upload_time']:
                improvement = ((base64_result['upload_time'] - form_result['upload_time']) / base64_result['upload_time']) * 100
                print(f"Form-data快 {improvement:.1f}%")
            else:
                improvement = ((form_result['upload_time'] - base64_result['upload_time']) / form_result['upload_time']) * 100
                print(f"Base64快 {improvement:.1f}%")
        else:
            print("❌ 测试失败")
            if not base64_result["success"]:
                print(f"Base64错误: {base64_result['error']}")
            if not form_result["success"]:
                print(f"Form-data错误: {form_result['error']}")

def main():
    import sys
    
    tester = UploadMethodTester()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "simple":
            tester.test_simple_comparison()
        elif sys.argv[1] == "full":
            tester.run_performance_comparison()
        else:
            print("用法: python test_upload_methods.py [simple | full]")
    else:
        tester.test_simple_comparison()

if __name__ == "__main__":
    main() 