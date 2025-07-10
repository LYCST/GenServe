#!/usr/bin/env python3
"""
GenServe API认证功能测试脚本
"""

import requests
import json
import time
import sys

# 配置
BASE_URL = "http://localhost:12411"
DEFAULT_API_KEY = "genserve-default-key-2024"

def test_api_call(endpoint, method="GET", headers=None, data=None, expected_status=200):
    """测试API调用"""
    url = f"{BASE_URL}{endpoint}"
    
    if headers is None:
        headers = {}
    
    print(f"\n🔍 测试 {method} {endpoint}")
    print(f"   请求头: {headers}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, data=data)
        else:
            print(f"❌ 不支持的HTTP方法: {method}")
            return False
        
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.text[:200]}...")
        
        if response.status_code == expected_status:
            print(f"✅ 测试通过")
            return True, response.json() if response.text else None
        else:
            print(f"❌ 测试失败，期望状态码: {expected_status}")
            return False, None
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False, None

def main():
    """主函数"""
    print("🔐 GenServe API认证功能测试")
    print("="*60)
    
    # 检查服务是否运行
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 401:
            print("✅ 服务正在运行，认证功能已启用")
        else:
            print("⚠️  服务正在运行，但认证功能可能未启用")
    except requests.exceptions.RequestException:
        print("❌ 无法连接到GenServe服务，请确保服务正在运行")
        print("   启动命令: ./start_optimized.sh")
        sys.exit(1)
    
    # 测试1: 无认证访问（应该失败）
    print("\n" + "="*50)
    print("测试1: 无认证访问")
    test_api_call("/", expected_status=401)
    
    # 测试2: 使用默认API密钥访问
    print("\n" + "="*50)
    print("测试2: 使用默认API密钥访问")
    headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
    test_api_call("/", headers=headers)
    
    # 测试3: 使用X-API-Key头访问
    print("\n" + "="*50)
    print("测试3: 使用X-API-Key头访问")
    headers = {"X-API-Key": DEFAULT_API_KEY}
    test_api_call("/", headers=headers)
    
    # 测试4: 无效API密钥（应该失败）
    print("\n" + "="*50)
    print("测试4: 无效API密钥")
    headers = {"Authorization": "Bearer invalid-key"}
    test_api_call("/", headers=headers, expected_status=401)
    
    # 测试5: 健康检查
    print("\n" + "="*50)
    print("测试5: 健康检查")
    headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
    test_api_call("/health", headers=headers)
    
    # 测试6: 获取API密钥列表
    print("\n" + "="*50)
    print("测试6: 获取API密钥列表")
    headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
    success, response = test_api_call("/auth/keys", headers=headers)
    
    if success and response:
        print(f"   当前密钥数量: {response.get('total', 0)}")
        for key_info in response.get('keys', []):
            print(f"   - {key_info['name']}: {key_info['permissions']} (来源: {key_info.get('source', 'unknown')})")
    
    # 测试7: 生成新的API密钥
    print("\n" + "="*50)
    print("测试7: 生成新的API密钥")
    headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
    data = {
        "name": "测试用户",
        "permissions": "generation,readonly"
    }
    success, response = test_api_call("/auth/generate-key", method="POST", headers=headers, data=data)
    
    new_api_key = None
    if success and response:
        new_api_key = response.get('api_key')
        print(f"   新生成的API密钥: {new_api_key}")
        print(f"   配置字符串: {response.get('config_string')}")
    
    # 测试8: 使用新生成的API密钥
    if new_api_key:
        print("\n" + "="*50)
        print("测试8: 使用新生成的API密钥")
        headers = {"Authorization": f"Bearer {new_api_key}"}
        test_api_call("/", headers=headers)
        
        # 测试新密钥的权限
        print("\n   测试新密钥权限:")
        test_api_call("/health", headers=headers)  # 应该成功
        test_api_call("/status", headers=headers)  # 应该成功
        test_api_call("/auth/keys", headers=headers, expected_status=403)  # 应该失败（没有admin权限）
    
    # 测试9: 再次获取API密钥列表（应该包含新密钥）
    print("\n" + "="*50)
    print("测试9: 再次获取API密钥列表")
    headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
    success, response = test_api_call("/auth/keys", headers=headers)
    
    if success and response:
        print(f"   更新后密钥数量: {response.get('total', 0)}")
        for key_info in response.get('keys', []):
            print(f"   - {key_info['name']}: {key_info['permissions']} (来源: {key_info.get('source', 'unknown')})")
    
    # 测试10: 删除新生成的API密钥
    if new_api_key:
        print("\n" + "="*50)
        print("测试10: 删除新生成的API密钥")
        headers = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
        data = {"api_key": new_api_key}
        success, response = test_api_call("/auth/delete-key", method="POST", headers=headers, data=data)
        
        if success:
            print("   密钥删除成功")
            
            # 验证密钥已被删除
            print("\n   验证密钥已被删除:")
            headers = {"Authorization": f"Bearer {new_api_key}"}
            test_api_call("/", headers=headers, expected_status=401)
    
    print("\n" + "="*60)
    print("🎉 认证功能测试完成")

if __name__ == "__main__":
    main() 