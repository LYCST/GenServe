#!/usr/bin/env python3
"""
演示API密钥删除功能
"""

import requests
import json

# 配置
BASE_URL = "http://localhost:12411"
ADMIN_KEY = "genserve-default-key-2024"

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def main():
    """主函数"""
    print("🔑 API密钥删除功能演示")
    
    # 1. 查看当前所有密钥
    print_separator("1. 查看当前所有API密钥")
    
    response = requests.get(f"{BASE_URL}/auth/keys", 
                          headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    
    if response.status_code == 200:
        data = response.json()
        print(f"当前密钥数量: {data.get('total', 0)}")
        
        for key_info in data.get('keys', []):
            print(f"\n密钥信息:")
            print(f"  - key_id: {key_info.get('key_id', 'N/A')}")
            print(f"  - 用户名: {key_info.get('name', 'N/A')}")
            print(f"  - 权限: {key_info.get('permissions', [])}")
            print(f"  - 来源: {key_info.get('source', 'unknown')}")
            print(f"  - 可删除: {'是' if key_info.get('can_delete', False) else '否'}")
            print(f"  - 预览: {key_info.get('key_preview', 'N/A')}")
    else:
        print(f"获取密钥列表失败: {response.status_code}")
        return
    
    # 2. 生成一个新密钥用于演示删除
    print_separator("2. 生成新密钥用于演示")
    
    response = requests.post(f"{BASE_URL}/auth/generate-key",
                           headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                           data={"name": "演示用户", "permissions": "generation"})
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            new_key = data.get('api_key')
            print(f"新生成的密钥: {new_key}")
            print(f"配置字符串: {data.get('config_string')}")
        else:
            print(f"生成密钥失败: {data.get('error')}")
            return
    else:
        print(f"生成密钥请求失败: {response.status_code}")
        return
    
    # 3. 再次查看密钥列表，找到新密钥的key_id
    print_separator("3. 查看新密钥的key_id")
    
    response = requests.get(f"{BASE_URL}/auth/keys", 
                          headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    
    if response.status_code == 200:
        data = response.json()
        target_key_id = None
        
        for key_info in data.get('keys', []):
            if key_info.get('name') == '演示用户':
                target_key_id = key_info.get('key_id')
                print(f"找到演示用户的key_id: {target_key_id}")
                break
        
        if not target_key_id:
            print("未找到演示用户的key_id")
            return
    else:
        print(f"获取密钥列表失败: {response.status_code}")
        return
    
    # 4. 使用key_id删除密钥
    print_separator("4. 使用key_id删除密钥")
    
    response = requests.post(f"{BASE_URL}/auth/delete-key-by-id",
                           headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                           data={"key_id": target_key_id})
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print(f"✅ 删除成功: {data.get('message')}")
            deleted_keys = data.get('deleted_keys', [])
            for key in deleted_keys:
                print(f"  - 删除的密钥: {key.get('name')} ({key.get('key_id')})")
        else:
            print(f"❌ 删除失败: {data.get('error')}")
    else:
        print(f"删除请求失败: {response.status_code}")
    
    # 5. 验证密钥已被删除
    print_separator("5. 验证密钥已被删除")
    
    response = requests.get(f"{BASE_URL}/auth/keys", 
                          headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    
    if response.status_code == 200:
        data = response.json()
        print(f"删除后密钥数量: {data.get('total', 0)}")
        
        # 检查演示用户是否还存在
        demo_user_exists = False
        for key_info in data.get('keys', []):
            if key_info.get('name') == '演示用户':
                demo_user_exists = True
                break
        
        if demo_user_exists:
            print("❌ 演示用户仍然存在")
        else:
            print("✅ 演示用户已成功删除")
    
    print_separator("演示完成")

if __name__ == "__main__":
    main() 