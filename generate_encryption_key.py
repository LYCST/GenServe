#!/usr/bin/env python3
"""
生成API密钥文件加密密钥的工具
"""

import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generate_encryption_key():
    """生成新的加密密钥"""
    key = Fernet.generate_key()
    return key.decode()

def generate_key_from_password(password: str, salt: bytes = None):
    """从密码生成加密密钥"""
    if salt is None:
        salt = b"genserve_salt_2024"  # 默认盐值
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key.decode()

def main():
    """主函数"""
    print("🔐 API密钥文件加密密钥生成工具")
    print("="*50)
    
    print("\n选择生成方式:")
    print("1. 生成随机加密密钥（推荐）")
    print("2. 从密码生成加密密钥")
    print("3. 从默认API密钥生成加密密钥")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == "1":
        # 生成随机密钥
        encryption_key = generate_encryption_key()
        print(f"\n✅ 生成的随机加密密钥:")
        print(f"export API_KEYS_ENCRYPTION_KEY=\"{encryption_key}\"")
        
    elif choice == "2":
        # 从密码生成
        password = input("请输入密码: ").strip()
        if not password:
            print("❌ 密码不能为空")
            return
        
        encryption_key = generate_key_from_password(password)
        print(f"\n✅ 从密码生成的加密密钥:")
        print(f"export API_KEYS_ENCRYPTION_KEY=\"{encryption_key}\"")
        print(f"\n⚠️  请记住您的密码，用于后续生成相同的密钥")
        
    elif choice == "3":
        # 从默认API密钥生成
        default_key = os.getenv("DEFAULT_API_KEY", "genserve-default-key-2024")
        encryption_key = generate_key_from_password(default_key)
        print(f"\n✅ 从默认API密钥生成的加密密钥:")
        print(f"export API_KEYS_ENCRYPTION_KEY=\"{encryption_key}\"")
        print(f"\n⚠️  此密钥基于默认API密钥生成，如果默认密钥改变，此加密密钥也会改变")
        
    else:
        print("❌ 无效选择")
        return
    
    print(f"\n📝 使用说明:")
    print(f"1. 将上述export命令添加到您的启动脚本中")
    print(f"2. 重启GenServe服务")
    print(f"3. 系统会自动将现有的明文api_keys.json文件转换为加密的api_keys.enc文件")
    print(f"4. 明文文件会被自动删除")
    
    print(f"\n🔒 安全建议:")
    print(f"- 将加密密钥保存在安全的地方")
    print(f"- 不要将加密密钥提交到版本控制系统")
    print(f"- 定期更换加密密钥")
    print(f"- 备份加密密钥，丢失后将无法解密API密钥文件")

if __name__ == "__main__":
    main() 