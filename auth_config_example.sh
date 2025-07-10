#!/bin/bash
# GenServe API密钥配置示例
# 复制此文件并根据需要修改

# =============================================================================
# API认证配置示例
# =============================================================================

# 默认API密钥（如果未配置其他密钥，将使用此密钥）
export DEFAULT_API_KEY="genserve-default-key-2024"

# API密钥配置（格式：key:name:permissions）
# 支持最多10个API密钥

# 示例1: 开发者密钥（生成和只读权限）
export API_KEY_1="dev123abc456:developer:generation,readonly"

# 示例2: 普通用户密钥（只有生成权限）
export API_KEY_2="user789xyz012:user:generation"

# 示例3: 管理员密钥（所有权限）
export API_KEY_3="admin456def789:admin:all"

# 示例4: 只读用户密钥（只有只读权限）
export API_KEY_4="readonly123:readonly_user:readonly"

# 示例5: 高级用户密钥（生成、只读和管理权限）
export API_KEY_5="advanced789:advanced_user:generation,readonly,admin"

# =============================================================================
# 权限说明
# =============================================================================

# generation: 图片生成权限
#   - 可以访问 /generate 和 /generate/upload 接口
#   - 可以提交图片生成任务

# readonly: 只读权限
#   - 可以访问 /, /health, /status, /models, /loras, /task/{id} 接口
#   - 可以查看服务状态、模型列表、任务结果等

# admin: 管理员权限
#   - 包含所有权限
#   - 可以访问 /auth/keys 和 /auth/generate-key 接口
#   - 可以管理API密钥

# all: 所有权限
#   - 等同于 admin 权限

# =============================================================================
# 使用方法
# =============================================================================

# 1. 复制此文件为 auth_config.sh
# cp auth_config_example.sh auth_config.sh

# 2. 修改密钥配置
# vim auth_config.sh

# 3. 在启动脚本中引用
# source auth_config.sh

# 4. 启动服务
# ./start_optimized.sh

# =============================================================================
# 安全建议
# =============================================================================

# 1. 使用强密码生成API密钥
# 2. 定期轮换API密钥
# 3. 为不同用户分配不同权限
# 4. 监控API密钥使用情况
# 5. 及时删除不再使用的密钥

# =============================================================================
# 生成API密钥的Python脚本示例
# =============================================================================

cat << 'EOF' > generate_api_key.py
#!/usr/bin/env python3
import hashlib
import time
import random
import string

def generate_api_key(name, permissions):
    """生成API密钥"""
    timestamp = str(int(time.time()))
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    base_string = f"{name}:{timestamp}:{random_str}:{','.join(permissions)}"
    
    # 使用SHA256生成密钥
    api_key = hashlib.sha256(base_string.encode()).hexdigest()[:32]
    
    return api_key

if __name__ == "__main__":
    name = input("请输入用户名: ")
    permissions = input("请输入权限（用逗号分隔，如：generation,readonly）: ").split(",")
    
    api_key = generate_api_key(name, permissions)
    config_string = f"{api_key}:{name}:{','.join(permissions)}"
    
    print(f"\n生成的API密钥配置:")
    print(f"export API_KEY_X=\"{config_string}\"")
    print(f"\nAPI密钥: {api_key}")
    print(f"用户名: {name}")
    print(f"权限: {permissions}")
EOF

echo "API密钥配置示例文件已创建"
echo "请根据实际需要修改密钥配置" 