# GenServe API认证功能说明

## 概述

GenServe现在支持基于API密钥的认证系统，确保只有授权用户才能访问API接口。

## 🔐 认证方式

### 1. Authorization Header (推荐)
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:12411/
```

### 2. X-API-Key Header
```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:12411/
```

## 🎯 权限级别

| 权限 | 说明 | 可访问的接口 |
|------|------|-------------|
| `generation` | 图片生成权限 | `/generate`, `/generate/upload` |
| `readonly` | 只读权限 | `/`, `/health`, `/status`, `/models`, `/loras`, `/task/{id}` |
| `admin` | 管理员权限 | 所有接口 + `/auth/keys`, `/auth/generate-key` |
| `all` | 所有权限 | 所有接口 |

## ⚙️ 配置API密钥

### 方法1: 环境变量配置

在启动脚本中配置API密钥：

```bash
# 格式：key:name:permissions
export API_KEY_1="abc123def456:developer:generation,readonly"
export API_KEY_2="xyz789ghi012:user:generation"
export API_KEY_3="admin123admin456:admin:all"
```

### 方法2: 使用配置示例文件

1. 复制配置示例文件：
```bash
cp auth_config_example.sh auth_config.sh
```

2. 修改密钥配置：
```bash
vim auth_config.sh
```

3. 在启动脚本中引用：
```bash
source auth_config.sh
```

### 方法3: 使用默认密钥

如果未配置任何API密钥，系统将使用默认密钥：
- **默认密钥**: `genserve-default-key-2024`
- **权限**: `all`

## 🔧 管理API密钥

### 获取API密钥列表（仅管理员）
```bash
curl -X GET "http://localhost:12411/auth/keys" \
  -H "Authorization: Bearer ADMIN_API_KEY"
```

### 生成新的API密钥（仅管理员）
```bash
curl -X POST "http://localhost:12411/auth/generate-key" \
  -H "Authorization: Bearer ADMIN_API_KEY" \
  -F "name=新用户" \
  -F "permissions=generation,readonly"
```

## 🛡️ 安全特性

### 1. 速率限制
- 每个API密钥每分钟最多100个请求
- 超过限制会返回429状态码

### 2. 使用统计
- 记录每个API密钥的使用次数
- 记录最后使用时间
- 管理员可以查看使用统计

### 3. 权限验证
- 每个接口都有相应的权限要求
- 权限不足会返回403状态码

## 📝 使用示例

### 基础认证测试
```bash
# 无认证访问（会失败）
curl http://localhost:12411/

# 使用默认密钥访问
curl -H "Authorization: Bearer genserve-default-key-2024" http://localhost:12411/

# 使用X-API-Key头访问
curl -H "X-API-Key: genserve-default-key-2024" http://localhost:12411/
```

### 图片生成（需要generation权限）
```bash
curl -X POST "http://localhost:12411/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "一只可爱的小猫",
    "mode": "text2img"
  }'
```

### 查看服务状态（需要readonly权限）
```bash
curl -X GET "http://localhost:12411/status" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 管理API密钥（需要admin权限）
```bash
# 获取密钥列表
curl -X GET "http://localhost:12411/auth/keys" \
  -H "Authorization: Bearer ADMIN_API_KEY"

# 生成新密钥
curl -X POST "http://localhost:12411/auth/generate-key" \
  -H "Authorization: Bearer ADMIN_API_KEY" \
  -F "name=新用户" \
  -F "permissions=generation,readonly"
```

## 🧪 测试认证功能

运行认证功能测试脚本：

```bash
python test_auth.py
```

测试脚本会验证：
- 无认证访问被拒绝
- 有效API密钥可以正常访问
- 无效API密钥被拒绝
- 不同认证方式都能正常工作

## 🔍 错误处理

### 常见错误码

| 状态码 | 错误 | 说明 |
|--------|------|------|
| 401 | Unauthorized | 缺少或无效的API密钥 |
| 403 | Forbidden | 权限不足 |
| 429 | Too Many Requests | 请求频率过高 |

### 错误响应示例

```json
{
  "detail": "缺少API密钥。请在Authorization header中使用Bearer token或在X-API-Key header中提供密钥"
}
```

## 📊 监控和日志

### 认证日志
系统会记录所有认证相关的活动：
- API密钥验证成功/失败
- 权限检查结果
- 速率限制触发

### 使用统计
管理员可以查看：
- 每个API密钥的使用次数
- 最后使用时间
- 权限分配情况

## 🔒 安全建议

1. **使用强密码**: 生成足够复杂的API密钥
2. **定期轮换**: 定期更换API密钥
3. **权限最小化**: 只分配必要的权限
4. **监控使用**: 定期检查API密钥使用情况
5. **及时清理**: 删除不再使用的API密钥
6. **环境隔离**: 为不同环境使用不同的API密钥

## 🚀 快速开始

1. **启动服务**:
```bash
./start_optimized.sh
```

2. **测试认证**:
```bash
python test_auth.py
```

3. **使用API**:
```bash
curl -H "Authorization: Bearer genserve-default-key-2024" http://localhost:12411/
```

## 📚 相关文档

- [API Curl请求指南](API_CURL_GUIDE.md) - 完整的API使用文档
- [并行使用指南](PARALLEL_GUIDE.md) - 并行处理说明
- [配置示例](auth_config_example.sh) - API密钥配置示例 