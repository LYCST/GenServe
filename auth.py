import os
import logging
import hashlib
import time
import json
import base64
from typing import Optional, Dict, List
from fastapi import HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class AuthManager:
    """API认证管理器"""
    
    def __init__(self):
        self.api_keys_file = "api_keys.json"  # API密钥持久化文件
        self.encrypted_file = "api_keys.enc"  # 加密后的文件
        self.fernet = self._init_encryption()
        self.api_keys = self._load_api_keys()
        self.rate_limits = {}  # 简单的速率限制存储
        self.security = HTTPBearer()
    
    def _init_encryption(self) -> Optional[Fernet]:
        """初始化加密功能"""
        try:
            # 从环境变量获取加密密钥
            encryption_key = os.getenv("API_KEYS_ENCRYPTION_KEY")
            
            if not encryption_key:
                # 如果没有设置加密密钥，生成一个基于默认密钥的加密密钥
                default_key = os.getenv("DEFAULT_API_KEY", "genserve-default-key-2024")
                salt = b"genserve_salt_2024"  # 固定盐值
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(default_key.encode()))
                fernet = Fernet(key)
                
                logger.info("使用基于默认密钥生成的加密密钥")
            else:
                # 使用环境变量中的加密密钥
                if len(encryption_key) != 44:  # Fernet密钥长度
                    logger.error("加密密钥长度不正确，应为44个字符")
                    return None
                
                fernet = Fernet(encryption_key.encode())
                logger.info("使用环境变量中的加密密钥")
            
            return fernet
        except Exception as e:
            logger.error(f"初始化加密功能失败: {e}")
            return None
    
    def _encrypt_data(self, data: str) -> Optional[bytes]:
        """加密数据"""
        if not self.fernet:
            logger.error("加密功能未初始化")
            return None
        
        try:
            return self.fernet.encrypt(data.encode('utf-8'))
        except Exception as e:
            logger.error(f"加密数据失败: {e}")
            return None
    
    def _decrypt_data(self, encrypted_data: bytes) -> Optional[str]:
        """解密数据"""
        if not self.fernet:
            logger.error("加密功能未初始化")
            return None
        
        try:
            decrypted = self.fernet.decrypt(encrypted_data)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"解密数据失败: {e}")
            return None
    
    def _save_encrypted_api_keys(self, file_keys: Dict):
        """保存加密的API密钥"""
        try:
            if not self.fernet:
                logger.warning("加密功能未初始化，使用明文保存")
                with open(self.api_keys_file, 'w', encoding='utf-8') as f:
                    json.dump(file_keys, f, indent=2, ensure_ascii=False)
                return True
            
            # 加密数据
            json_data = json.dumps(file_keys, ensure_ascii=False, indent=2)
            encrypted_data = self._encrypt_data(json_data)
            
            if encrypted_data is None:
                logger.error("加密失败，无法保存API密钥")
                return False
            
            # 保存加密文件
            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            
            # 删除明文文件（如果存在）
            if os.path.exists(self.api_keys_file):
                os.remove(self.api_keys_file)
                logger.info("已删除明文API密钥文件")
            
            logger.info(f"API密钥已加密保存到文件: {self.encrypted_file}")
            return True
        except Exception as e:
            logger.error(f"保存加密API密钥失败: {e}")
            return False
    
    def _load_encrypted_api_keys(self) -> Dict:
        """加载加密的API密钥"""
        try:
            # 优先尝试加载加密文件
            if os.path.exists(self.encrypted_file) and self.fernet:
                with open(self.encrypted_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self._decrypt_data(encrypted_data)
                if decrypted_data:
                    file_keys = json.loads(decrypted_data)
                    logger.info(f"从加密文件加载API密钥: {self.encrypted_file}")
                    return file_keys
                else:
                    logger.error("解密API密钥文件失败")
            
            # 如果加密文件不存在或解密失败，尝试加载明文文件
            if os.path.exists(self.api_keys_file):
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    file_keys = json.load(f)
                logger.info(f"从明文文件加载API密钥: {self.api_keys_file}")
                
                # 如果加密功能可用，将明文文件转换为加密文件
                if self.fernet:
                    logger.info("将明文文件转换为加密文件")
                    self._save_encrypted_api_keys(file_keys)
                
                return file_keys
            
            return {}
        except Exception as e:
            logger.error(f"加载API密钥文件失败: {e}")
            return {}
    
    def _load_api_keys(self) -> Dict[str, Dict]:
        """从环境变量和持久化文件加载API密钥"""
        api_keys = {}
        
        # 首先从环境变量读取API密钥配置
        # 格式: API_KEY_1=key1:name1:permissions1, API_KEY_2=key2:name2:permissions2
        for i in range(1, 11):  # 支持最多10个API密钥
            key_config = os.getenv(f"API_KEY_{i}")
            if key_config:
                try:
                    parts = key_config.split(":")
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        name = parts[1].strip()
                        permissions = parts[2].strip().split(",") if len(parts) > 2 else ["all"]
                        
                        api_keys[key] = {
                            "name": name,
                            "permissions": permissions,
                            "created_at": time.time(),
                            "last_used": None,
                            "usage_count": 0,
                            "source": "environment"
                        }
                        logger.info(f"从环境变量加载API密钥: {name}")
                except Exception as e:
                    logger.error(f"解析API密钥配置失败: {e}")
        
        # 然后从持久化文件加载动态添加的密钥
        try:
            file_keys = self._load_encrypted_api_keys() # 使用新的加载方法
            for key, info in file_keys.items():
                if key not in api_keys:  # 避免覆盖环境变量中的密钥
                    api_keys[key] = {
                        **info,
                        "source": "file"
                    }
                    logger.info(f"从文件加载API密钥: {info['name']}")
        except Exception as e:
            logger.error(f"加载API密钥文件失败: {e}")
        
        # 如果没有配置API密钥，创建一个默认密钥
        if not api_keys:
            default_key = os.getenv("DEFAULT_API_KEY", "genserve-default-key-2024")
            api_keys[default_key] = {
                "name": "default",
                "permissions": ["all"],
                "created_at": time.time(),
                "last_used": None,
                "usage_count": 0,
                "source": "default"
            }
            logger.warning("未配置API密钥，使用默认密钥")
        
        return api_keys
    
    def _save_api_keys(self):
        """保存API密钥到持久化文件"""
        try:
            # 只保存动态添加的密钥（source为file的）
            file_keys = {}
            for key, info in self.api_keys.items():
                if info.get("source") == "file":
                    # 创建副本，移除source字段
                    file_info = {k: v for k, v in info.items() if k != "source"}
                    file_keys[key] = file_info
            
            # 使用新的加密保存方法
            success = self._save_encrypted_api_keys(file_keys)
            
            if success:
                file_path = self.encrypted_file if self.fernet else self.api_keys_file
                logger.info(f"API密钥已保存到文件: {file_path}")
            else:
                logger.error("保存API密钥失败")
        except Exception as e:
            logger.error(f"保存API密钥文件失败: {e}")
    
    def add_api_key(self, api_key: str, name: str, permissions: List[str]) -> bool:
        """动态添加新的API密钥"""
        try:
            if api_key in self.api_keys:
                logger.warning(f"API密钥已存在: {name}")
                return False
            
            self.api_keys[api_key] = {
                "name": name,
                "permissions": permissions,
                "created_at": time.time(),
                "last_used": None,
                "usage_count": 0,
                "source": "file"
            }
            
            # 保存到持久化文件
            self._save_api_keys()
            
            logger.info(f"成功添加API密钥: {name}")
            return True
        except Exception as e:
            logger.error(f"添加API密钥失败: {e}")
            return False
    
    def remove_api_key(self, api_key: str) -> bool:
        """删除API密钥"""
        try:
            if api_key not in self.api_keys:
                logger.warning(f"API密钥不存在")
                return False
            
            key_info = self.api_keys[api_key]
            if key_info.get("source") == "environment":
                logger.warning(f"无法删除环境变量中的API密钥: {key_info['name']}")
                return False
            
            del self.api_keys[api_key]
            
            # 保存到持久化文件
            self._save_api_keys()
            
            logger.info(f"成功删除API密钥: {key_info['name']}")
            return True
        except Exception as e:
            logger.error(f"删除API密钥失败: {e}")
            return False
    
    def remove_api_key_by_name(self, name: str) -> Dict:
        """通过用户名删除API密钥"""
        try:
            # 查找匹配的密钥
            keys_to_remove = []
            for key, info in self.api_keys.items():
                if info["name"] == name:
                    if info.get("source") == "environment":
                        return {
                            "success": False,
                            "error": f"无法删除环境变量中的API密钥: {name}",
                            "deleted_keys": []
                        }
                    keys_to_remove.append(key)
            
            if not keys_to_remove:
                return {
                    "success": False,
                    "error": f"未找到用户名为 '{name}' 的API密钥",
                    "deleted_keys": []
                }
            
            # 删除找到的密钥
            deleted_keys = []
            for key in keys_to_remove:
                key_info = self.api_keys[key]
                del self.api_keys[key]
                deleted_keys.append({
                    "key_preview": f"{key[:8]}...{key[-4:]}" if len(key) > 12 else key,
                    "name": key_info["name"],
                    "permissions": key_info["permissions"]
                })
                logger.info(f"成功删除API密钥: {key_info['name']} ({key[:8]}...)")
            
            # 保存到持久化文件
            self._save_api_keys()
            
            return {
                "success": True,
                "message": f"成功删除 {len(deleted_keys)} 个API密钥",
                "deleted_keys": deleted_keys
            }
        except Exception as e:
            logger.error(f"通过用户名删除API密钥失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_keys": []
            }
    
    def remove_api_key_by_id(self, key_id: str) -> Dict:
        """通过key_id删除API密钥"""
        try:
            # 解析key_id格式：前8位-后4位
            if "-" not in key_id:
                return {
                    "success": False,
                    "error": "无效的key_id格式，应为：前8位-后4位",
                    "deleted_keys": []
                }
            
            prefix, suffix = key_id.split("-", 1)
            if len(prefix) != 8 or len(suffix) != 4:
                return {
                    "success": False,
                    "error": "无效的key_id格式，应为：前8位-后4位",
                    "deleted_keys": []
                }
            
            # 查找匹配的密钥
            target_key = None
            for key in self.api_keys.keys():
                if key.startswith(prefix) and key.endswith(suffix):
                    target_key = key
                    break
            
            if not target_key:
                return {
                    "success": False,
                    "error": f"未找到key_id为 '{key_id}' 的API密钥",
                    "deleted_keys": []
                }
            
            key_info = self.api_keys[target_key]
            if key_info.get("source") == "environment":
                return {
                    "success": False,
                    "error": f"无法删除环境变量中的API密钥: {key_info['name']}",
                    "deleted_keys": []
                }
            
            # 删除密钥
            del self.api_keys[target_key]
            
            # 保存到持久化文件
            self._save_api_keys()
            
            deleted_key_info = {
                "key_id": key_id,
                "key_preview": f"{target_key[:8]}...{target_key[-4:]}",
                "name": key_info["name"],
                "permissions": key_info["permissions"]
            }
            
            logger.info(f"成功删除API密钥: {key_info['name']} ({key_id})")
            
            return {
                "success": True,
                "message": f"成功删除API密钥: {key_info['name']}",
                "deleted_keys": [deleted_key_info]
            }
        except Exception as e:
            logger.error(f"通过key_id删除API密钥失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted_keys": []
            }
    
    def reload_api_keys(self):
        """重新加载API密钥"""
        logger.info("重新加载API密钥...")
        self.api_keys = self._load_api_keys()
        logger.info(f"重新加载完成，共 {len(self.api_keys)} 个密钥")
    
    def validate_api_key(self, api_key: str) -> Dict:
        """验证API密钥"""
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="缺少API密钥",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if api_key not in self.api_keys:
            raise HTTPException(
                status_code=401,
                detail="无效的API密钥",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # 更新使用统计
        key_info = self.api_keys[api_key]
        key_info["last_used"] = time.time()
        key_info["usage_count"] += 1
        
        logger.info(f"API密钥验证成功: {key_info['name']}")
        return key_info
    
    def check_permission(self, key_info: Dict, required_permission: str) -> bool:
        """检查权限"""
        permissions = key_info.get("permissions", [])
        return "all" in permissions or required_permission in permissions
    
    def get_rate_limit_info(self, api_key: str) -> Dict:
        """获取速率限制信息"""
        # 简单的速率限制实现
        current_time = time.time()
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = {
                "requests": [],
                "limit": 100,  # 每分钟100个请求
                "window": 60   # 60秒窗口
            }
        
        rate_info = self.rate_limits[api_key]
        
        # 清理过期的请求记录
        rate_info["requests"] = [
            req_time for req_time in rate_info["requests"]
            if current_time - req_time < rate_info["window"]
        ]
        
        return rate_info
    
    def check_rate_limit(self, api_key: str) -> bool:
        """检查速率限制"""
        rate_info = self.get_rate_limit_info(api_key)
        
        if len(rate_info["requests"]) >= rate_info["limit"]:
            return False
        
        rate_info["requests"].append(time.time())
        return True

# 全局认证管理器实例
auth_manager = AuthManager()

async def get_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer()),
    x_api_key: Optional[str] = Header(None)
) -> Dict:
    """获取并验证API密钥"""
    api_key = None
    
    # 优先使用Authorization header
    if authorization:
        api_key = authorization.credentials
    # 备用X-API-Key header
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="缺少API密钥。请在Authorization header中使用Bearer token或在X-API-Key header中提供密钥",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # 验证API密钥
    key_info = auth_manager.validate_api_key(api_key)
    
    # 检查速率限制
    if not auth_manager.check_rate_limit(api_key):
        raise HTTPException(
            status_code=429,
            detail="请求频率过高，请稍后再试"
        )
    
    return key_info

def require_permission(permission: str):
    """权限检查装饰器"""
    def permission_checker(key_info: Dict = Depends(get_api_key)):
        if not auth_manager.check_permission(key_info, permission):
            raise HTTPException(
                status_code=403,
                detail=f"权限不足，需要权限: {permission}"
            )
        return key_info
    return permission_checker

# 预定义的权限检查器
require_generation = require_permission("generation")
require_readonly = require_permission("readonly")
require_admin = require_permission("admin")

class AuthUtils:
    """认证工具类"""
    
    @staticmethod
    def generate_api_key(name: str, permissions: List[str] = None) -> str:
        """生成新的API密钥"""
        if permissions is None:
            permissions = ["generation"]
        
        # 生成基于时间戳和名称的密钥
        timestamp = str(int(time.time()))
        base_string = f"{name}:{timestamp}:{','.join(permissions)}"
        
        # 使用SHA256生成密钥
        api_key = hashlib.sha256(base_string.encode()).hexdigest()[:32]
        
        return api_key
    
    @staticmethod
    def get_api_key_config(api_key: str) -> str:
        """生成API密钥配置字符串"""
        key_info = auth_manager.api_keys.get(api_key)
        if not key_info:
            return ""
        
        permissions_str = ",".join(key_info["permissions"])
        return f"{api_key}:{key_info['name']}:{permissions_str}"
    
    @staticmethod
    def list_api_keys() -> List[Dict]:
        """列出所有API密钥信息（不包含实际密钥）"""
        keys_info = []
        for key, info in auth_manager.api_keys.items():
            # 生成唯一标识符（基于密钥的前8位和后4位）
            key_id = f"{key[:8]}-{key[-4:]}"
            
            keys_info.append({
                "key_id": key_id,  # 唯一标识符，用于删除
                "name": info["name"],
                "permissions": info["permissions"],
                "created_at": info["created_at"],
                "last_used": info["last_used"],
                "usage_count": info["usage_count"],
                "key_preview": f"{key[:8]}...{key[-4:]}" if len(key) > 12 else key,
                "source": info.get("source", "unknown"),
                "can_delete": info.get("source") != "environment"  # 是否可以删除
            })
        return keys_info
    
    @staticmethod
    def add_api_key(api_key: str, name: str, permissions: List[str]) -> bool:
        """添加新的API密钥"""
        return auth_manager.add_api_key(api_key, name, permissions)
    
    @staticmethod
    def remove_api_key(api_key: str) -> bool:
        """删除API密钥"""
        return auth_manager.remove_api_key(api_key)
    
    @staticmethod
    def remove_api_key_by_name(name: str) -> Dict:
        """通过用户名删除API密钥"""
        return auth_manager.remove_api_key_by_name(name)
    
    @staticmethod
    def remove_api_key_by_id(key_id: str) -> Dict:
        """通过key_id删除API密钥"""
        return auth_manager.remove_api_key_by_id(key_id)
    
    @staticmethod
    def reload_api_keys():
        """重新加载API密钥"""
        auth_manager.reload_api_keys() 