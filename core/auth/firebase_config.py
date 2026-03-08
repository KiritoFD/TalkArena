"""
Firebase 配置和初始化模块
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FirebaseConfig:
    """Firebase 配置管理类"""
    
    def __init__(self):
        self.config_file = Path(__file__).parent / "firebase_config.json"
        self.config: Optional[Dict[str, Any]] = None
        self.initialized = False
    
    def load_config(self) -> bool:
        """
        加载 Firebase 配置文件
        
        Returns:
            bool: 加载是否成功
        """
        try:
            if not self.config_file.exists():
                logger.warning(f"Firebase 配置文件不存在：{self.config_file}")
                logger.warning("请复制 firebase_config.example.json 为 firebase_config.json 并填入配置信息")
                return False
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 验证必要的配置项
            required_fields = [
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
                "client_id",
                "auth_uri",
                "token_uri"
            ]
            
            missing_fields = [field for field in required_fields if field not in self.config]
            if missing_fields:
                logger.error(f"Firebase 配置缺少必要字段：{missing_fields}")
                return False
            
            self.initialized = True
            logger.info("Firebase 配置加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载 Firebase 配置失败：{e}")
            return False
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """获取 Firebase 配置"""
        if not self.initialized:
            self.load_config()
        return self.config


# 全局配置实例
firebase_config = FirebaseConfig()


def get_firebase_config() -> Optional[Dict[str, Any]]:
    """获取 Firebase 配置"""
    return firebase_config.get_config()


def is_firebase_initialized() -> bool:
    """检查 Firebase 是否已初始化"""
    return firebase_config.initialized
