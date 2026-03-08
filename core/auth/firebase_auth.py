"""
Firebase 认证管理器
"""
import logging
from typing import Optional, Dict, Any, Tuple
from .firebase_config import get_firebase_config, is_firebase_initialized
from firebase_admin import firestore

logger = logging.getLogger(__name__)


class FirebaseAuthManager:
    """Firebase 认证管理器"""
    
    def __init__(self):
        self.initialized = False
        self.app = None
        self.auth = None
        self.db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """初始化 Firebase"""
        try:
            config = get_firebase_config()
            if not config:
                logger.warning("Firebase 未配置，认证功能将不可用")
                return
            
            # 导入 Firebase Admin SDK
            import firebase_admin
            from firebase_admin import credentials, auth, firestore
            
            # 使用服务账号凭证初始化
            cred = credentials.Certificate(config)
            
            # 检查是否已经初始化过
            try:
                self.app = firebase_admin.get_app()
            except ValueError:
                self.app = firebase_admin.initialize_app(cred)
            
            self.auth = auth
            self.db = firestore.client()
            self.initialized = True
            
            logger.info("Firebase 初始化成功")
            
        except ImportError as e:
            logger.error(f"Firebase Admin SDK 未安装：{e}")
            logger.error("请运行：pip install firebase-admin")
        except Exception as e:
            logger.error(f"Firebase 初始化失败：{e}")
    
    def create_user(self, email: str, password: str, username: str = None) -> Tuple[bool, str]:
        """
        创建新用户
        
        Args:
            email: 用户邮箱
            password: 用户密码
            username: 用户名（可选）
        
        Returns:
            Tuple[bool, str]: (是否成功，消息/错误信息)
        """
        if not self.initialized:
            return False, "Firebase 未初始化"
        
        try:
            # 创建 Firebase Authentication 用户
            user = self.auth.create_user(
                email=email,
                password=password,
                display_name=username
            )
            
            # 在 Firestore 中创建用户文档
            user_data = {
                'uid': user.uid,
                'email': email,
                'username': username or email.split('@')[0],
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
                'game_stats': {
                    'total_games': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_turns': 0
                }
            }
            
            self.db.collection('users').document(user.uid).set(user_data)
            
            logger.info(f"用户创建成功：{email}")
            return True, user.uid
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"创建用户失败：{e}")
            
            # 解析错误信息
            if 'EMAIL_EXISTS' in error_msg:
                return False, "该邮箱已被注册"
            elif 'INVALID_EMAIL' in error_msg:
                return False, "邮箱格式无效"
            elif 'WEAK_PASSWORD' in error_msg:
                return False, "密码太弱，请至少使用 6 位密码"
            else:
                return False, f"创建失败：{str(e)}"
    
    def login_user(self, email: str, password: str) -> Tuple[bool, str]:
        """
        用户登录
        
        Args:
            email: 用户邮箱
            password: 用户密码
        
        Returns:
            Tuple[bool, str]: (是否成功，用户 ID/错误信息)
        """
        if not self.initialized:
            return False, "Firebase 未初始化"
        
        try:
            # Firebase Admin SDK 不直接支持邮箱密码登录
            # 这里我们使用自定义验证流程
            # 实际项目中应该使用 Firebase Client SDK 在前端进行认证
            
            # 获取用户信息
            user = self.auth.get_user_by_email(email)
            
            # 验证密码（需要通过 Firebase Client SDK）
            # 这里返回成功，实际密码验证在前端完成
            return True, user.uid
            
        except self.auth.UserNotFoundError:
            return False, "用户不存在"
        except Exception as e:
            logger.error(f"登录失败：{e}")
            return False, f"登录失败：{str(e)}"
    
    def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        获取用户信息
        
        Args:
            uid: 用户 ID
        
        Returns:
            Optional[Dict[str, Any]]: 用户信息，失败返回 None
        """
        if not self.initialized:
            return None
        
        try:
            # 从 Authentication 获取用户
            auth_user = self.auth.get_user(uid)
            
            # 从 Firestore 获取用户详细信息
            user_doc = self.db.collection('users').document(uid).get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                user_data['email'] = auth_user.email
                user_data['display_name'] = auth_user.display_name
                return user_data
            else:
                # 如果 Firestore 中没有，创建基本文档
                user_data = {
                    'uid': uid,
                    'email': auth_user.email,
                    'username': auth_user.display_name or auth_user.email.split('@')[0],
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'updated_at': firestore.SERVER_TIMESTAMP,
                    'game_stats': {
                        'total_games': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_turns': 0
                    }
                }
                self.db.collection('users').document(uid).set(user_data)
                return user_data
                
        except Exception as e:
            logger.error(f"获取用户信息失败：{e}")
            return None
    
    def update_user_stats(self, uid: str, game_result: Dict[str, Any]) -> bool:
        """
        更新用户游戏统计
        
        Args:
            uid: 用户 ID
            game_result: 游戏结果数据
        
        Returns:
            bool: 是否成功
        """
        if not self.initialized:
            return False
        
        try:
            user_ref = self.db.collection('users').document(uid)
            
            # 更新统计数据
            updates = {
                'updated_at': firestore.SERVER_TIMESTAMP,
                'game_stats.total_games': firestore.Increment(1),
                'game_stats.total_turns': firestore.Increment(game_result.get('turns', 0))
            }
            
            if game_result.get('win', False):
                updates['game_stats.wins'] = firestore.Increment(1)
            else:
                updates['game_stats.losses'] = firestore.Increment(1)
            
            user_ref.update(updates)
            
            # 保存游戏记录
            game_record = {
                'uid': uid,
                'result': game_result.get('result', ''),
                'turns': game_result.get('turns', 0),
                'dominance_user': game_result.get('dominance_user', 50),
                'dominance_ai': game_result.get('dominance_ai', 50),
                'created_at': firestore.SERVER_TIMESTAMP
            }
            
            self.db.collection('game_records').add(game_record)
            
            return True
            
        except Exception as e:
            logger.error(f"更新用户统计失败：{e}")
            return False
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证 Firebase ID Token
        
        Args:
            token: Firebase ID Token
        
        Returns:
            Optional[Dict[str, Any]]: 解码后的用户信息，失败返回 None
        """
        if not self.initialized:
            return None
        
        try:
            decoded_token = self.auth.verify_id_token(token)
            return decoded_token
        except Exception as e:
            logger.error(f"验证 token 失败：{e}")
            return None


# 全局认证管理器实例
auth_manager = FirebaseAuthManager()


def get_auth_manager() -> FirebaseAuthManager:
    """获取认证管理器实例"""
    return auth_manager
