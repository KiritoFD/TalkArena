"""
用户认证 API 路由
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["认证"])


class RegisterRequest(BaseModel):
    """注册请求"""
    email: EmailStr
    password: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    """登录请求"""
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    """认证响应"""
    success: bool
    message: str
    uid: Optional[str] = None
    user: Optional[Dict[str, Any]] = None


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """
    用户注册
    
    Args:
        request: 注册请求，包含邮箱、密码、用户名
    
    Returns:
        AuthResponse: 注册结果
    """
    from .firebase_auth import get_auth_manager
    
    auth_manager = get_auth_manager()
    
    if not auth_manager.initialized:
        return AuthResponse(
            success=False,
            message="认证服务未初始化，请检查 Firebase 配置"
        )
    
    # 创建用户
    success, result = auth_manager.create_user(
        email=request.email,
        password=request.password,
        username=request.username
    )
    
    if success:
        # 获取用户信息
        user_info = auth_manager.get_user(result)
        return AuthResponse(
            success=True,
            message="注册成功",
            uid=result,
            user=user_info
        )
    else:
        return AuthResponse(
            success=False,
            message=result
        )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    用户登录
    
    Args:
        request: 登录请求，包含邮箱和密码
    
    Returns:
        AuthResponse: 登录结果
    """
    from .firebase_auth import get_auth_manager
    
    auth_manager = get_auth_manager()
    
    if not auth_manager.initialized:
        return AuthResponse(
            success=False,
            message="认证服务未初始化，请检查 Firebase 配置"
        )
    
    # 登录验证
    success, result = auth_manager.login_user(
        email=request.email,
        password=request.password
    )
    
    if success:
        # 获取用户信息
        user_info = auth_manager.get_user(result)
        return AuthResponse(
            success=True,
            message="登录成功",
            uid=result,
            user=user_info
        )
    else:
        return AuthResponse(
            success=False,
            message=result
        )


@router.get("/me", response_model=AuthResponse)
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    获取当前用户信息
    
    Args:
        authorization: Bearer Token
    
    Returns:
        AuthResponse: 用户信息
    """
    from .firebase_auth import get_auth_manager
    
    auth_manager = get_auth_manager()
    
    if not auth_manager.initialized:
        return AuthResponse(
            success=False,
            message="认证服务未初始化"
        )
    
    if not authorization or not authorization.startswith("Bearer "):
        return AuthResponse(
            success=False,
            message="未提供认证令牌"
        )
    
    token = authorization.replace("Bearer ", "")
    
    # 验证 token
    decoded_token = auth_manager.verify_token(token)
    if not decoded_token:
        return AuthResponse(
            success=False,
            message="认证令牌无效或已过期"
        )
    
    # 获取用户信息
    uid = decoded_token.get('uid')
    user_info = auth_manager.get_user(uid)
    
    if user_info:
        return AuthResponse(
            success=True,
            message="获取成功",
            uid=uid,
            user=user_info
        )
    else:
        return AuthResponse(
            success=False,
            message="用户不存在"
        )


@router.post("/update-stats")
async def update_user_stats(
    uid: str,
    game_result: Dict[str, Any],
    authorization: Optional[str] = Header(None)
):
    """
    更新用户游戏统计
    
    Args:
        uid: 用户 ID
        game_result: 游戏结果数据
        authorization: Bearer Token
    
    Returns:
        Dict: 更新结果
    """
    from .firebase_auth import get_auth_manager
    
    auth_manager = get_auth_manager()
    
    if not auth_manager.initialized:
        return {"success": False, "message": "认证服务未初始化"}
    
    # 验证 token（可选，如果允许匿名更新则跳过）
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        decoded_token = auth_manager.verify_token(token)
        if not decoded_token or decoded_token.get('uid') != uid:
            raise HTTPException(status_code=401, detail="认证失败")
    
    # 更新统计
    success = auth_manager.update_user_stats(uid, game_result)
    
    if success:
        return {"success": True, "message": "更新成功"}
    else:
        return {"success": False, "message": "更新失败"}
