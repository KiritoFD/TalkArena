# Firebase 认证配置指南

## 前置准备

### 1. 创建 Firebase 项目

1. 访问 [Firebase 控制台](https://console.firebase.google.com/)
2. 点击"添加项目"创建新项目
3. 按向导完成项目创建

### 2. 启用 Authentication

1. 在 Firebase 控制台中选择你的项目
2. 左侧菜单选择"Authentication"
3. 点击"开始使用"
4. 在"登录方法"标签页中启用"邮箱/密码"

### 3. 创建服务账号

1. 点击项目概览旁边的齿轮图标，进入"项目设置"
2. 选择"服务账号"标签页
3. 点击"生成新私钥"
4. 下载 JSON 格式的私钥文件

### 4. 配置项目

1. 将下载的 JSON 文件重命名为 `firebase_config.json`
2. 将其放置在此目录下：`TalkArena/core/auth/firebase_config.json`
3. **注意：** 不要将 `firebase_config.json` 提交到 Git（已添加到 .gitignore）

### 5. 安装依赖

```bash
pip install firebase-admin>=6.0.0
```

## 使用示例

### 后端 API

```python
from core.auth.firebase_auth import get_auth_manager

auth_manager = get_auth_manager()

# 创建用户
success, result = auth_manager.create_user(
    email="user@example.com",
    password="password123",
    username="张三"
)

# 获取用户信息
user_info = auth_manager.get_user(uid)

# 更新用户统计
auth_manager.update_user_stats(uid, {
    'win': True,
    'turns': 10,
    'dominance_user': 60,
    'dominance_ai': 40
})
```

### 前端使用

```javascript
// 注册
const response = await fetch('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        email: 'user@example.com',
        password: 'password123',
        username: '张三'
    })
});

// 登录
const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        email: 'user@example.com',
        password: 'password123'
    })
});

// 获取当前用户
const response = await fetch('/api/auth/me', {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});
```

## API 端点

- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录
- `GET /api/auth/me` - 获取当前用户信息
- `POST /api/auth/update-stats` - 更新用户游戏统计

## 注意事项

1. **安全性**: Firebase 配置文件包含敏感信息，切勿提交到版本控制
2. **密码验证**: 当前实现中，密码验证在前端通过 Firebase Client SDK 完成，后端仅做用户查询
3. **Token 管理**: 登录成功后，token 存储在 localStorage 中，建议实现自动刷新机制
4. **跨域**: 如果部署到生产环境，需要配置 CORS 规则

## 故障排查

### 问题：Firebase 未初始化

**解决方案**:
1. 确认 `firebase_config.json` 文件存在
2. 确认 JSON 格式正确
3. 确认所有必要字段都已填写
4. 检查是否安装了 `firebase-admin` 包

### 问题：认证失败

**解决方案**:
1. 检查 Firebase 控制台中是否启用了邮箱/密码登录
2. 确认邮箱格式正确
3. 确认密码长度至少 6 位
4. 查看服务器日志获取详细错误信息
