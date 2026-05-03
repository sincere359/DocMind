"""认证路由：注册 / 登录"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docmind.auth import UserStore, create_access_token

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=4, max_length=64)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    ok: bool = True
    token: str = ""
    username: str = ""


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """注册新用户"""
    store = UserStore()
    user = store.create_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    token = create_access_token(req.username)
    return AuthResponse(token=token, username=req.username)


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """用户登录"""
    store = UserStore()
    user = store.verify_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(req.username)
    return AuthResponse(token=token, username=req.username)
