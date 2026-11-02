import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt # 비밀번호 암호화(해싱) 라이브러리
import jwt    # 로그인 상태 유지 및 인증용 토큰(JWT) 라이브러리
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db # DB 세션 생성 및 반환 함수
from app.models import User     # USER 테이블 ORM 모델

# ==========================================
# 1. 라우터 및 JWT 보안 기본 설정
# ==========================================
# /auth 경로로 들어오는 모든 요청을 담당하는 라우터 생성
router = APIRouter(prefix="/auth", tags=["auth"])

# JWT 서명에 사용할 비밀키 (환경변수에 없으면 기본값 사용)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "hmate-secret-key-2026-production")  
ALGORITHM = "HS256"                     # 토큰 암호화 알고리즘
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24    # 토큰 유효 기간 (24시간)

# 요청 헤더의 'Authorization: Bearer <토큰>' 값을 추출하기 위한 스키마
security_scheme = HTTPBearer(auto_error=False)


# ==========================================
# 2. 데이터 양식 (Pydantic 스키마)
# 프론트엔드와 주고받는 데이터의 형태를 검증하고 정의합니다.
# ==========================================

# 회원가입 요청 시 프론트엔드에서 보내주는 데이터 양식
class SignupRequest(BaseModel):
    id: str = Field(..., min_length=2, max_length=50, description="사용자 아이디 (2~50자)")
    pw: str = Field(..., min_length=4, description="비밀번호 (최소 4자 이상)")
    email: str = Field(..., description="이메일 주소")
    service_agreed: bool = Field(..., description="서비스 이용약관 동의 (필수)")
    privacy_agreed: bool = Field(..., description="개인정보 수집 및 이용 동의 (필수)")
    save_agreed: bool = Field(False, description="문서 저장 동의 (선택: 자소서/이력서 저장 여부)")


# 로그인 요청 시 프론트엔드에서 보내주는 데이터 양식
class LoginRequest(BaseModel):
    id: str = Field(..., description="사용자 아이디")
    pw: str = Field(..., description="비밀번호")


# 아이디 중복 확인 결과 응답 양식
class CheckIdResponse(BaseModel):
    available: bool  # 사용 가능 여부 (True: 사용 가능, False: 중복)
    message: str    # 안내 메시지


# 사용자 정보 반환용 응답 양식 (보안상 비밀번호는 제외)
class UserResponse(BaseModel):
    id: str
    email: str
    service_agreed: bool
    privacy_agreed: bool
    save_agreed: bool
    joindate: Optional[datetime] = None

    class Config:
        from_attributes = True # SQLAlchemy 모델 객체를 Pydantic 모델로 자동 변환 허용


# 로그인 성공 시 프론트엔드에 전달할 토큰 및 유저 정보 양식
class TokenResponse(BaseModel):
    access_token: str        # 프론트가 다음 요청 헤더에 담아야 할 JWT 토큰
    token_type: str = "bearer" # 토큰 유형 (Bearer)
    user: UserResponse       # 로그인한 사용자 프로필 정보


# ==========================================
# 3. 보안 유틸 함수 (암호화 & 토큰 발급/검증)
# ==========================================

def hash_password(password: str) -> str:

    # 사용자가 입력한 일반 비밀번호를 안전하게 단방향 암호화(bcrypt 해싱)합니다.
    salt = bcrypt.gensalt() # 비밀번호를 암호화 할 때 추가되는 임의의 데이터(보안강도를 높임)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    
    # 사용자가 입력한 비밀번호(평문)와 DB에 저장된 암호화된 비밀번호가 일치하는지 비교합니다.
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: dict) -> str:
    # 사용자 아이디(sub)와 유효기간(exp)을 담아 암호화된 JWT 토큰을 발급합니다.
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    [로그인 필수 API에서 사용되는 의존성 함수]
    - 프론트가 보낸 Authorization 헤더에서 토큰을 추출하고 유효한지 검증합니다.
    - 토큰 속 사용자 아이디(sub)로 DB에서 유저를 찾아 반환합니다.
    - 토큰이 없거나 만료/위조된 경우 401 Unauthorized 에러를 발생시킵니다.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="로그인이 필요하거나 인증 토큰이 만료되었습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # 헤더에 토큰 정보가 없는 경우
    if not credentials:
        raise credentials_exception

    try:
        # 토큰 복호화 및 유효성 검사
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_exception
    except jwt.PyJWTError:
        # 토큰 위조 혹은 유효기간 만료 시
        raise credentials_exception

    # DB에서 해당 유저 조회
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise credentials_exception
    return user


# ==========================================
# 4. 실제 API 엔드포인트
# ==========================================

@router.get("/check-id", response_model=CheckIdResponse, summary="아이디 중복 확인")
def check_id(
    user_id: str = Query(..., min_length=2, max_length=50, description="확인할 아이디"),
    db: Session = Depends(get_db),
):

# 회원가입 화면에서 사용자가 입력한 아이디가 이미 존재하는지 실시간으로 검사
    existing_user = db.query(User).filter(User.id == user_id).first()
    if existing_user:
        return CheckIdResponse(available=False, message="이미 사용 중인 아이디입니다.")
    return CheckIdResponse(available=True, message="사용 가능한 아이디입니다.")


@router.post("/signup", status_code=status.HTTP_201_CREATED, summary="회원가입")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
# 회원가입 정보를 받아 약관 동의 및 중복 검사를 거친 후 새 유저를 생성

    # 1. 필수 약관 동의 여부 검증 (하나라도 체크 안 하면 400 에러)
    if not req.service_agreed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="서비스 이용약관에 동의해야 가입할 수 있습니다.",
        )
    if not req.privacy_agreed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="개인정보 수집 및 이용에 동의해야 가입할 수 있습니다.",
        )

    # 2. 아이디 중복 여부 재확인 (중복 시 409 에러)
    existing_user = db.query(User).filter(User.id == req.id).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 존재하는 아이디입니다.",
        )

    # 3. 비밀번호 단방향 암호화 처리 후 DB 저장
    hashed_pw = hash_password(req.pw)
    new_user = User(
        id=req.id,
        pw=hashed_pw,
        email=req.email,
        service_agreed=req.service_agreed,
        privacy_agreed=req.privacy_agreed,
        save_agreed=req.save_agreed,
    )

    db.add(new_user)
    db.commit()          # DB 반영
    db.refresh(new_user) # 생성된 joindate 등 최신 필드값 동기화

    return {
        "message": "회원가입이 완료되었습니다.",
        "user": UserResponse.model_validate(new_user),
    }


@router.post("/login", response_model=TokenResponse, summary="로그인")
def login(req: LoginRequest, db: Session = Depends(get_db)):

    # 아이디와 비밀번호를 검증하고 성공 시 향후 인증에 사용할 JWT 토큰을 발급
    # 1. 아이디로 사용자 조회
    user = db.query(User).filter(User.id == req.id).first()
    
    # 2. 유저가 없거나 비밀번호가 일치하지 않으면 401 에러
    if not user or not verify_password(req.pw, user.pw):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    # 3. 사용자 아이디를 담은 JWT 토큰 발급
    access_token = create_access_token(data={"sub": user.id})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


# 프론트엔드가 페이지를 새로고침하거나 마이페이지 접근 시 로그인 유지 여부를 확인합니다.
# 헤더에 유효한 Bearer 토큰이 전달되어야만 호출 가능합니다.
@router.get("/me", response_model=UserResponse, summary="내 정보 조회 (로그인 유지 확인)")
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


# 로그아웃 안내를 반환합니다. 
# JWT 방식에서는 프론트엔드(브라우저)에서 저장해 둔 토큰을 삭제하면 안전하게 로그아웃 처리됩니다.
@router.post("/logout", summary="로그아웃")
def logout(current_user: User = Depends(get_current_user)):

    return {"message": "로그아웃되었습니다.", "user_id": current_user.id}
