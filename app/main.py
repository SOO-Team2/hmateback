from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.models import User, Resume, Career, Certificate, Interview, Dialogue, CL, Answer, PdfChunk
from app.routes.auth import router as auth_router
from app.routes.cl import router as cl_router
from app.ai.local import router as local_router

Base.metadata.create_all(bind=engine) #테이블 없으면 자동 생성

app = FastAPI()

# 프론트엔드 연결
app.add_middleware( #요청 검사
  CORSMiddleware, #다른 포트 번호 풀어줌
  allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
  allow_methods=["*"], #GET, POST, PUT, DELETE 모두 허용
  allow_headers=["*"], #헤더 전부 허용
)

# 라우터 등록
app.include_router(auth_router)
app.include_router(cl_router)
app.include_router(local_router)