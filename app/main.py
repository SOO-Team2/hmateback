from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database import get_db, Base, engine
from app.models import User, Resume, Career, Certificate, Interview, Dialogue, CL, Answer
from app.routes import cl, mypage, auth, interview
from app.ai import local

Base.metadata.create_all(bind=engine)

app = FastAPI()

# 프론트엔드 연결
app.add_middleware( #요청 검사
  CORSMiddleware, #다른 포트 번호 풀어줌
  allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
  allow_methods=["*"], #GET, POST, PUT, DELETE 모두 허용
  allow_headers=["*"], #헤더 전부 허용
)

app.include_router(cl.router)
app.include_router(auth.router)
app.include_router(mypage.router)
app.include_router(local.router)
app.include_router(interview.router)