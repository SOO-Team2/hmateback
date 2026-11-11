import logging
from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database import get_db, Base, engine
from app.models import User, Resume, Career, Certificate, Interview, Dialogue, CL, Answer
from app.routes import cl, mypage, auth, interview
from app.ai.local import LLM_URL, LLM_MODEL

logger = logging.getLogger("uvicorn.info")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 Ollama 연결 상태 확인 (3초 타임아웃)
    base_ai_url = LLM_URL.split("/v1")[0]
    try:
        res = requests.get(base_ai_url, timeout=3)
        if res.status_code == 200:
            logger.info(" [AI] Ollama 서버 연결 확인됨 (모델: %s)", LLM_MODEL)
        else:
            logger.warning("⚠️ [AI] Ollama 서버 응답 이상 (HTTP %s)", res.status_code)
    except Exception:
        logger.warning("⚠️ [AI] Ollama 서버에 연결할 수 없습니다. (%s 가 켜져 있는지 확인하세요)", base_ai_url)

    yield

Base.metadata.create_all(bind=engine)

app = FastAPI(title="합격메이트 API", lifespan=lifespan)

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
app.include_router(interview.router)
