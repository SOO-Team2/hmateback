import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# .env 파일 로드
env_path = Path(__file__).resolve().parent.parent / ".env" #절대 경로
load_dotenv(dotenv_path=env_path)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True) #안 하면 단순 문자열, DB 연결 끊어졌는지 확인

# 세션 -> 동시 처리 가능
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM 사용
Base = declarative_base()

def get_db():
  db = Session()
  try:
    yield db #API에게 빌려주기
  finally:
    db.close() #작업 끝나면 닫기