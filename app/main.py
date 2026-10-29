from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, Resume, Career, Certificate, Interview, Dialogue, CL, Answer
from app.routes.cl import router as cl_router 

app = FastAPI()

# 프론트엔드 연결
app.add_middleware( #요청 검사
  CORSMiddleware, #다른 포트 번호 풀어줌
  allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
  allow_methods=["*"], #GET, POST, PUT, DELETE 모두 허용
  allow_headers=["*"], #헤더 전부 허용
)

app.include_router(cl_router) 

@app.get("/users")
def get_users(db: Session = Depends(get_db)):
  users = db.query(User).all()
  return users

@app.get("/resumes")
def get_resumes(db: Session = Depends(get_db)):
  return db.query(Resume).all()

@app.get("/careers")
def get_careers(db: Session = Depends(get_db)):
  return db.query(Career).all()

@app.get("/certificates")
def get_certificates(db: Session = Depends(get_db)):
  return db.query(Certificate).all()

@app.get("/interviews")
def get_interviews(db: Session = Depends(get_db)):
  return db.query(Interview).all()

@app.get("/dialogues")
def get_dialogues(db: Session = Depends(get_db)):
  return db.query(Dialogue).all()

@app.get("/cls")
def get_cls(db: Session = Depends(get_db)):
  return db.query(CL).all()

@app.get("/answers")
def get_answers(db: Session = Depends(get_db)):
  return db.query(Answer).all()