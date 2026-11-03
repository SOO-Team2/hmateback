import hashlib
import pymupdf
import pymupdf4llm
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PdfChunk
from app.ai.local import chunking, embed, ask_question

router = APIRouter()

resume_text = ""
cl_text = ""

@router.post("/question")
async def first_question(
  job: str = Form(...),
  resume: UploadFile = File(None),
  cl: UploadFile = File(None),
  db: Session = Depends(get_db)
):
  global resume_text, cl_text
  resume_text = ""
  cl_text = ""

  if resume:
    resume_bytes = await resume.read()
    resume_text = extract_text(resume_bytes)
    save_pdf(db, "resume", resume.filename, resume_bytes, resume_text)
  if cl:
    cl_bytes = await cl.read()
    cl_text = extract_text(cl_bytes)
    save_pdf(db, "cl", cl.filename, cl_bytes, cl_text)

  result = ask_question("지원 동기", job, resume_text, cl_text)
  return {"question": result["question"], "evidences": result.get("evidences", [])} #프론트에 전달

@router.post("/answer")
def submit_answer(data: dict):
  result = ask_question(data["category"], data["job"], resume_text, cl_text)
  return {"question": result["question"], "evidences": result.get("evidences", [])} #다음 카테고리 질문

# 1. 추출
def extract_text(pdf_bytes) -> str:
  doc = pymupdf.open(stream=pdf_bytes, filetype="pdf") #메모리에서 엶
  text = pymupdf4llm.to_markdown(doc) #폰트 크기 등 표시
  doc.close()
  return text.strip()

# 4. 저장
def save_pdf(db, doctype: str, filename: str, pdf_bytes: bytes, text: str): #db = Session()
  hash = hashlib.md5(pdf_bytes).hexdigest()[:8] #파일 지문
  hash_name = f"{hash}_{filename}"

  if db.query(PdfChunk).filter_by(doctype=doctype, filename=hash_name).first():
    return

  chunks = embed(chunking(text))
  for chunk in chunks:
    row = PdfChunk(
      doctype=doctype,
      filename=hash_name,
      index=chunk["index"],
      content=chunk["content"],
      embedding=chunk["embedding"]
    )
    db.add(row)

  db.commit()
