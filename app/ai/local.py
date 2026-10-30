import json
import hashlib
import pymupdf
import pymupdf4llm
import requests
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PdfChunk

router = APIRouter()

@router.post("/pdf")
async def upload_pdf(
  file: UploadFile = File(...), # ...: 필수값, 파일 안 보내면 에러
  doctype: str = Form(...),
  db: Session = Depends(get_db)
):
  pdf_bytes = await file.read() # 010101...
  save_pdf(db, doctype, file.filename, pdf_bytes)
  return {"filename": file.filename, "doctype": doctype} # 프론트에 응답

LLM_URL     = 'http://127.0.0.1:11434/v1/chat/completions' #localhost보다 답변이 훨씬 빠름
LLM_MODEL   = 'qwen2.5:7b-instruct-q4_K_M'
EMBED_URL   = 'http://127.0.0.1:11434/v1/embeddings'
EMBED_MODEL = 'bge-m3'

# 1. 추출
def extract_text(pdf_bytes) -> str:
  doc = pymupdf.open(stream=pdf_bytes, filetype="pdf") #메모리에서 엶
  text = pymupdf4llm.to_markdown(doc) #폰트 크기 등 표시
  doc.close()
  return text.strip()

# 2. 청킹
def chunking(text):
  prompt = """제공된 텍스트를 다음 규칙에 따라 청킹해 주세요.

[규칙]
1. 원문 보존:
  - 원문의 내용을 절대 요약, 수정, 생략하지 말고 원문 그대로 포함해 주세요.

2. 의미 및 구조 기준:
  - 문서의 본래 구조(마크다운 헤딩 `#`, `##`, 장·절, 또는 문단 `\n\n`)를 최우선 기준으로 의미 단위가 끊기지 않도록 나누어 주세요.
  - 단일 청크의 권장 크기는 대략 300 ~ 500자 내외를 목표로 하되, 문단과 소제목 구조를 우선시해 주세요.

3. 조건부 오버랩 적용:
  - 동일한 소제목/문항 내에서 텍스트가 길어져 분할될 때만 이전 청크의 마지막 문장(50자 내외)을 다음 청크 시작 부분에 중복 포함시켜 주세요.
  - 주제가 바뀌는 새로운 소제목/문항이 시작될 때는 오버랩을 적용하지 마세요.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "chunks": [
    {
      "index": 1,
      "content": "청킹 텍스트 내용..."
    },
    {
      "index": 2,
      "content": "다음 청킹 텍스트 내용..."
    }
  ]
}
"""

  res = requests.post(
    LLM_URL,
    json={
      "model": LLM_MODEL,
      "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
      ],
      "response_format": {"type": "json_object"}, #오직 JSON만 반환
      "temperature": 0.1
    }
  )
  content = res.json()["choices"][0]["message"]["content"]
  data = json.loads(content) # str -> dict
  return data["chunks"]

# 3. 임베딩
def embed(chunks):
  texts = [c["content"] for c in chunks]
  res = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
  embeddings = res.json()["data"]

  for chunk, emb in zip(chunks, embeddings):
    chunk["embedding"] = emb["embedding"] #[-0.02314, 0.04128, ..., 0.00891]

  return chunks

# 4. 저장
def save_pdf(db, doctype: str, filename: str, pdf_bytes: bytes): #db = Session()
  hash = hashlib.md5(pdf_bytes).hexdigest()[:8] #파일 지문
  hash_name = f"{hash}_{filename}"

  if db.query(PdfChunk).filter_by(doctype=doctype, filename=hash_name).first():
    return
  
  chunks = embed(chunking(extract_text(pdf_bytes)))
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