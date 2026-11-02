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

resume_text = ""
cl_text = ""

@router.post("/pdf")
async def upload_pdf(
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

# LLM 호출
def call_llm(prompt: str, content: str) -> dict:
  res = requests.post(
    LLM_URL,
    json={
      "model": LLM_MODEL,
      "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
      ],
      "response_format": {"type": "json_object"},
      "temperature": 0.3
    }
  )
  content = res.json()["choices"][0]["message"]["content"]
  return json.loads(content)

PROMPT_CATEGORIES = {
  "지원 동기": """
지원자의 직무, 이력서, 자기소개서 내용을 바탕으로 '왜 우리 회사에 지원했는지' 그 동기를 파고드는 면접 질문 1개와 참고한 근거 문장을 추출해 주세요.

[제약 사항]
1. 면접관의 입장에서 질문할 것 (※ '귀사' 표현 금지, '우리 회사'나 '당사' 사용)
2. '지원동기' 단계이므로, 오직 **과거와 현재의 선택 이유("왜 이 회사인가?", "왜 이 직무인가?")**에만 집중할 것.
3. 절대 '입사 후 포부'나 미래 계획("입사해서 어떤 기여를 할 것인지", "향후 목표")에 대한 내용을 질문에 포함하지 말 것.
4. 이력서·자기소개서에서 참고한 문장이 있다면 원문 문장 1개를 그대로 가져오고, 일반적인 질문을 하는 경우에는 빈 배열([])로 출력할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장",
  "evidences": [
    "참고한 원문 문장 1"
  ]
}
""",
  "경험 및 역량": """
지원자의 직무, 이력서, 자기소개서 내용을 바탕으로 직무 관련 프로젝트 경험과 기술 역량을 검증하는 면접 질문 1개와 참고한 근거 문장을 추출해 주세요.

[제약 사항]
1. 면접관의 입장에서 질문할 것 (※ '귀사' 표현 금지, '우리 회사'나 '당사' 사용)
2. 이력서·자기소개서에 적힌 구체적인 프로젝트, 문제 해결 과정, 본인의 역할이나 기술적 선택 이유를 파고들 것.
3. 이력서·자기소개서에서 참고한 문장이 있다면 원문 문장 1개를 그대로 가져오고, 일반적인 질문을 하는 경우에는 빈 배열([])로 출력할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장",
  "evidences": [
    "참고한 원문 문장 1"
  ]
}
""",
  "인성 및 협업": """
지원자의 직무, 이력서, 자기소개서 내용을 바탕으로 협업 태도, 소통 방식, 인성을 검증하는 면접 질문 1개와 참고한 근거 문장을 추출해 주세요.

[제약 사항]
1. 면접관의 입장에서 질문할 것 (※ '귀사' 표현 금지, '우리 회사'나 '당사' 사용)
2. 팀원과의 협업 과정, 갈등 해결 경험, 또는 본인의 장단점이 드러나는 구체적인 사례를 물어볼 것.
3. 이력서·자기소개서에서 참고한 문장이 있다면 원문 문장 1개를 그대로 가져오고, 일반적인 질문을 하는 경우에는 빈 배열([])로 출력할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장",
  "evidences": [
    "참고한 원문 문장 1"
  ]
}
""",
  "입사 후 포부": """
지원자의 직무, 이력서, 자기소개서 내용을 바탕으로 입사 후 포부와 성장 계획을 확인하는 면접 질문 1개와 참고한 근거 문장을 추출해 주세요.

[제약 사항]
1. 면접관의 입장에서 질문할 것 (※ '귀사' 표현 금지, '우리 회사'나 '당사' 사용)
2. 입사 후 우리 회사에서 어떻게 기여할 것인지, 단기·장기적인 목표와 실천 계획을 구체적으로 물어볼 것.
3. 이력서·자기소개서에서 참고한 문장이 있다면 원문 문장 1개를 그대로 가져오고, 일반적인 질문을 하는 경우에는 빈 배열([])로 출력할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장",
  "evidences": [
    "참고한 원문 문장 1"
  ]
}
"""
}

# 5. 질문
def ask_question(category: str, job: str, resume_text: str, cl_text: str) -> dict:
  prompt = PROMPT_CATEGORIES[category]

  user_content = f"""
[지원 직무]
{job}

[이력서]
{resume_text}

[자기소개서]
{cl_text}
"""
  return call_llm(prompt, user_content)