import hashlib
import re
import pymupdf
import pymupdf4llm
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PdfChunk, JobUnit
from app.ai.local import chunking, embed, ask_question, check_answer

router = APIRouter()

@router.post("/question")
def first_question(
    job: str = Form(...),
    resume_pdf: UploadFile = File(None),
    my_resume: str = Form(None),
    cl_pdf: UploadFile = File(None),
    my_cl: str = Form(None),
    db: Session = Depends(get_db),
):
    filenames = []

    if resume_pdf:
        resume_bytes = resume_pdf.file.read()
        resume_text = extract_text(resume_bytes)
        filenames.append(save_pdf(db, "resume", resume_pdf.filename, resume_bytes, resume_text))
    elif my_resume:
        filenames.append(save_pdf(db, "resume", "my_resume", my_resume.encode(), my_resume))

    if cl_pdf:
        cl_bytes = cl_pdf.file.read()
        cl_text = extract_text(cl_bytes)
        filenames.append(save_pdf(db, "cl", cl_pdf.filename, cl_bytes, cl_text))
    elif my_cl:
        filenames.append(save_pdf(db, "cl", "my_cl", my_cl.encode(), my_cl))

    evidences = search_chunks(db, filenames, "지원 동기")
    result = ask_question("지원 동기", job, evidences)
    return {
        "question": result["question"],
        "evidences": result["evidences"],
        "filenames": filenames,
    }  # 프론트에 전달


@router.post("/answer")
def submit_answer(data: dict, db: Session = Depends(get_db)):
    filenames = data.get("filenames", [])
    evidences = search_chunks(db, filenames, data["category"])

    job_units = []
    if data["category"] == "경험 및 역량" and not evidences:
        job_units = search_job_units(db, data["job"], data["job"])

    result = ask_question(data["category"], data["job"], evidences, job_units)
    return {
        "question": result["question"],
        "evidences": result["evidences"],
    }  # 다음 카테고리 질문


@router.post("/verify")
def verify_answer(data: dict, db: Session = Depends(get_db)):
    filenames = data.get("filenames", [])
    rows = search_rows(db, filenames, data["answer"], top_k=3)
    if not rows:
        return {"mismatches": []}

    mismatches = []
    # 서류 종류별로 따로 비교해서, 경고마다 정확한 서류 이름을 붙인다
    for doctype, source in (("resume", "이력서"), ("cl", "자기소개서")):
        evidences = [clean_evidence(row.content) for row in rows if row.doctype == doctype]
        for mismatch in check_answer(data["answer"], evidences):
            mismatch["source"] = source  # 화면에 어느 서류인지 표시
            mismatches.append(mismatch)
    return {"mismatches": mismatches}
    return {"mismatches": mismatches}

# 1. 추출
def extract_text(pdf_bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # 메모리에서 엶
    text = pymupdf4llm.to_markdown(doc)  # 폰트 크기 등 표시
    for page in doc:
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                lines.append(("".join(s["text"] for s in line["spans"]), line["bbox"][2]))
        right = max((x for _, x in lines), default=0)
        for (current, x), (following, _) in zip(lines, lines[1:]):
            # 오른쪽 끝까지 찼고, 공백 없이 한글로 끝나고, 다음 줄이 한글로 시작하면 잘린 단어
            if x >= right - 15 and re.search(r"[가-힣]$", current) and re.match(r"[가-힣]", following):
                tail = current.split()[-1]
                head = following.split()[0]
                text = re.sub(re.escape(tail) + r"\s+" + re.escape(head), lambda _: tail + head, text)
    doc.close()
    return text.strip()


# 화면에 보여 줄 근거에서 제목과 마크다운 기호 지우기
def clean_evidence(text: str) -> str:
    text = re.sub(r"^\s*#+.*$", "", text, flags=re.M)  # # 제목 줄
    text = re.sub(r"^\s*\[[^\]\n]+\]\s*", "", text, flags=re.M)  # [지원 동기] 같은 제목
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()



# 4. 저장
def save_pdf(
    db, doctype: str, filename: str, pdf_bytes: bytes, text: str
) -> str:  # db = Session()
    hash = hashlib.md5(pdf_bytes).hexdigest()[:8]  # 파일 지문
    hash_name = f"{hash}_{filename}"

    if db.query(PdfChunk).filter_by(doctype=doctype, filename=hash_name).first():
        return hash_name

    chunks = embed(chunking(text))
    for chunk in chunks:
        row = PdfChunk(
            doctype=doctype,
            filename=hash_name,
            index=chunk["index"],
            content=chunk["content"],
            embedding=chunk["embedding"],
        )
        db.add(row)

    db.commit()
    return hash_name


# 원문 청크 검색 (DB 행 그대로 돌려줌)
def search_rows(db: Session, filenames: list[str], query: str, top_k: int = 1) -> list[PdfChunk]:
    if not filenames:
        return []

    query_vec = embed([{"content": query}])[0]["embedding"]
    return (
        db.query(PdfChunk)
        .filter(PdfChunk.filename.in_(filenames))
        .order_by(PdfChunk.embedding.cosine_distance(query_vec))
        .limit(top_k)
        .all()
    )


# 원문 청크 검색 (화면에 보여 줄 글만 돌려줌)
def search_chunks(db: Session, filenames: list[str], query: str, top_k: int = 1) -> list[str]:
    rows = search_rows(db, filenames, query, top_k)
    cleaned = [clean_evidence(row.content) for row in rows]
    return [text for text in cleaned if text]

# 직무에 맞는 NCS 능력단위 검색
def search_job_units(db: Session, job: str, query: str, top_k: int = 3) -> list[str]:
    query_vec = embed([{"content": query}])[0]["embedding"]
    rows = (
        db.query(JobUnit)
        .filter(JobUnit.job == job)
        .order_by(JobUnit.embedding.cosine_distance(query_vec))
        .limit(top_k)
        .all()
    )
    return [f"{row.unit_name}: {row.definition}" for row in rows]