import hashlib
import re
import pymupdf
import pymupdf4llm
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PdfChunk, JobUnit, User
from app.routes.auth import get_current_user
from app.ai.local import chunking, embed, ask_question, check_answer, ask_followup_question

router = APIRouter()
MAX_PDF_SIZE = 10 * 1024 * 1024  # 10MB 크기 제한
MAX_PDF_PAGES = 10  # 최대 10페이지 제한

@router.post("/question")
def first_question(
    job: str = Form(...),
    resume_pdf: UploadFile = File(None),
    my_resume: str = Form(None),
    cl_pdf: UploadFile = File(None),
    my_cl: str = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    filenames = []

    if resume_pdf:
        resume_bytes = resume_pdf.file.read()
        if len(resume_bytes) > MAX_PDF_SIZE:
            raise HTTPException(status_code=400, detail="이력서 PDF 파일 크기는 10MB 이하여야 합니다.")
        resume_text = extract_text(resume_bytes)
        filenames.append(save_pdf(db, user.id, "resume", resume_pdf.filename, resume_bytes, resume_text))
    elif my_resume:
        filenames.append(save_pdf(db, user.id, "resume", "my_resume", my_resume.encode(), my_resume))

    if cl_pdf:
        cl_bytes = cl_pdf.file.read()
        if len(cl_bytes) > MAX_PDF_SIZE:
            raise HTTPException(status_code=400, detail="자기소개서 PDF 파일 크기는 10MB 이하여야 합니다.")
        cl_text = extract_text(cl_bytes)
        filenames.append(save_pdf(db, user.id, "cl", cl_pdf.filename, cl_bytes, cl_text))
    elif my_cl:
        filenames.append(save_pdf(db, user.id, "cl", "my_cl", my_cl.encode(), my_cl))

    evidences = search_chunks(db, user.id, filenames, "지원 동기")
    result = ask_question("지원 동기", job, evidences)
    return {
        "question": result["question"],
        "evidences": result["evidences"],
        "filenames": filenames,
    }  # 프론트에 전달


@router.post("/answer")
def submit_answer(data: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    filenames = data.get("filenames", [])
    evidences = search_chunks(db, user.id, filenames, data["category"])

    job_units = []
    if data["category"] == "경험 및 역량" and not evidences:
        job_units = search_job_units(db, data["job"], data["job"])

    result = ask_question(data["category"], data["job"], evidences, job_units)
    return {
        "question": result["question"],
        "evidences": result["evidences"],
    }  # 다음 카테고리 질문


# 꼬리 질문 생성 여부 판단 및 질문 반환
@router.post("/followup")
def followup_question(data: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    filenames = data.get("filenames", [])
    evidences = search_chunks(db, user.id, filenames, data["category"])

    result = ask_followup_question(
        category=data["category"],
        job=data["job"],
        question=data["question"],
        answer=data["answer"],
        evidences=evidences,
    )
    return {
        "is_followup": result["need_followup"],
        "question": result["question"],
    }


@router.post("/verify")
def verify_answer(data: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    filenames = data.get("filenames", [])

    # 답변을 문장으로 나누고, 문장마다 가장 가까운 서류 조각 1개를 찾는다
    # 같은 조각을 가리키는 문장끼리 묶어서, 관련 있는 조각하고만 비교한다
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", data["answer"]) if s.strip()]
    groups = {}  # 조각 번호 -> (조각, 그 조각과 비교할 문장들)
    for sentence in sentences:
        for row in search_rows(db, user.id, filenames, sentence, top_k=1):
            groups.setdefault(row.chunk_id, (row, []))[1].append(sentence)

    mismatches = []
    source_names = {"resume": "이력서", "cl": "자기소개서"}
    for row, parts in groups.values():
        for mismatch in check_answer(" ".join(parts), [clean_evidence(row.content)]):
            mismatch["source"] = source_names.get(row.doctype, row.doctype)  # 화면에 어느 서류인지 표시
            mismatches.append(mismatch)
    return {"mismatches": mismatches}


# 1. 추출
def extract_text(pdf_bytes) -> str:
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # 메모리에서 엶
    except Exception:
        raise HTTPException(status_code=400, detail="손상되었거나 올바르지 않은 PDF 파일입니다.")

    if doc.is_encrypted:
        doc.close()
        raise HTTPException(status_code=400, detail="암호가 걸려 있는 PDF 파일은 업로드할 수 없습니다.")

    if doc.page_count > MAX_PDF_PAGES:
        page_cnt = doc.page_count
        doc.close()
        raise HTTPException(status_code=400, detail=f"PDF 파일은 최대 {MAX_PDF_PAGES}페이지 이하여야 합니다. (현재 {page_cnt}페이지)")

    if doc.page_count == 0:
        doc.close()
        raise HTTPException(status_code=400, detail="내용이 없는 빈 PDF 파일입니다.")

    try:
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
    finally:
        doc.close()

    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다. (스캔된 이미지 전용 PDF는 지원하지 않습니다)")

    return text.strip()


# 화면에 보여 줄 근거에서 제목과 마크다운 기호 지우기
def clean_evidence(text: str) -> str:
    text = re.sub(r"^\s*#+.*$", "", text, flags=re.M)  # # 제목 줄
    text = re.sub(r"^\s*\[[^\]\n]+\]\s*", "", text, flags=re.M)  # [지원 동기] 같은 제목
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()



# 4. 저장
def save_pdf(
    db,  user_id: str, doctype: str, filename: str, pdf_bytes: bytes, text: str
) -> str:  # db = Session()
    hash = hashlib.md5(pdf_bytes).hexdigest()[:8]  # 파일 지문
    hash_name = f"{hash}_{filename}"

    if db.query(PdfChunk).filter_by(user_id=user_id, doctype=doctype, filename=hash_name).first():
        return hash_name

    chunks = embed(chunking(text))
    for chunk in chunks:
        row = PdfChunk(
            user_id=user_id,
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
def search_rows(db: Session, user_id: str, filenames: list[str], query: str, top_k: int = 1) -> list[PdfChunk]:
    if not filenames or not user_id:
        return []

    query_vec = embed([{"content": query}])[0]["embedding"]
    return (
        db.query(PdfChunk)
        .filter(
            PdfChunk.user_id == user_id,
            PdfChunk.filename.in_(filenames)
        )
        .order_by(PdfChunk.embedding.cosine_distance(query_vec))
        .limit(top_k)
        .all()
    )


# 원문 청크 검색 (화면에 보여 줄 글만 돌려줌)
def search_chunks(db: Session, user_id: str, filenames: list[str], query: str, top_k: int = 1) -> list[str]:
    rows = search_rows(db, user_id, filenames, query, top_k)
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