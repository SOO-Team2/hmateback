import hashlib
import pymupdf
import pymupdf4llm
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PdfChunk
from app.ai.local import chunking, embed, ask_question

router = APIRouter()

@router.post("/question")
def first_question(
    job: str = Form(...),
    resume: UploadFile = File(None),
    resume_saved: str = Form(None),
    cl: UploadFile = File(None),
    cl_saved: str = Form(None),
    db: Session = Depends(get_db),
):
    filenames = []

    if resume:
        resume_bytes = resume.read()
        resume_text = extract_text(resume_bytes)
        filenames.append(save_pdf(db, "resume", resume.filename, resume_bytes, resume_text))
    elif resume_saved:
        filenames.append(save_pdf(db, "resume", "resume_saved", resume_saved.encode(), resume_saved))

    if cl:
        cl_bytes = cl.read()
        cl_text = extract_text(cl_bytes)
        filenames.append(save_pdf(db, "cl", cl.filename, cl_bytes, cl_text))
    elif cl_saved:
        filenames.append(save_pdf(db, "cl", "cl_saved", cl_saved.encode(), cl_saved))

    evidences = search_chunks(db, filenames, f"{job} 지원 동기")
    result = ask_question("지원 동기", job, evidences)
    return {
        "question": result["question"],
        "evidences": evidences,
        "filenames": filenames,
    }  # 프론트에 전달


@router.post("/answer")
def submit_answer(data: dict, db: Session = Depends(get_db)):
    filenames = data.get("filenames", [])
    evidences = search_chunks(db, filenames, f"{data['job']} {data['category']}")
    result = ask_question(data["category"], data["job"], evidences)
    return {
        "question": result["question"],
        "evidences": evidences,
    }  # 다음 카테고리 질문


# 1. 추출
def extract_text(pdf_bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # 메모리에서 엶
    text = pymupdf4llm.to_markdown(doc)  # 폰트 크기 등 표시
    doc.close()
    return text.strip()


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


# 원문 청크 검색
def search_chunks(db: Session, filenames: list[str], query: str, top_k: int = 1) -> list[str]:
    if not filenames:
        return []

    query_vec = embed([{"content": query}])[0]["embedding"]
    rows = (
        db.query(PdfChunk)
        .filter(PdfChunk.filename.in_(filenames))
        .order_by(PdfChunk.embedding.cosine_distance(query_vec))
        .limit(top_k)
        .all()
    )
    return [row.content for row in rows]
