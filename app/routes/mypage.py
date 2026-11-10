import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from fastapi import UploadFile, File, Response

from app.database import get_db
from app.models import (
    User,
    Resume,
    Career,
    Certificate,
    Interview,
    Dialogue,
    CL,
    Answer,
    PdfChunk,
)
from app.routes.auth import get_current_user
from app.routes.cl import ClSave, check_length

router = APIRouter(prefix="/mypage")  # 이 파일 주소는 모두 /mypage로 시작
logger = logging.getLogger("uvicorn.error")  # 오류를 서버 터미널에 적는 도구

# ---------------------------------------------------------------------------
# 자기소개서
# ---------------------------------------------------------------------------


# 내 자소서만 찾는다 (남의 것이면 없는 것처럼 404)
def find_cl(db, user_id, cl_id):
    cl = db.query(CL).filter(CL.cl_id == cl_id, CL.id == user_id).first()

    if not cl:
        raise HTTPException(status_code=404, detail="자기소개서를 찾을 수 없어요.")
    return cl


# CL 표(제목)와 ANSWER 표(4칸)를 합쳐서 화면에 줄 모양으로
def cl_to_dict(db, cl):
    answer = db.query(Answer).filter(Answer.cl_id == cl.cl_id).first()

    return {
        "cl_id": cl.cl_id,
        "title": cl.title,
        "motivation": answer.motivation if answer else "",
        "experience": answer.experience if answer else "",
        "personality": answer.personality if answer else "",
        "plan": answer.plan if answer else "",
    }


@router.get("/cl")
def list_cls(db: Session = Depends(get_db), user: User = Depends(get_current_user)):

    cls = (
        db.query(CL).filter(CL.id == user.id).order_by(CL.cl_id.desc()).all()
    )  # 최신순
    return [cl_to_dict(db, cl) for cl in cls]


@router.put("/cl/{cl_id}")
def update_cl(
    cl_id: int,
    body: ClSave,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):

    cl = find_cl(db, user.id, cl_id)

    # 저장(cl.py)과 같은 규칙: 제목 1~200자, 회원 글자 수 한도
    title = body.title.strip()
    if not title or len(title) > 200:
        raise HTTPException(status_code=400, detail="제목을 1~200자로 입력해 주세요.")
    sections = check_length(body, is_member=True)

    answer = db.query(Answer).filter(Answer.cl_id == cl.cl_id).first()
    if not answer:  # 4칸이 없으면 새로 만든다
        answer = Answer(cl_id=cl.cl_id)
        db.add(answer)

    cl.title = title
    for key, value in sections.items():  # motivation, experience, personality, plan
        setattr(answer, key, value)

    db.commit()
    return cl_to_dict(db, cl)


@router.delete("/cl/{cl_id}")
def delete_cl(
    cl_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    cl = find_cl(db, user.id, cl_id)

    db.query(Answer).filter(Answer.cl_id == cl.cl_id).delete()  # 자식(4칸) 먼저
    db.delete(cl)
    db.commit()
    return {"message": "삭제했어요."}


# ---------------------------------------------------------------------------
# 이력서
# ---------------------------------------------------------------------------


# 글자 수 한도는 models.py의 칸 크기와 같게 (넘으면 FastAPI가 422로 막아줌)
class CareerIn(BaseModel):
    company: str = Field("", max_length=100)
    role: str = Field("", max_length=500)
    start_year: str = Field("", max_length=7)  # 예) 2024.03
    end_year: str = Field("", max_length=7)


class CertificateIn(BaseModel):
    cert: str = Field(..., min_length=1, max_length=100)
    org: str = Field("", max_length=100)


class ResumeIn(BaseModel):
    name: str = Field(..., max_length=50)
    school: str = Field(..., max_length=100)
    major: str = Field(..., max_length=100)
    skill: str = Field("", max_length=255)
    careers: List[CareerIn] = []
    certificates: List[CertificateIn] = []


def resume_to_dict(db, resume):
    careers = (
        db.query(Career).filter(Career.id == resume.id).order_by(Career.career_id).all()
    )
    certs = (
        db.query(Certificate)
        .filter(Certificate.id == resume.id)
        .order_by(Certificate.cert_id)
        .all()
    )

    return {
        "name": resume.name,
        "school": resume.school,
        "major": resume.major,
        "skill": resume.skill or "",
        "has_portfolio": resume.portfolio is not None,  # 포트폴리오 등록 여부 (True/False)
        "careers": [
            {
                "company": c.company,
                "role": c.role,
                "start_year": c.start_year,
                "end_year": c.end_year,
            }
            for c in careers
        ],
        "certificates": [{"cert": c.cert, "org": c.org} for c in certs],
    }


@router.get("/resume")
def get_resume(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    resume = db.query(Resume).filter(Resume.id == user.id).first()
    if not resume:
        return None  # 아직 안 썼으면 빈 값 → 화면은 빈 칸으로
    return resume_to_dict(db, resume)


# 이력서는 한 사람당 하나: 있으면 덮어쓰고, 없으면 새로 만든다
# 경력·자격증은 개수가 바뀔 수 있어서 전부 지우고 받은 대로 다시 넣는다
@router.put("/resume")
def save_resume(
    body: ResumeIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        resume = db.query(Resume).filter(Resume.id == user.id).first()
        if not resume:
            resume = Resume(id=user.id)
            db.add(resume)
        resume.name = body.name
        resume.school = body.school
        resume.major = body.major
        resume.skill = body.skill
        db.flush()  # 이력서가 먼저 있어야 경력·자격증을 붙일 수 있음
        db.query(Career).filter(Career.id == user.id).delete()
        db.query(Certificate).filter(Certificate.id == user.id).delete()

        for c in body.careers:
            db.add(Career(id=user.id, **c.model_dump()))
        for c in body.certificates:
            db.add(Certificate(id=user.id, **c.model_dump()))
        db.commit()

    except Exception as e:
        db.rollback()  # 반쪽만 저장되지 않게 전부 취소
        logger.error("이력서 저장 실패: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="저장하지 못했습니다.")
    return resume_to_dict(db, resume)


# 포트폴리오 PDF 업로드 (POST /mypage/resume/portfolio)
@router.post("/resume/portfolio")
def upload_portfolio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="포트폴리오는 PDF 파일만 업로드할 수 있습니다.")

    content = file.file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB 제한
        raise HTTPException(status_code=400, detail="포트폴리오 파일은 20MB 이하여야 합니다.")

    resume = db.query(Resume).filter(Resume.id == user.id).first()
    if not resume:  # 이력서가 아직 없으면 빈 이력서를 만든다
        resume = Resume(id=user.id)
        db.add(resume)

    resume.portfolio = content  # DB의 LargeBinary 칸에 파일 내용을 그대로 저장
    db.commit()
    return {"message": "포트폴리오가 성공적으로 등록되었습니다."}


# 포트폴리오 열람 (GET /mypage/resume/portfolio)
@router.get("/resume/portfolio")
def download_portfolio(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    resume = db.query(Resume).filter(Resume.id == user.id).first()
    if not resume or not resume.portfolio:
        raise HTTPException(status_code=404, detail="등록된 포트폴리오가 없습니다.")

    return Response(
        content=resume.portfolio,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=portfolio.pdf"},
    )


# 포트폴리오 삭제 (DELETE /mypage/resume/portfolio)
@router.delete("/resume/portfolio")
def delete_portfolio(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    resume = db.query(Resume).filter(Resume.id == user.id).first()
    if resume and resume.portfolio:
        resume.portfolio = None
        db.commit()
    return {"message": "포트폴리오가 삭제되었습니다."}


# ---------------------------------------------------------------------------
# 면접 기록
# ---------------------------------------------------------------------------


def find_interview(db, user_id, interview_id):
    interview = (
        db.query(Interview)
        .filter(Interview.interview_id == interview_id, Interview.id == user_id)
        .first()
    )
    if not interview:
        raise HTTPException(status_code=404, detail="면접 기록을 찾을 수 없어요.")
    return interview


@router.get("/interview")
def list_interviews(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    interviews = (
        db.query(Interview)
        .filter(Interview.id == user.id)
        .order_by(Interview.created_at.desc())
        .all()
    )

    return [
        {
            "interview_id": i.interview_id,
            "created_at": i.created_at,
            "cat_major": i.cat_major,
            "cat_minor": i.cat_minor,
        }
        for i in interviews
    ]


@router.get("/interview/{interview_id}")
def get_interview(
    interview_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    interview = find_interview(db, user.id, interview_id)

    dialogues = (
        db.query(Dialogue)
        .filter(Dialogue.interview_id == interview_id)
        .order_by(Dialogue.dialogue_id)
        .all()
    )

    return {
        "interview_id": interview.interview_id,
        "created_at": interview.created_at,
        "cat_major": interview.cat_major,
        "cat_minor": interview.cat_minor,
        "dialogues": [{"question": d.question, "answer": d.answer} for d in dialogues],
    }


@router.delete("/interview/{interview_id}")
def delete_interview(
    interview_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    interview = find_interview(db, user.id, interview_id)
    db.query(Dialogue).filter(
        Dialogue.interview_id == interview_id
    ).delete()  # 자식(대화) 먼저
    db.delete(interview)
    db.commit()
    return {"message": "삭제했어요."}


# ---------------------------------------------------------------------------
# 회원정보 수정
# ---------------------------------------------------------------------------


class AgreementIn(BaseModel):
    save_agreed: bool


@router.patch("/agreement")
def update_agreement(
    body: AgreementIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    user.save_agreed = body.save_agreed
    db.commit()
    return {"save_agreed": user.save_agreed}


# 회원 탈퇴: 내 데이터를 자식 표부터 전부 지우고 마지막에 회원을 지운다
@router.delete("/me")
def withdraw(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    uid = user.id
    try:
        # ANSWER, DIALOGUE에는 회원 id가 없어서 자소서·면접 번호부터 모은다
        cl_ids = [row.cl_id for row in db.query(CL.cl_id).filter(CL.id == uid)]
        interview_ids = [
            row.interview_id
            for row in db.query(Interview.interview_id).filter(Interview.id == uid)
        ]

        db.query(Answer).filter(Answer.cl_id.in_(cl_ids)).delete()  # 자소서 내용
        db.query(CL).filter(CL.id == uid).delete()  # 자소서
        db.query(Dialogue).filter(
            Dialogue.interview_id.in_(interview_ids)
        ).delete()  # 대화
        db.query(Interview).filter(Interview.id == uid).delete()  # 면접
        db.query(Career).filter(Career.id == uid).delete()  # 경력
        db.query(Certificate).filter(Certificate.id == uid).delete()  # 자격증
        db.query(Resume).filter(Resume.id == uid).delete()  # 이력서
        db.query(PdfChunk).filter(PdfChunk.user_id == uid).delete()  # PDF 청크 (탈퇴 시 개인정보 및 외래키 정리)
        db.delete(user)  # 회원
        db.commit()
    except Exception as e:
        db.rollback()  # 하나라도 실패하면 전부 취소
        logger.error("회원 탈퇴 실패: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="탈퇴하지 못했습니다.")
    return {"message": "탈퇴가 완료되었습니다."}
