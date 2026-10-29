import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.ai.cloud import generate_feedback, generate_spelling, generate_titles
from app.database import get_db
from app.models import CL, Answer

router = APIRouter()                            # 이 파일 주소들을 담을 상자
logger = logging.getLogger("uvicorn.error")     # 오류를 서버 터미널에 적는 도구

# 화면이 보내는 4칸의 모양
class ClSections(BaseModel):
    motivation: str = ""
    growth: str = ""
    personality: str = ""
    plan: str = ""

# 화면(Cl.jsx)의 GUEST_LIMIT, MEMBER_LIMIT과 같이 바꿀 것
GUEST_MAX_TOTAL_CHARS = 500  # 비회원 (화면 200자 + 공백 여유)
MEMBER_MAX_TOTAL_CHARS = 7000  # 회원 (화면 5000자 + 공백 여유)

# TODO: (auth.py) 작성 후 수정할 것
def check_length(sections, is_member=False):    # 글자 수 확인 (안 넘기면 비회원)
    data = sections.model_dump() # 4칸을 사전(dict)으로 바꿔주는 함수
    limit = MEMBER_MAX_TOTAL_CHARS if is_member else GUEST_MAX_TOTAL_CHARS
    if sum(len(v) for v in data.values()) > limit:
        raise HTTPException(status_code=400, detail="자기소개서가 너무 깁니다.")
    return data

# ---------------------------------------------------------------------------
# 자소서 첨삭
# ---------------------------------------------------------------------------

@router.post("/cl/feedback")
def cl_feedback(sections: ClSections):
    data = check_length(sections)                       # ① 글자 수 확인
    try:
        feedbacks, problems = generate_feedback(data)   # ② cloud.py 함수 부르기
    except ValueError as e:                             # ③ 빈 자소서
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 자소서는 개인정보라서 오류 종류만 기록한다
        logger.error("자소서 첨삭 실패: %s", type(e).__name__)
        raise HTTPException(status_code=502, detail="첨삭을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.")
    return {"feedbacks": feedbacks, "warnings": problems}

# ---------------------------------------------------------------------------
# 맞춤법 검사
# ---------------------------------------------------------------------------

@router.post("/cl/spellcheck")
def cl_spellcheck(sections: ClSections):
    data = check_length(sections)
    try:
        fixes, problems = generate_spelling(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("맞춤법 검사 실패: %s", type(e).__name__)
        raise HTTPException(status_code=502, detail="맞춤법 검사를 하지 못했습니다. 잠시 후 다시 시도해 주세요.")
    return {"feedbacks": fixes, "warnings": problems}

# ---------------------------------------------------------------------------
# 제목 추천
# ---------------------------------------------------------------------------

@router.post("/cl/title")
def cl_title(sections: ClSections):
    data = check_length(sections)
    try:
        titles, _ = generate_titles(data)     #  남은 문제는 안 써서 _로 받고 버림
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("제목 추천 실패: %s", type(e).__name__)
        titles = []            # 오류가 나면 "제목 0개"로 보고 아래에서 같이 처리
    if not titles:
        raise HTTPException(status_code=502, detail="제목을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.")
    return {"titles": titles}

# ---------------------------------------------------------------------------
# 마이페이지에 저장
# ---------------------------------------------------------------------------

# 제목 + 4칸 (ClSections를 그대로 물려받고 제목만 더함)
class ClSave(ClSections):
    title: str = ""

# 로그인(auth.py)이 완성되면 실제 사용자 id를 꺼내기
def current_user_id():
    return None

# CL 표(제목)와 ANSWER 표(4칸)에 한 번에 저장하고, 새 자소서 번호(cl_id)를 돌려준다
# 둘 중 하나라도 실패하면 둘 다 취소해서, 제목만 저장되는 반쪽 데이터를 막는다
def save_cl(db, user_id, title, sections):
    try:
        cl = CL(title=title, id=user_id)
        db.add(cl)
        db.flush()                                   # cl.cl_id에 번호가 생김 → 아직 확정 아님
        db.add(
            Answer(
                cl_id=cl.cl_id,
                motivation=sections["motivation"],
                growth=sections["growth"],
                personality=sections["personality"],
                plan=sections["plan"],
            )
        )
        db.commit()                                 # 진짜 확정 (이때 DB에 저장)
        return cl.cl_id
    except Exception:
        db.rollback()                               # 전부 취소 (없던 일로)
        raise


@router.post("/cl/save")
def cl_save(
    body: ClSave,
    db: Session = Depends(get_db),
    user_id=Depends(current_user_id),  # 로그인한 사용자 id
):
    # 1. 로그인하지 않았으면 401
    if user_id is None:
        raise HTTPException(status_code=401, detail="로그인 후 저장할 수 있어요.")
    # 2. 제목 확인 (DB의 제목 칸은 비울 수 없고 200자까지)
    title = body.title.strip()
    if not title or len(title) > 200:
        raise HTTPException(status_code=400, detail="제목을 1~200자로 입력해 주세요.")
    # 3. 글자 수 확인 (저장은 회원만 하니 회원 한도)
    sections = check_length(body, is_member=True)
    # 4. 저장 (실패해도 자소서 내용은 기록하지 않고 오류 종류만 남긴다)
    try:
        cl_id = save_cl(db, user_id, title, sections)
    except Exception as e:
        logger.error("자소서 저장 실패: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="저장하지 못했습니다.")
    return {"cl_id": cl_id}
