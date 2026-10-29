import re
import difflib
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# 공통 설정
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-mini"
MAX_RETRIES = 2 # 처음1번 + 재시도 2번(총 3번 요청)
TIMEOUT_SECONDS = 60 # 응답 60초만 기다림

KEYS = ["motivation", "growth", "personality", "plan"]

# ---------------------------------------------------------------------------
# 공통 도구 (세 기능이 같이 씀)
# ---------------------------------------------------------------------------

# ((?<=[.!?])\s+) 기준 문장 나누기
def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())

# 문장마다 [문단-문장] 번호 붙이기
def number_cl(sections):
    lines = []
    index = {}
    for i, key in enumerate(KEYS, 1):
        text = sections.get(key, "").strip()
        if not text:
            continue
        for j, sentence in enumerate(split_sentences(text), 1):
            sid = f"{i}-{j}"
            index[sid] = sentence
            lines.append(f"[{sid}] {sentence}")
    return "\n".join(lines), index

# OpenAI 연결(한 번!)
_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(timeout=TIMEOUT_SECONDS)
    return _client

# AI에게 묻고 → 검사 → 틀리면 다시 묻기 (최대 3번, 세 기능 공통)
def ask_ai(base_messages, text_format, check_fn):

    messages = base_messages
    result, problems = None, []

    for _ in range(MAX_RETRIES + 1):                     # 최대 3번
        response = get_client().responses.parse(         # AI에게 요청
            model=MODEL,                                 # 어떤 AI에게
            input=messages,                              # 무엇을
            text_format=text_format,                     # 어떤 틀로 답할지
        )
        result = response.output_parsed                  # 틀에 맞춰 정리된 답 꺼내기
        if result is not None:
            problems = check_fn(result)                  # 답이 있으면 검사
        else:
            problems = ["모델이 정해진 형식으로 답하지 않았다."]
        if not problems:
            break                                        # 문제 없으면 끝

        messages = base_messages + [                     # 다시 물을 준비
            {
                "role": "user",
                "content": "이전 시도의 문제점이다. 이를 피해서 처음부터 다시 만들어라.\n- "
                + "\n- ".join(problems),
            }
        ]
    return result, problems

# ---------------------------------------------------------------------------
# 자소서 첨삭
#
# 자소서를 [문단-문장] 번호로 보내고, 모델은 "어느 문장을 어떻게 고칠지"를 번호로 답한다.
# ---------------------------------------------------------------------------

# 첨삭 3~5개
MIN_FEEDBACKS, MAX_FEEDBACKS = 3, 5

# AI 답의 틀
class FeedbackItem(BaseModel):   # tag(둘 중 하나) + 문장 번호 + 수정안 + 이유
    tag: Literal["수정 필요", "문장 다듬기"]
    sentence_id: str
    suggestion: str
    reason: str

class FeedbackSet(BaseModel):     # 첨삭 목록 담는 상자
    feedbacks: list[FeedbackItem]

def feedback_prompt(numbered, sentence_count):
    minimum = min(MIN_FEEDBACKS, sentence_count)
    maximum = min(MAX_FEEDBACKS, sentence_count)
    system = (
        "너는 지원자의 자기소개서를 고쳐 주는 첨삭 전문가다. "
        f"자기소개서에서 고치면 좋은 문장을 {minimum}개에서 {maximum}개 고른다. "
        "각 항목은 문장 번호 하나(sentence_id)를 대상으로 하고, 그 문장을 고친 문장(suggestion)과 이유(reason)를 쓴다. "
        "서로 다른 문장을 고르고, 이미 좋은 문장은 건드리지 않는다. "
        "원문에 없는 사실, 경험, 숫자를 새로 만들어 넣지 않는다. 원문의 숫자와 고유명사는 그대로 둔다. "
        "tag 기준: "
        "수정 필요는 글의 구성이나 내용의 문제(예: 지원 이유가 뒤에 있음, 근거가 부족함), "
        "문장 다듬기는 표현의 문제(예: 너무 긴 문장, 어색한 표현)다. "
        "수정안은 자기소개서에 어울리는 '~합니다' 체로 쓴다. "
        "수정 이유는 한두 문장으로, '~해요'로 끝나는 부드러운 말투로 쓴다. "
        "suggestion과 reason에는 [2-1] 같은 문장 번호를 쓰지 않는다. "
    )
    user = f"[자기소개서]\n{numbered}\n\n고치면 좋은 문장을 골라 첨삭해라."
    return [
        {"role": "system", "content": system},      # 1장: 규칙
        {"role": "user", "content": user},          # 2장: 자소서
    ]

# 첨삭 하나 검사
def feedback_problem(f, index):
    original = index.get(f.sentence_id)
    if original is None:
        return f"문장 번호 [{f.sentence_id}]는 자소서에 없는 번호다."
    if f.suggestion.strip() == original.strip():
        return f"[{f.sentence_id}]의 수정안이 원문과 같다."
    if not f.reason.strip():
        return f"[{f.sentence_id}]에 수정 이유가 없다."
    if sorted(re.findall(r"\d+", f.suggestion)) != sorted(re.findall(r"\d+", original)):
        return f"[{f.sentence_id}]의 수정안에서 숫자가 원문과 다르다. 숫자는 그대로 둬야 한다."
    return None

# 답 전체 검사
def check_feedback(result, index):

    problems = []
    items = result.feedbacks

    minimum = min(MIN_FEEDBACKS, len(index))
    maximum = min(MAX_FEEDBACKS, len(index))
    if not minimum <= len(items) <= maximum:
        problems.append(
            f"첨삭이 {len(items)}개다. {minimum}개에서 {maximum}개여야 한다."
        )

    ids = [f.sentence_id for f in items]                    # ids : 번호들
    dups = sorted({i for i in ids if ids.count(i) > 1})     # duplicates(중복)
    if dups:
        problems.append(f"같은 문장을 두 번 고쳤다: {dups}.")

    for f in items:
        problem = feedback_problem(f, index)
        if problem:
            problems.append(problem)

    return problems

def generate_feedback(sections):

    numbered, index = number_cl(sections)           # 1. 번호 붙이기
    if not index:
        raise ValueError("자소서가 비어 있다.")
    # 2. ask_ai에 지시문, FeedbackSet, 검사 방법을 넘긴다.
    result, problems = ask_ai(
        feedback_prompt(numbered, len(index)),
        FeedbackSet,
        lambda r: check_feedback(r, index),
    )
    if result is None:
        return [], problems
    # 3. 통과한 것만 카드로
    feedbacks = []
    for f in result.feedbacks:
        if feedback_problem(f, index):
            continue
        paragraph = int(f.sentence_id.split("-")[0])
        feedbacks.append(
            {
                "tag": f.tag,
                "sentence_id": f.sentence_id,
                "section": KEYS[paragraph - 1],
                "original": index[f.sentence_id],
                "suggestion": f.suggestion,
                "reason": f.reason,
            }
        )
    return feedbacks, problems

# ---------------------------------------------------------------------------
# 맞춤법 검사
# ---------------------------------------------------------------------------

# 원문에서 5글자 넘게 바뀌면 거절 (임의로 정함)
MAX_CHANGE = 5

# AI 답의 틀
class SpellingItem(BaseModel):
    sentence_id: str
    corrected: str

class SpellingSet(BaseModel):
    fixes: list[SpellingItem]

def spelling_prompt(numbered):
    system = (
        "너는 한국어 맞춤법 검사기다. 자기소개서에서 맞춤법, 띄어쓰기, 오탈자가 틀린 문장만 골라 고친다. "
        "틀린 부분만 고치고, 단어 선택, 문장 구조, 말투, 의미는 바꾸지 않는다. "
        "맞춤법이 틀리지 않은 문장은 결과에 넣지 않는다. 틀린 곳이 없으면 fixes를 빈 목록으로 답한다. "
        "각 항목은 문장 번호(sentence_id)와, 그 문장 전체를 고친 문장(corrected)이다. "
        "숫자와 고유명사는 그대로 둔다. corrected에는 [2-1] 같은 문장 번호를 쓰지 않는다. "
    )
    user = f"[자기소개서]\n{numbered}\n\n맞춤법이 틀린 문장만 골라 고쳐라."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# 바뀐 글자수 세기
def changed_chars(a, b):
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")

# 하나 검사
def spelling_problem(f, index):
    original = index.get(f.sentence_id)
    if original is None:
        return f"문장 번호 [{f.sentence_id}]는 자소서에 없는 번호다."
    if f.corrected.strip() == original.strip():
        return f"[{f.sentence_id}]의 수정안이 원문과 같다. 틀린 곳이 없으면 결과에 넣지 마라."
    if sorted(re.findall(r"\d+", f.corrected)) != sorted(re.findall(r"\d+", original)):
        return f"[{f.sentence_id}]의 수정안에서 숫자가 원문과 다르다. 숫자는 그대로 둬야 한다."
    if changed_chars(original, f.corrected) > MAX_CHANGE:
        return f"[{f.sentence_id}]의 수정안이 원문과 너무 다르다. 맞춤법과 띄어쓰기만 고쳐야 한다."
    return None

# 답 전체를 검사(문제를 모아서 AI에게 재요청)
def check_spelling(result, index):
    problems = []
    ids = [f.sentence_id for f in result.fixes]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        problems.append(f"같은 문장을 두 번 고쳤다: {dups}.")
    for f in result.fixes:
        problem = spelling_problem(f, index)
        if problem:
            problems.append(problem)
    return problems

def generate_spelling(sections):
    numbered, index = number_cl(sections)           # 1. 번호 붙이기
    if not index:
        raise ValueError("자소서가 비어 있다.")
    # ask_ai에 지시문, SpellingSet, 검사 방법을 넘기기
    result, problems = ask_ai(
        spelling_prompt(numbered),
        SpellingSet,
        lambda r: check_spelling(r, index),
    )
    if result is None:
        return [], problems
    # 통과한 것만 카드로. 이유는 고정 문구.
    fixes = []
    for f in result.fixes:
        if spelling_problem(f, index):  #  "문제 있는 항목은 화면에 안 보내기"
            continue
        paragraph = int(f.sentence_id.split("-")[0])
        original = index[f.sentence_id]
        fixes.append(
            {
                "tag": "맞춤법",
                "sentence_id": f.sentence_id,
                "section": KEYS[paragraph - 1],
                "original": original,
                "suggestion": f.corrected,
                "reason": "맞춤법을 바로잡았어요.",
            }
        )
    return fixes, problems

# ---------------------------------------------------------------------------
# 제목 추천
# ---------------------------------------------------------------------------
# 20자 이내, 1개
MAX_TITLE_CHARS, TITLE_COUNT = 20, 1

# AI 답의 틀
class TitleSet(BaseModel):
    titles: list[str]

# 제목 다듬기
def clean_title(title):
    title = re.sub(r"\s+", " ", title).strip()  # 여러 칸 공백을 한 칸으로
    title = title.strip("\"'“”‘’「」『』")       # 앞뒤 따옴표 떼기
    return title.rstrip(".。!?").strip()        # 끝의 마침표, 느낌표, 물음표 떼기

def title_prompt(text):
    system = (
        f"너는 자기소개서에 어울리는 제목을 지어 주는 전문가다. 자기소개서를 읽고 제목 후보를 {TITLE_COUNT}개 만든다. "
        f"제목은 한 줄이고 {MAX_TITLE_CHARS}자 이내로 짧게 쓴다. "
        "자기소개서의 핵심 경험이나 지원 동기가 드러나게 쓴다. "
        "자기소개서에 없는 사실, 경력, 숫자를 만들어 넣지 않는다. "
        "서로 다른 방향으로 쓰고, 같은 제목을 반복하지 않는다. "
        "따옴표, 마침표, 이모지는 쓰지 않는다. "
    )
    user = f"[자기소개서]\n{text}\n\n제목 후보 {TITLE_COUNT}개를 만들어라."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# 제목 하나 검사
def title_problem(title, source):

    if not title:
        return "빈 제목이 있다."
    if len(title) > MAX_TITLE_CHARS:
        return f"'{title}'은 {len(title)}자다. {MAX_TITLE_CHARS}자 이내여야 한다."
    if set(re.findall(r"\d+", title)) - set(re.findall(r"\d+", source)):
        return f"'{title}'에 자소서에 없는 숫자가 들어 있다."
    return None

# 답 전체 검사
def check_titles(result, source):
    problems = []
    valid = 0
    for t in (clean_title(x) for x in result.titles):
        problem = title_problem(t, source)
        if problem:
            problems.append(problem)
            continue
        valid += 1
    if valid < TITLE_COUNT:
        problems.append(f"쓸 수 있는 제목이 {valid}개다. {TITLE_COUNT}개여야 한다.")
    return problems

def generate_titles(sections):
    _, index = number_cl(sections)
    if not index:
        raise ValueError("자소서가 비어 있다.")
    source = "\n".join(index.values())      # 번호 붙은 글은 안 쓴다
    # ask_ai에 지시문, TitleSet, 검사 방법을 넘긴다.
    result, problems = ask_ai(
        title_prompt(source),
        TitleSet,
        lambda r: check_titles(r, source),
    )
    if result is None:
        return [], problems

    titles = []
    for t in (clean_title(x) for x in result.titles):
        if title_problem(t, source):
            continue
        titles.append(t)
    return titles[:TITLE_COUNT], problems
