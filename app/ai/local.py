import os
import json
import re
import requests
from app.ai.local_prompts import (
  CHUNKING_PROMPT,
  QUESTION_COMMON,
  QUESTION_PROMPTS,
  FOLLOWUP_COMMON,
  FOLLOWUP_PROMPTS,
  JOB_UNIT_RULE,
  VERIFY_PROMPT,
  CONFLICT_PROMPT,
)

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:11434/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:11434/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "60"))
MAX_INPUT_TEXT_LENGTH = int(os.getenv("MAX_INPUT_TEXT_LENGTH", "10000"))

# 응답값의 boolean 변환
def to_bool(value, default: bool = False) -> bool:
  if isinstance(value, bool): #이미 True/False 상태
    return value
  if isinstance(value, (int, float)):
    return bool(value)
  if isinstance(value, str):
    val = value.strip().lower()
    if val in ("true", "1", "yes", "t", "y"):
      return True
    if val in ("false", "0", "no", "f", "n"):
      return False
  return default

# 참고 문장 필터링 (인적사항, 메타데이터 제외하고 서술형 줄글만)
def is_narrative_sentence(text: str) -> bool:
  s = text.strip()
  # 1. 너무 짧은 단어나 단순 소제목
  if len(s) < 15:
    return False
  # 2. 이메일, 전화번호, URL
  if re.search(r"@|\b(010|02|031)[-\s]?\d{3,4}[-\s]?\d{4}\b|https?://|www\.", s):
    return False
  # 3. 연락처, 이메일, 생년월일, 포트폴리오 등 인적사항
  if re.search(r"연락처|이메일|휴대전화|전화번호|생년월일|인적사항|github|portfolio", s, re.I):
    return False
  # 4. |로 구분된 헤더/메타데이터
  if "|" in s:
    return False
  return True

# 원문 문장 분리 및 정제
def clean_context_sentences(evidences: list[str], used_evidences: list[str] | None = None) -> tuple[list[str], str]:
  target_evidences = evidences or []
  if used_evidences and target_evidences:
    unused = [ev for ev in target_evidences if ev not in used_evidences]
    if unused:
      target_evidences = unused

  raw_sentences = [
    re.sub(r"\s+", " ", s).strip()
    for ev in target_evidences
    for s in re.split(r"(?<=[^0-9][.?!])\s+", ev)
    if s.strip()
  ]
  sentences = [s for s in raw_sentences if is_narrative_sentence(s)] or raw_sentences
  context = "\n\n".join(sentences) if sentences else ("\n\n".join(target_evidences) if target_evidences else "없음")
  return sentences, context

# LLM 호출
def call_llm(prompt: str, content: str, temperature: float = 0.1) -> dict:
  if content and len(content) > MAX_INPUT_TEXT_LENGTH:
    content = content[:MAX_INPUT_TEXT_LENGTH]

  res = requests.post(
    LLM_URL,
    json={
      "model": LLM_MODEL,
      "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
      ],
      "response_format": {"type": "json_object"}, #오직 JSON만 반환
      "temperature": temperature
    },
    timeout=AI_TIMEOUT
  )
  res.raise_for_status()
  raw_content = res.json()["choices"][0]["message"]["content"].replace("。", ".")
  try:
    return json.loads(raw_content)
  except json.JSONDecodeError:
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if match:
      return json.loads(match.group(0))
    raise

# 2. 청킹
def chunking(text):
  data = call_llm(CHUNKING_PROMPT, text)
  return data.get("chunks", [])

# 3. 임베딩
def embed(chunks):
  texts = [c["content"][:2000] for c in chunks]  # 임베딩 모델 컨텍스트 초과 방지
  res = requests.post(
    EMBED_URL,
    json={"model": EMBED_MODEL, "input": texts},
    timeout=AI_TIMEOUT
  )
  res.raise_for_status()
  embeddings = res.json()["data"]

  for chunk, emb in zip(chunks, embeddings):
    chunk["embedding"] = emb["embedding"] #[-0.02314, 0.04128, ..., 0.00891]

  return chunks

# 지원자 말투를 면접관 어투로 변환
def fix_applicant_tone(text: str) -> str:
  if not text:
    return ""
  s = str(text).strip()
  # 1. 어미 변환
  s = re.sub(r"지원하게\s*되었습니다", "지원하셨다고 작성해 주셨습니다", s)
  s = re.sub(r"지원하게\s*된\s*것입니다", "지원하셨다고 작성해 주셨습니다", s)
  s = re.sub(r"지원했습니다", "지원하셨다고 작성해 주셨습니다", s)

  # 2. 호칭 및 대명사
  s = re.sub(r"귀\s*사", "우리 회사", s)
  s = re.sub(r"\b저희\s*회사\b", "우리 회사", s)
  s = re.sub(r"(?<![가-힣])(?:제가|저는|저의|제)(?![가-힣])", "지원자님께서", s)

  # 3. 문법 및 오탈자
  s = s.replace("생각하신가요", "생각하시나요")
  s = s.replace("느끼신가요", "느끼시나요")
  return s.strip()

# 과거 시제 종결 확인(종성이 ㅆ(쌍시옷))
def has_ssangsiot_jongseong(ch: str) -> bool:
  if not ch or len(ch) != 1 or not ('\uac00' <= ch <= '\ud7a3'):
    return False
  return (ord(ch) - 0xAC00) % 28 == 20

# 배경 문장 및 면접관 어조 보정 (~이었습니다 -> ~이었다고 작성해 주셨습니다 등)
def format_background(bg: str | None) -> str:
  if not bg or str(bg).strip().lower() in ("null", "none", ""):
    return ""
  s = fix_applicant_tone(str(bg)).strip()
  if not s.endswith((".", "!", "?")):
    s += "."

  # 이미 인용 어투인 경우 유지
  if re.search(r"(?:작성|언급|소개|기재)(?:해\s*주|하)?셨(?:습니다|다고)", s):
    return s

  # 1. ~이었습니다. / ~였습니다. (체언 + 서술격조사 과거형)
  if re.search(r"이었습니다\.$", s):
    return re.sub(r"이었습니다\.$", "이었다고 작성해 주셨습니다.", s)
  if re.search(r"였습니다\.$", s):
    return re.sub(r"였습니다\.$", "였다고 작성해 주셨습니다.", s)
  if re.search(r"이었다\.$", s):
    return re.sub(r"이었다\.$", "이었다고 작성해 주셨습니다.", s)
  if re.search(r"였다\.$", s):
    return re.sub(r"였다\.$", "였다고 작성해 주셨습니다.", s)

  # 2. ~입니다. / ~이다.
  if re.search(r"([가-힣])입니다\.$", s):
    return re.sub(r"([가-힣])입니다\.$", r"\1이라고 작성해 주셨습니다.", s)
  if re.search(r"([가-힣])이다\.$", s):
    return re.sub(r"([가-힣])이다\.$", r"\1이라고 작성해 주셨습니다.", s)

  # 3. 모든 과거형 종결어미: ~[ㅆ받침]습니다. (했습니다, 었습니다, 았습니다, 되었습니다, 받았습니다, 배웠습니다 등)
  m = re.search(r"([가-힣])습니다\.$", s)
  if m and has_ssangsiot_jongseong(m.group(1)):
    return re.sub(r"([가-힣])습니다\.$", r"\1다고 작성해 주셨습니다.", s)

  # 평어체 과거형: ~[ㅆ받침]다. (했다, 었다, 았다, 됐다 등)
  m_plain = re.search(r"([가-힣])다\.$", s)
  if m_plain and has_ssangsiot_jongseong(m_plain.group(1)):
    return re.sub(r"([가-힣])다\.$", r"\1다고 작성해 주셨습니다.", s)

  # 4. 현재형: ~합니다. / ~한다.
  if re.search(r"합니다\.$", s):
    return re.sub(r"합니다\.$", "한다고 작성해 주셨습니다.", s)
  if re.search(r"한다\.$", s):
    return re.sub(r"한다\.$", "한다고 작성해 주셨습니다.", s)

  # 5. ~하셨습니다.
  if re.search(r"하셨습니다\.$", s):
    return re.sub(r"하셨습니다\.$", "하셨다고 작성해 주셨습니다.", s)

  return s

# LLM이 지정한 evidence를 매핑
def match_evidence(cited: str | None, sentences: list[str]) -> list[str]:
  if not cited or str(cited).strip().lower() in ("null", "none", ""):
    return []
  cited_str = str(cited).strip()
  # 1. 원문 문장 중 정확히 일치하거나 포함
  for s in sentences:
    if s == cited_str or cited_str in s or s in cited_str:
      return [s]
  # 2. 약간의 변형이 있더라도 LLM이 지정한 문장
  return [cited_str]

# AI가 질문을 주지 못했을 때 쓰는 단계별 기본 질문
DEFAULT_QUESTIONS = {
  "지원 동기": "우리 회사와 이 직무에 지원하게 된 계기를 말씀해 주시겠어요?",
  "경험 및 역량": "지원 직무와 관련해 가장 자신 있는 경험 하나를 소개해 주시겠어요?",
  "인성 및 협업": "팀으로 일하면서 의견이 달랐던 경험이 있다면, 어떻게 해결하셨나요?",
  "입사 후 포부": "입사 후 이 직무에서 이루고 싶은 목표는 무엇인가요?",
}
# 5. 질문
def ask_question(
  category: str,
  job: str,
  evidences: list[str],
  job_units: list[str] | None = None,
  previous_dialogues: list[dict] | None = None,
  used_evidences: list[str] | None = None,
) -> dict:
  prompt = f"{QUESTION_COMMON}\n\n{QUESTION_PROMPTS[category]}\n질문은 반드시 한국어로만 작성하세요.\n"
  sentences, context = clean_context_sentences(evidences, used_evidences)

  user_content = f"""
[지원 직무]
{job}

[참고 원문]
{context}
"""
  if previous_dialogues:
    dialogue_texts = []
    for d in previous_dialogues:
      q = str(d.get("question", "")).strip()
      a = str(d.get("answer", "")).strip()
      if q and a:
        dialogue_texts.append(f"Q: {q}\nA: {a}")
      elif q:
        dialogue_texts.append(f"Q: {q}")
    if dialogue_texts:
      dialogue_block = "\n\n".join(dialogue_texts)
      user_content += f"""
[이전 면접 대화]
{dialogue_block}
※ 주의: 위 [이전 면접 대화]에 이미 나온 질문 의도나 지원자가 이미 답변에서 밝힌 내용은 다시 묻지 말고, 완전히 다른 새로운 관점의 질문을 생성하세요.
"""

  if job_units:
    prompt += JOB_UNIT_RULE
    user_content += "\n[직무 역량]\n" + "\n".join(job_units) + "\n"

  result = call_llm(prompt, user_content, temperature=0.35)
  raw_bg = result.get("background")
  raw_q = str(result.get("question") or "")

  # 질문이 비었거나 한자가 섞이면 1번 다시 요청
  if not raw_q or re.search(r"[\u4e00-\u9fff]", raw_q) or (raw_bg and re.search(r"[\u4e00-\u9fff]", str(raw_bg))):
    result = call_llm(prompt, user_content, temperature=0.35)
    raw_bg = result.get("background")
    raw_q = str(result.get("question") or "")

  if not raw_q:
    raw_q = DEFAULT_QUESTIONS.get(category, "이 부분에 대해 조금 더 자세히 말씀해 주시겠어요?")

  # 말투 보정 및 결합
  clean_q = fix_applicant_tone(raw_q)
  clean_bg = format_background(raw_bg)

  # background가 비어있고 question에 2문장 이상 들어온 경우
  if not clean_bg:
    q_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_q) if s.strip()]
    if len(q_sentences) >= 2:
      clean_bg = format_background(" ".join(q_sentences[:-1]))
      clean_q = q_sentences[-1]

  if clean_bg:
    final_question = f"{clean_bg} {clean_q}"
  else:
    final_question = clean_q

  result["question"] = final_question

  # 참고 문장 매핑
  raw_cited = result.get("cited_evidence")
  result_sentences = match_evidence(raw_cited, sentences)

  # 폴백: evidence가 누락되었으나 background가 생성된 경우
  if not result_sentences and clean_bg and sentences:
    for s in sentences:
      if s in clean_bg or clean_bg in s:
        result_sentences = [s]
        break

  result["evidences"] = result_sentences
  result["used_evidence"] = bool(result_sentences)
  return result

# ---------------------------------------------------------------------------
# 심층 질문 생성 함수 (규칙 강제 및 결과 정제)
# 1. 지원 동기 / 입사 후 포부: 심층 질문 생략 (즉시 False 반환)
# 2. 경험 및 역량 / 인성 및 협업: 무조건 need_followup=True 필수 1개 보장
# ---------------------------------------------------------------------------
def ask_followup_question(
    category: str,
    job: str,
    question: str,
    answer: str,
    evidences: list[str] | None = None,
    used_evidences: list[str] | None = None,
) -> dict:
    """
    지원자의 직전 질문과 답변을 분석하여 심층 질문을 생성하거나 건너뜁니다.
    반환 형식: {"need_followup": bool, "question": str | None, "evidences": list[str]}
    """
    # 지원 동기와 입사 후 포부는 심층 질문 생략
    if category in ["지원 동기", "입사 후 포부"]:
        return {"need_followup": False, "question": None, "evidences": []}

    # 1. 해당 카테고리에 맞는 심층질문 프롬프트 준비
    prompt = f"{FOLLOWUP_COMMON}\n\n[면접 단계: {category}]\n\n{FOLLOWUP_PROMPTS.get(category, '')}\n질문은 반드시 한국어로만 작성하세요.\n"
    sentences, context = clean_context_sentences(evidences or [], used_evidences)

    # 2. LLM에 전달할 사용자 컨텍스트 구성
    user_content = f"""
[지원 직무]
{job}

[이전 면접 질문]
{question}

[지원자의 답변]
{answer}

[참고 원문]
{context}
"""

    # 3. 로컬 LLM 호출
    result = call_llm(prompt, user_content, temperature=0.35)

    # 4. 한자 섞임 방지 (한자 발견 시 1회 재호출)
    if result.get("question") and re.search(r"[\u4e00-\u9fff]", str(result["question"])):
        result = call_llm(prompt, user_content, temperature=0.35)

    followup_q = result.get("question")

    # 5. [규칙 보장] 경험/인성 카테고리는 필수 1개 생성 보장
    if not followup_q:
        if category == "경험 및 역량":
            followup_q = "답변해 주신 경험을 진행하시면서 왜 해당 방식을 선택하셨는지, 그리고 그 과정에서 얻은 구체적인 결과와 배운 점은 무엇이었나요?"
        else:
            followup_q = "그 조율 과정에서 해당 소통 방식을 선택하신 이유는 무엇이었으며, 그 경험을 통해 협업에 대해 어떤 점을 배우셨나요?"

    # 6. 말투 필터링 및 1~2문장 질문 정제
    followup_q = fix_applicant_tone(str(followup_q))
    q_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", followup_q) if s.strip()]
    # 3문장 이상 길어지면 최대 2문장 유지
    if len(q_sentences) > 2:
        followup_q = " ".join(q_sentences[-2:])
    else:
        followup_q = " ".join(q_sentences)

    # 7. 심층 질문의 참고 문장 매핑 (LLM이 cited_evidence를 지정한 경우)
    raw_cited = result.get("cited_evidence")
    followup_evidences = match_evidence(raw_cited, sentences)

    return {
        "need_followup": True,
        "question": followup_q,
        "evidences": followup_evidences,
    }



# AI 추출 결과의 코드 레벨 오탐(False Positive) 검증
def is_false_positive(mismatch: dict, context: str, answer_full: str) -> bool:
  doc = str(mismatch.get("document", "")).strip()
  ans = str(mismatch.get("answer", "")).strip()
  if not doc or not ans:
    return True

  norm_doc = re.sub(r"[\s\.,·~!?]", "", doc)
  norm_ans = re.sub(r"[\s\.,·~!?]", "", ans)
  norm_full = re.sub(r"[\s\.,·~!?]", "", answer_full)
  norm_ctx = re.sub(r"[\s\.,·~!?]", "", context)

  # 1. 기호/공백 제거 후 일치
  if norm_doc == norm_ans:
    return True

  # 2. 서류 구절(4자 이상)이 답변 속에 그대로 포함되어 있는 경우 (서류 내용을 그대로 답변한 경우)
  if len(norm_doc) >= 4 and (norm_doc in norm_ans or norm_doc in norm_full):
    return True

  # 3. 답변 구절(4자 이상)이 서류 원문에 그대로 존재하는 경우
  if len(norm_ans) >= 4 and norm_ans in norm_ctx:
    return True

  return False


# 답변과 서류 비교
def check_answer(answer: str, evidences: list[str]) -> list[dict]:
  if not evidences:
    return []

  context = "\n\n".join(evidences)
  user_content = f"""
[서류 원문]
{context}

[면접 답변]
{answer}
"""
  result = call_llm(VERIFY_PROMPT, user_content)
  mismatches = result.get("mismatches", [])
  return [
    m for m in mismatches
    if not is_false_positive(m, context, answer) and is_conflict(m, context, answer)
  ]


# 2차 확인: 두 내용이 동시에 사실일 수 없을 때만 불일치로 인정
# 짧은 구절만 보면 무슨 이야기인지 몰라서, 서류 원문과 답변 전체도 같이 보여 준다
def is_conflict(mismatch: dict, context: str, answer_full: str) -> bool:
  document = str(mismatch.get("document", "")).strip()
  answer = str(mismatch.get("answer", "")).strip()
  if not document or not answer:
    return False

  content = f"\n[서류 원문]\n{context}\n\n[면접 답변]\n{answer_full}\n\n서류: {document}\n답변: {answer}\n"
  result = call_llm(CONFLICT_PROMPT, content)
  return result.get("conflict") is True