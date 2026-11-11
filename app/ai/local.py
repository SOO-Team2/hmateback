import os
import json
import re
import requests
from app.ai.local_prompts import (
  CHUNKING_PROMPT,
  QUESTION_PROMPTS,
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

# 2. 청킹
def chunking(text):
  if text and len(text) > MAX_INPUT_TEXT_LENGTH:
    text = text[:MAX_INPUT_TEXT_LENGTH]

  res = requests.post(
    LLM_URL,
    json={
      "model": LLM_MODEL,
      "messages": [
        {"role": "system", "content": CHUNKING_PROMPT},
        {"role": "user", "content": text},
      ],
      "response_format": {"type": "json_object"}, #오직 JSON만 반환
      "temperature": 0.1
    },
    timeout=AI_TIMEOUT
  )
  res.raise_for_status()
  content = res.json()["choices"][0]["message"]["content"]
  try:
    data = json.loads(content)
  except json.JSONDecodeError:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
      data = json.loads(match.group(0))
    else:
      raise
  for c in data["chunks"]:
    c["content"] = c["content"].replace("。", ".")
  return data["chunks"]

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

# LLM 호출
def call_llm(prompt: str, content: str) -> dict:
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
      "response_format": {"type": "json_object"},
      "temperature": 0.3
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

# 5. 질문
def ask_question(
  category: str,
  job: str,
  evidences: list[str],
  job_units: list[str] | None = None,
  previous_questions: list[str] | None = None,
) -> dict:
  prompt = QUESTION_PROMPTS[category] + "\n질문은 반드시 한국어로만 쓰고, 중국어와 한자는 쓰지 말 것.\n"
  context = "\n\n".join(evidences) if evidences else "없음"

  user_content = f"""
[지원 직무]
{job}

[참고 원문]
{context}
"""
  if previous_questions:
    prev_q_text = "\n".join(f"- {q}" for q in previous_questions if q)
    user_content += f"""
[이전 질문 내역]
{prev_q_text}
※ 주의: 위 [이전 질문 내역]에 이미 출제된 질문들과 질문 의도나 내용이 겹치지 않도록, 완전히 다른 관점의 새로운 질문을 생성하세요.
"""

  if job_units:
    prompt += JOB_UNIT_RULE
    user_content += "\n[직무 역량]\n" + "\n".join(job_units) + "\n"

  result = call_llm(prompt, user_content)
  if re.search(r"[\u4e00-\u9fff]", result["question"]):
    result = call_llm(prompt, user_content)
  result["question"] = re.sub(r"귀\s*사", "우리 회사", result["question"])
  result["question"] = result["question"].replace("생각하신가요", "생각하시나요")

  # 질문과 가까운 원문 1문장 고르기
  sentences = [
    s.strip()
    for ev in evidences
    for s in re.split(r"(?<=[^0-9][.?!])\s+", ev)
    if s.strip()
  ]
  if len(sentences) > 1:
    items = embed([{"content": result["question"]}] + [{"content": s} for s in sentences])
    q_vec = items[0]["embedding"]
    best = max(
      items[1:],
      key=lambda item: sum(a * b for a, b in zip(q_vec, item["embedding"])),
    )
    sentences = [best["content"]]

  result["evidences"] = sentences
  return result

# ---------------------------------------------------------------------------
# 심층 질문 생성 함수 (규칙 강제 및 결과 정제)
# 1. 지원동기 / 입사 후 포부: 답변이 충분하면 need_followup=False, 부족할 때만 질문 생성
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
    # 1. 해당 카테고리에 맞는 심층질문 프롬프트 준비
    prompt = FOLLOWUP_PROMPTS.get(category, "")
    prompt += "\n질문은 반드시 한국어로만 쓰고, 중국어와 한자는 쓰지 말 것.\n"

    # 이미 사용한 참고 문장 제외
    sentences = [
        s.strip()
        for ev in (evidences or [])
        for s in re.split(r"(?<=[^0-9][.?!])\s+", ev)
        if s.strip()
    ]
    used_set = {s.strip() for s in (used_evidences or []) if s.strip()}
    other_sentences = [s for s in sentences if s not in used_set]

    # 2. LLM에 전달할 사용자 컨텍스트 구성
    user_content = f"""
[지원 직무]
{job}

[이전 면접 질문]
{question}

[지원자의 답변]
{answer}
"""
    if category in ["경험 및 역량", "인성 및 협업"]:
        used_text = "\n".join(used_evidences) if used_evidences else "없음"
        other_text = "\n".join(other_sentences) if other_sentences else "없음"
        user_content += f"""
[이전 질문의 참고 문장]
{used_text}

[서류의 다른 원문]
{other_text}
"""
    elif evidences:
        user_content += f"""
[참고 원문]
{"\n\n".join(evidences)}
"""

    # 3. 로컬 LLM 호출
    result = call_llm(prompt, user_content)

    # 4. 한자 섞임 방지 (한자 발견 시 1회 재호출)
    if result.get("question") and re.search(r"[\u4e00-\u9fff]", str(result["question"])):
        result = call_llm(prompt, user_content)

    need_followup = result.get("need_followup", False)
    followup_q = result.get("question")

    # 5. [규칙 보장] 경험/인성 카테고리는 필수 1개 강제 (LLM이 false를 냈을 경우 대비한 안전장치)
    if category in ["경험 및 역량", "인성 및 협업"]:
        need_followup = True
        if not followup_q:
            if category == "경험 및 역량":
                followup_q = "답변해 주신 경험을 진행하시면서 왜 해당 방식을 선택하셨는지, 그리고 그 과정에서 얻은 구체적인 결과와 배운 점은 무엇이었나요?"
            else:
                followup_q = "그 조율 과정에서 해당 소통 방식을 선택하신 이유는 무엇이었으며, 그 경험을 통해 협업에 대해 어떤 점을 배우셨나요?"

    # 6. 지원동기 / 포부에서 심층질문이 필요 없다고 판단한 경우 question을 null로 정리
    if not need_followup:
        return {"need_followup": False, "question": None, "evidences": []}

    # 7. 말투 필터링 ('귀사' -> '우리 회사', '생각하신가요' -> '생각하시나요')
    if followup_q:
        followup_q = re.sub(r"귀\s*사", "우리 회사", str(followup_q))
        followup_q = followup_q.replace("생각하신가요", "생각하시나요").strip()

    # 8. 다른 경험을 질문한 경우, 새 참고 문장 선택
    followup_evidences = []
    if (
        category in ["경험 및 역량", "인성 및 협업"]
        and other_sentences
        and followup_q
    ):
        if len(other_sentences) > 1:
            items = embed([{"content": followup_q}] + [{"content": s} for s in other_sentences])
            q_vec = items[0]["embedding"]
            best = max(
                items[1:],
                key=lambda item: sum(a * b for a, b in zip(q_vec, item["embedding"])),
            )
            followup_evidences = [best["content"]]
        else:
            followup_evidences = [other_sentences[0]]

    return {
        "need_followup": True,
        "question": followup_q,
        "evidences": followup_evidences,
    }



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
  return [m for m in result.get("mismatches", []) if is_conflict(m, context, answer)]


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