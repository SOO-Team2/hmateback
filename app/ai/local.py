import json
import re
import requests

LLM_URL     = 'http://127.0.0.1:11434/v1/chat/completions' #localhost보다 답변이 훨씬 빠름
LLM_MODEL   = 'qwen2.5:7b-instruct-q4_K_M'
EMBED_URL   = 'http://127.0.0.1:11434/v1/embeddings'
EMBED_MODEL = 'bge-m3'

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
지원자의 직무와 참고 원문(이력서·자기소개서 발췌)을 바탕으로 '왜 우리 회사에 지원했는지' 그 동기를 파고드는 면접 질문 1개를 생성해 주세요.

[제약 사항]
1. 면접관이 지원자에게 묻는 말투로, 참고 원문 문장을 베끼지 말고 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
2. '지원동기' 단계이므로, 오직 **과거와 현재의 선택 이유("왜 이 회사인가?", "왜 이 직무인가?")**에만 집중할 것.
3. 절대 '입사 후 포부'나 미래 계획("입사해서 어떤 기여를 할 것인지", "향후 목표")에 대한 내용을 질문에 포함하지 말 것.
4. 참고 원문이 '없음'인 경우 해당 직무에 맞는 일반적인 지원 동기 질문을 생성할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장"
}
""",
  "경험 및 역량": """
지원자의 직무와 참고 원문(이력서·자기소개서 발췌)을 바탕으로 직무 관련 프로젝트 경험과 기술 역량을 검증하는 면접 질문 1개를 생성해 주세요.

[제약 사항]
1. 면접관이 지원자에게 묻는 말투로, 참고 원문 문장을 베끼지 말고 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
2. 참고 원문에 적힌 경험 하나만 골라 본인이 한 일, 판단 근거, 결과를 물을 것. 원문에 이미 답이 적힌 내용은 다시 묻지 말 것.
3. 참고 원문이 '없음'인 경우 해당 직무에 맞는 일반적인 경험 및 역량 질문을 생성할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장"
}
""",
  "인성 및 협업": """
지원자의 직무와 참고 원문(이력서·자기소개서 발췌)을 바탕으로 협업 태도, 소통 방식, 인성을 검증하는 면접 질문 1개를 생성해 주세요.

[제약 사항]
1. 면접관이 지원자에게 묻는 말투로, 참고 원문 문장을 베끼지 말고 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
2. 팀원과의 협업 과정, 갈등 해결 경험, 또는 본인의 장단점이 드러나는 구체적인 사례를 물어볼 것.
3. 참고 원문이 '없음'인 경우 해당 직무에 맞는 일반적인 인성 및 협업 질문을 생성할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장"
}
""",
  "입사 후 포부": """
지원자의 직무와 참고 원문(이력서·자기소개서 발췌)을 바탕으로 입사 후 포부와 성장 계획을 확인하는 면접 질문 1개를 생성해 주세요.

[제약 사항]
1. 면접관이 지원자에게 묻는 말투로, 참고 원문 문장을 베끼지 말고 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
2. 입사 후 우리 회사에서 어떻게 기여할 것인지, 단기·장기적인 목표와 실천 계획을 묻되, 질문은 60자 이내로 쓰고 참고 원문을 길게 옮기지 말 것.
  - 좋은 예: 입사 후 우리 회사에서 이루고 싶은 단기 목표와 실천 계획은 무엇인가요?
3. 참고 원문이 '없음'인 경우 해당 직무에 맞는 일반적인 입사 후 포부 질문을 생성할 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "question": "면접 질문 문장"
}
"""
}

JOB_UNIT_RULE = """
[직무 역량 사용 규칙]
1. [직무 역량]의 업무 하나만 골라, 수행해 본 경험이 있는지와 그때 맡은 역할을 물을 것.
2. 지원자가 그 업무를 경험했다고 단정하지 말 것.
  - 나쁜 예: 인사평가 경험을 가진 지원자님은 어떻게 생각하시나요?
  - 좋은 예: 인사평가 업무를 수행해 본 경험이 있다면, 그때 맡은 역할을 말씀해 주시겠어요?
3. 지원자를 평가하거나 칭찬하지 말고, 앞으로의 역할이나 계획을 묻지 말 것.
4. [직무 역량]에 없는 기술명, 제품명, 법규, 수치는 쓰지 말 것.
"""

# 5. 질문
def ask_question(category: str, job: str, evidences: list[str], job_units: list[str] | None = None) -> dict:
  prompt = PROMPT_CATEGORIES[category] + "\n질문은 반드시 한국어로만 쓰고, 중국어와 한자는 쓰지 말 것.\n"
  context = "\n\n".join(evidences) if evidences else "없음"

  user_content = f"""
[지원 직무]
{job}

[참고 원문]
{context}
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


VERIFY_PROMPT = """
지원자의 면접 답변과 서류 원문을 비교해, 같은 사실의 내용이 서로 다른 부분만 찾아 주세요.

[판단 규칙]
1. 숫자, 기간, 인원, 역할, 사용 기술처럼 확인할 수 있는 사실이 서로 다를 때만 불일치로 판단할 것.
2. 서류에 없는 내용을 답변에서 새로 말하거나, 서류 내용을 답변에서 말하지 않은 것은 불일치가 아님.
3. 표현이나 말투가 달라도 뜻이 같으면 불일치가 아님.
4. 불일치가 없으면 빈 배열([])로 출력할 것.
5. document와 answer에는 문장을 그대로 옮기지 말고, 서로 다른 사실만 15자 이내의 짧은 구절로 쓸 것.
  - 예: 서류에 "프로젝트를 3개월 진행했습니다", 답변에 "프로젝트를 6개월 진행했습니다"라면 document는 "프로젝트 기간 3개월", answer는 "프로젝트 기간 6개월"

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "mismatches": [
    {"document": "서류의 다른 사실", "answer": "답변의 다른 사실"}
  ]
}
"""


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
  return [m for m in result.get("mismatches", []) if is_conflict(m)]


CONFLICT_PROMPT = """
서류 내용과 면접 답변이 같은 대상에 대한 사실로서 서로 모순되는지 판단해 주세요.

[판단 규칙]
1. 두 내용이 동시에 사실일 수 없을 때만 true로 판단할 것.
2. 서로 다른 이야기를 하고 있거나, 답변이 내용을 덧붙이는 것뿐이면 false로 판단할 것.
3. 표현만 다르고 뜻이 같으면 false로 판단할 것.

[예시]
- 서류: 프로젝트 기간 3개월 / 답변: 프로젝트 기간 6개월 -> true
- 서류: 기획 담당 / 답변: 디자인 담당 -> true
- 서류: 고객과 갈등이 있었음 / 답변: 회의를 제안함 -> false
- 서류: 3개월 진행 / 답변: 석 달 동안 진행 -> false

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{"conflict": true 또는 false}
"""


# 2차 확인: 두 내용이 동시에 사실일 수 없을 때만 불일치로 인정
def is_conflict(mismatch: dict) -> bool:
  document = str(mismatch.get("document", "")).strip()
  answer = str(mismatch.get("answer", "")).strip()
  if not document or not answer:
    return False

  result = call_llm(CONFLICT_PROMPT, f"\n서류: {document}\n답변: {answer}\n")
  return result.get("conflict") is True
