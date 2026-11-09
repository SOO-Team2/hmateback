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

#  꼬리 질문 생성 프롬프트
FOLLOWUP_PROMPTS = {
  "지원 동기": """
지원자의 직무, 이전 면접 질문, 지원자의 답변을 바탕으로 꼬리 질문 생성 여부를 엄격히 판단하고, 추가 검증이 필요할 때만 꼬리 질문 1개를 생성해 주세요.

[제약 사항]
1. 질문 생성 판단 기준:
  - 지원자가 회사/직무 선택 계기와 구체적인 이유를 답변에서 충분히 설명했다면, 추가 질문을 만들지 말고 need_followup을 false, question은 null로 출력할 것.
  - 답변이 한두 줄로 너무 짧거나, "성장 가능성이 높아서", "열심히 배우고 싶어서"처럼 모호한 상투적 표현에 그칠 때만 need_followup을 true로 설정할 것.
2. 면접관이 지원자에게 묻는 말투로, 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
  - 지원자가 답변에서 언급한 특정 단어나 문맥을 짚어, 왜 그렇게 생각했는지 구체적인 계기를 물을 것.

3. [절대 금지 사항]:
  - 이전 질문과 동일하거나 유사한 질문을 다시 던지는 행위 절대 금지.
  - 지원자가 답변에서 이미 말한 내용을 되묻지 말 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "need_followup": true 또는 false,
  "question": "꼬리 질문 문장" 또는 null
}
""",
  "경험 및 역량": """
지원자의 직무, 이전 면접 질문, 지원자의 답변을 바탕으로 직무 관련 프로젝트 경험과 기술 역량을 '깊이 파고드는(Deep-Dive)' 꼬리 질문 1개를 생성해 주세요.

[제약 사항]
1. 이 단계는 필수 검증 단계이므로 반드시 need_followup은 true로 설정할 것.
2. 면접관이 지원자에게 묻는 말투로, 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
3. [꼬리 질문 작성 규칙]:
  - 반드시 지원자의 [답변]에 실제로 등장한 '구체적인 프로젝트명, 기술 스택, 시도한 해결책' 중 하나를 직접 인용할 것.
  - 답변에서 미처 다 설명하지 않은 '기술적 판단 근거', '예상치 못한 문제 상황과 대처', 혹은 '결과의 한계와 개선점'을 날카롭게 파고들 것.
  - 예시: "답변에서 언급하신 OO 방식을 적용하셨을 때, 기존 방식 대비 발생했던 성능 저하나 부작용은 없었나요? 어떻게 해결하셨나요?"
  - 해당 방식을 적용하여 얻은 실제 성과나 정량적 결과
4. [절대 금지 사항]:
  - 이전 질문과 동일하거나 유사한 질문을 다시 던지는 행위 절대 금지.
  - 지원자가 답변에서 이미 말한 내용을 되묻지 말 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "need_followup": true,
  "question": "꼬리 질문 문장"
}
""",
  "인성 및 협업": """
지원자의 직무, 이전 면접 질문, 지원자의 답변을 바탕으로 소통 방식과 협업 태도를 '깊이 파고드는(Deep-Dive)' 꼬리 질문 1개를 생성해 주세요.

[제약 사항]
1. 이 단계는 필수 검증 단계이므로 반드시 need_followup은 true로 설정할 것.
2. 면접관이 지원자에게 묻는 말투로, 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
3. [꼬리 질문 작성 규칙]:
  - 반드시 지원자의 [답변]에 실제로 등장한 '구체적인 갈등 대상(예: 학술팀 vs 체험팀), 제안했던 해결책, 특정 상황'을 직접 인용할 것.
  - 답변에 나오지 않은 '당시 반대 의견을 가졌던 동료의 반응과 설득 과정', 혹은 '해결책 실행 중 겪었던 돌발 변수와 조율 과정'을 구체적으로 파고들 것.
  - 예시: "답변에서 말씀하신 융합형 전시 모델을 제안하셨을 때, 학술적 깊이를 고집하던 팀원들을 최종적으로 동의하게 만든 결정적인 설득 방식이나 대화는 무엇이었나요?"
  - 그 경험을 통해 소통과 협업에 대해 무엇을 배웠는지
4. [절대 금지 사항]:
  - 이전 질문과 동일하거나 유사한 질문을 다시 던지는 행위 절대 금지.
  - 지원자가 답변에서 이미 말한 내용을 되묻지 말 것.
[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "need_followup": true,
  "question": "꼬리 질문 문장"
}
""",
  "입사 후 포부": """
지원자의 직무, 이전 면접 질문, 지원자의 답변을 바탕으로 꼬리 질문 생성 여부를 엄격히 판단하고, 필요할 때만 꼬리 질문 1개를 생성해 주세요.

[제약 사항]
1. 질문 생성 판단 기준:
  - 지원자가 입사 후 목표와 구체적인 실천 계획을 답변에서 이미 조리 있게 설명했다면, 절대로 추가 질문을 만들지 말고 need_followup을 false, question은 null로 출력할 것.
  - 답변이 "최선을 다하겠습니다", "성실히 일하겠습니다"처럼 단순한 다짐에 불과하고 구체적 계획이 없을 때만 need_followup을 true로 설정할 것.
2. 면접관이 지원자에게 묻는 말투로, 1~2문장으로 짧게 새로 쓸 것.
  - 회사는 '우리 회사', 지원자는 '지원자님'으로 부를 것.
  - '귀사', '저', '제'는 지원자가 쓰는 표현이므로 질문에 쓰지 말 것.
  - 나쁜 예: 귀사의 직무가 왜 제 경험과 강점을 살릴 수 있는 자리인가요?
  - 좋은 예: 지원자님의 경험과 강점이 우리 회사의 이 직무에 어떻게 도움이 된다고 생각하시나요?
3. 목표를 달성하기 위해 입사 후 1년 이내에 가장 먼저 실천할 구체적인 행동 계획을 물어볼 것.
4. [절대 금지 사항]:
  - 이전 질문과 동일하거나 유사한 질문을 다시 던지는 행위 절대 금지.
  - 지원자가 답변에서 이미 말한 내용을 되묻지 말 것.

[출력 형식]
다른 설명 없이 반드시 아래 JSON 객체 형식으로만 출력해 주세요.
{
  "need_followup": true 또는 false,
  "question": "꼬리 질문 문장" 또는 null
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

# ---------------------------------------------------------------------------
# 꼬리 질문 생성 함수 (규칙 강제 및 결과 정제)
# 1. 지원동기 / 입사 후 포부: 답변이 충분하면 need_followup=False, 부족할 때만 질문 생성
# 2. 경험 및 역량 / 인성 및 협업: 무조건 need_followup=True 필수 1개 보장
# ---------------------------------------------------------------------------
def ask_followup_question(
    category: str,
    job: str,
    question: str,
    answer: str,
    evidences: list[str] | None = None,
) -> dict:
    """
    지원자의 직전 질문과 답변을 분석하여 꼬리 질문을 생성하거나 건너뜁니다.
    반환 형식: {"need_followup": bool, "question": str | None}
    """
    # 1. 해당 카테고리에 맞는 꼬리질문 프롬프트 준비
    prompt = FOLLOWUP_PROMPTS.get(category, "")
    prompt += "\n질문은 반드시 한국어로만 쓰고, 중국어와 한자는 쓰지 말 것.\n"

    # 2. LLM에 전달할 사용자 컨텍스트 구성
    user_content = f"""
[지원 직무]
{job}

[이전 면접 질문]
{question}

[지원자의 답변]
{answer}
"""
    if evidences:
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
                followup_q = "답변해 주신 경험을 진행하시면서, 사전에 예상하지 못했던 가장 큰 기술적 변수나 한계점은 무엇이었나요?"
            else:
                followup_q = "그 조율 과정에서 지원자님의 중재안에 대해 끝까지 반대하거나 우려를 표했던 팀원은 없었나요? 그 팀원은 어떻게 설득하셨나요?"

    # 6. 지원동기 / 포부에서 꼬리질문이 필요 없다고 판단한 경우 question을 null로 정리
    if not need_followup:
        return {"need_followup": False, "question": None}

    # 7. 말투 필터링 ('귀사' -> '우리 회사', '생각하신가요' -> '생각하시나요')
    if followup_q:
        followup_q = re.sub(r"귀\s*사", "우리 회사", str(followup_q))
        followup_q = followup_q.replace("생각하신가요", "생각하시나요").strip()

    return {
        "need_followup": True,
        "question": followup_q,
    }



VERIFY_PROMPT = """
지원자의 면접 답변과 서류 원문을 비교해, 같은 사실의 내용이 서로 다른 부분만 찾아 주세요.

[판단 규칙]
1. 숫자, 기간, 인원, 역할, 사용 기술뿐만 아니라 규모·범위(대규모/소규모, 개인/팀), 결과·상태(성공/실패, 증가/감소), 대상·방식(온라인/오프라인) 등 같은 대상에 대한 사실이나 특징이 서로 다를 때 불일치로 판단할 것.
2. 서류에 없는 내용을 답변에서 새로 말하거나, 서류 내용을 답변에서 말하지 않은 것은 불일치가 아님.
3. 표현이나 말투가 달라도 뜻이 같으면 불일치가 아님.
4. 불일치가 없으면 빈 배열([])로 출력할 것.
5. document와 answer에는 문장을 그대로 옮기지 말고, 서로 다른 사실만 15자 이내의 짧은 구절로 쓸 것.
  - 예1: 서류에 "프로젝트를 3개월 진행했습니다", 답변에 "프로젝트를 6개월 진행했습니다"라면 document는 "프로젝트를 3개월 진행", answer는 "프로젝트를 6개월 진행"
  - 예2: 서류에 "대용량 처리 시스템을 구축했습니다", 답변에 "소규모 처리 시스템을 만들었습니다"라면 document는 "대용량 처리 시스템", answer는 "소규모 처리 시스템"

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
