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
2. 참고 원문에 적힌 구체적인 프로젝트, 문제 해결 과정, 본인의 역할이나 기술적 선택 이유를 파고들 것.
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

# 5. 질문
def ask_question(category: str, job: str, evidences: list[str]) -> dict:
  prompt = PROMPT_CATEGORIES[category]
  context = "\n\n".join(evidences) if evidences else "없음"

  user_content = f"""
[지원 직무]
{job}

[참고 원문]
{context}
"""
  result = call_llm(prompt, user_content)
  result["question"] = re.sub(r"귀\s*사", "우리 회사", result["question"])
  result["question"] = result["question"].replace("생각하신가요", "생각하시나요")
  return result