from pathlib import Path
import pandas as pd
from app.database import Session
from app.models import JobUnit
from app.ai.local import embed


BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "한국산업인력공단_직무NCS분류별정보.csv"

df = pd.read_csv(path, encoding="cp949", dtype=str)

latest = df["생성일자"].max()
df = df[df["생성일자"] == latest].copy()

cols = [
    "NCS대분류코드명",
    "NCS중분류코드명",
    "NCS소분류코드명",
    "NCS세분류코드명",
    "능력단위명",
    "능력단위정의",
    "NCS분류코드",
]
df = df[cols].copy()

for col in cols:
    df[col] = (
        df[col]
        .str.replace("ㆍ", "·", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

# 분류가 N/A인 행 제외
is_na = df[cols[:4]].isna().any(axis=1)
df = df[~is_na].copy()

# 삭제 표시가 있거나 정의가 없는 행 제외
is_deleted = df["능력단위명"].str.contains("삭제", regex=False, na=False)
is_short = df["능력단위정의"].fillna("").str.len() < 15
df = df[~(is_deleted | is_short)].copy()

# 구버전 행 제외
is_old = df["능력단위명"].str.contains("구버전", regex=False, na=False)
df = df[~is_old].copy()

# 분류코드 끝의 _16v4에서 16과 4를 숫자로 꺼내기
version = df["NCS분류코드"].str.extract(r"_(\d+)v(\d+)$").astype(int)
df["year"] = version[0]
df["version"] = version[1]

# 비교할 때만 공백을 뺀 값 사용
df["detail_key"] = df["NCS세분류코드명"].str.replace(" ", "", regex=False)
df["name_key"] = df["능력단위명"].str.replace(" ", "", regex=False)

# 최신 버전이 위로 오게 정렬한 뒤, 같은 능력단위 중 첫 행만 남기기
df = df.sort_values(["year", "version"], ascending=False, kind="stable")
df = df.drop_duplicates(subset=["detail_key", "name_key"], keep="first")

# 연결표에 있는 직무의 능력단위만 추리기
job_map = pd.read_csv(BASE_DIR / "job_ncs_map.csv", encoding="utf-8-sig", dtype=str)
job_units = job_map.merge(df[cols], left_on="ncs_detail", right_on="NCS세분류코드명")

# 임베딩할 글: 능력단위명 + 정의
texts = (job_units["능력단위명"] + ": " + job_units["능력단위정의"]).tolist()
unique_texts = list(dict.fromkeys(texts))

# 같은 글은 한 번만 임베딩
vectors = {}
for i in range(0, len(unique_texts), 50):
    batch = [{"content": text} for text in unique_texts[i:i + 50]]
    for chunk in embed(batch):
        vectors[chunk["content"]] = chunk["embedding"]

# 기존 데이터를 지우고 새로 저장
db = Session()
try:
    db.query(JobUnit).delete()
    for row, text in zip(job_units.to_dict("records"), texts):
        db.add(JobUnit(
            job=row["job"],
            ncs_detail=row["ncs_detail"],
            unit_name=row["능력단위명"],
            definition=row["능력단위정의"],
            ncs_code=row["NCS분류코드"],
            embedding=vectors[text],
        ))
    db.commit()
finally:
    db.close()
