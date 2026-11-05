from sqlalchemy import Column, String, Integer, Boolean, Text, TIMESTAMP, LargeBinary, ForeignKey, func
from pgvector.sqlalchemy import Vector
from app.database import Base


class User(Base):
    __tablename__ = "USER"
    id = Column(String(50), primary_key=True)
    pw = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    joindate = Column(TIMESTAMP, server_default=func.current_timestamp(), nullable=False) #SQL 내장 함수
    service_agreed = Column(Boolean, nullable=False)
    privacy_agreed = Column(Boolean, nullable=False)
    save_agreed = Column(Boolean, nullable=False)


class Resume(Base):
    __tablename__ = "RESUME"
    # 1) 기본 이력서
    id = Column(String(50), ForeignKey("USER.id"), primary_key=True)
    name = Column(String(50))
    school = Column(String(100))
    major = Column(String(100))
    skill = Column(String(255))
    portfolio = Column(LargeBinary)
    # 2) PDF 이력서
    pdf_file = Column(LargeBinary)
    pdf_name = Column(String(255))


class Career(Base):
    __tablename__ = "CAREER"
    career_id = Column(Integer, primary_key=True, autoincrement=True)
    company = Column(String(100))
    role = Column(String(500))
    start_year = Column(String(7))
    end_year = Column(String(7))
    id = Column(String(50), ForeignKey("RESUME.id"))


class Certificate(Base):
    __tablename__ = "CERTIFICATE"
    cert_id = Column(Integer, primary_key=True, autoincrement=True)
    cert = Column(String(100), nullable=False)
    org = Column(String(100))
    id = Column(String(50), ForeignKey("RESUME.id"), nullable=False)


class Interview(Base):
    __tablename__ = "INTERVIEW"
    interview_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    cat_major = Column(String(50), nullable=False)
    cat_minor = Column(String(50), nullable=False)
    saved_resume = Column(LargeBinary)
    id = Column(String(50), ForeignKey("USER.id"), nullable=False)


class Dialogue(Base):
    __tablename__ = "DIALOGUE"
    dialogue_id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    interview_id = Column(Integer, ForeignKey("INTERVIEW.interview_id"), nullable=False)


class CL(Base):
    __tablename__ = "CL"
    cl_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    id = Column(String(50), ForeignKey("USER.id"), nullable=False)


class Answer(Base):
    __tablename__ = "ANSWER"
    answer_id = Column(Integer, primary_key=True, autoincrement=True)
    cl_id = Column(Integer, ForeignKey("CL.cl_id"), nullable=False)
    motivation = Column(Text)
    growth = Column(Text)
    personality = Column(Text)
    plan = Column(Text)


class PdfChunk(Base):
    __tablename__ = "PDF_CHUNK"
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    doctype = Column(String(50), nullable=False)
    filename = Column(String(255))
    index = Column(Integer, nullable=False) #청크 순서
    content = Column(Text, nullable=False) #원문 텍스트
    embedding = Column(Vector(1024))

class JobUnit(Base):
    __tablename__ = "JOB_UNIT"
    unit_id = Column(Integer, primary_key=True, autoincrement=True)
    job = Column(String(50), nullable=False, index=True) #프론트 소분류 이름
    ncs_detail = Column(String(100), nullable=False) #NCS 세분류
    unit_name = Column(String(200), nullable=False) #능력단위명
    definition = Column(Text, nullable=False) #능력단위 정의
    ncs_code = Column(String(30)) #NCS 분류코드
    embedding = Column(Vector(1024), nullable=False)