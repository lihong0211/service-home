"""
生成测试用的简历 PDF。运行：python test-fixtures/generate_resume_pdf.py
"""
try:
    from fpdf import FPDF, XPos, YPos
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF, XPos, YPos

import os

out = os.path.join(os.path.dirname(__file__), "sample_resume.pdf")

pdf = FPDF()
pdf.set_margins(20, 20, 20)
pdf.add_page()

W = 170  # usable width

def ln_reset():
    """Ensure cursor is at left margin (line() leaves cursor at right edge)."""
    pdf.set_x(pdf.l_margin)

def section(title):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    ln_reset()
    pdf.cell(W, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(160, 160, 160)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.l_margin + W, y)
    ln_reset()
    pdf.ln(2)

def para(text):
    ln_reset()
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(W, 6, text)

def bullet(text):
    ln_reset()
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(W, 6, f"  - {text}")

def job(title, company, period):
    ln_reset()
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(W, 7, f"{title}  @  {company}  ({period})",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def kv(key, value):
    ln_reset()
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(38, 6, f"{key}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(W - 38, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# ── Header ──────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 18)
ln_reset()
pdf.cell(W, 10, "Zhang Wei", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(70, 70, 70)
ln_reset()
pdf.cell(W, 6, "Senior Software Engineer", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
ln_reset()
pdf.cell(W, 6, "zhangwei@example.com  |  +86-138-0000-1234  |  github.com/zhangwei",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)

# ── Summary ──────────────────────────────────────────────
section("Professional Summary")
para(
    "Software engineer with 6+ years of expertise in Python, FastAPI, and distributed systems. "
    "Proficient in building AI-powered backend services and cloud deployments. "
    "Passionate about LLM applications and RAG system design."
)

# ── Experience ───────────────────────────────────────────
section("Work Experience")

job("Senior Backend Engineer", "ByteDance", "2022.03 - Present")
bullet("Led AI recommendation engine serving 50M+ daily users")
bullet("Built LLM-based content moderation pipeline using Python and FastAPI")
bullet("Reduced API latency by 40% through Redis caching and async optimization")
pdf.ln(2)

job("Backend Engineer", "Alibaba Cloud", "2019.07 - 2022.02")
bullet("Developed microservices for cloud storage platform handling 1TB+ daily uploads")
bullet("Built distributed task queue system using Celery and RabbitMQ")
bullet("Implemented vector search using FAISS for similarity matching")

# ── Skills ───────────────────────────────────────────────
section("Technical Skills")
kv("Languages",    "Python, TypeScript, Go, SQL")
kv("Frameworks",   "FastAPI, Django, React, LangChain, LangGraph")
kv("AI/ML",        "OpenAI API, DashScope, Ollama, Qdrant, FAISS, RAG systems")
kv("Infra",        "Docker, Kubernetes, AWS, Aliyun, Redis, PostgreSQL, MySQL")

# ── Education ────────────────────────────────────────────
section("Education")
ln_reset()
pdf.set_font("Helvetica", "B", 10)
pdf.cell(W, 7, "B.Eng. Computer Science  -  Tsinghua University  (2015-2019)",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
ln_reset()
pdf.set_font("Helvetica", "", 10)
pdf.cell(W, 6, "GPA: 3.8/4.0  |  Algorithms, OS, Databases, Machine Learning",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.output(out)
print(f"Generated: {out}")
