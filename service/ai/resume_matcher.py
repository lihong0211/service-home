#!/usr/bin/env python3
"""简历与职位匹配：解析简历 PDF，与 JD 对比，输出结构化分析结果。"""

from __future__ import annotations

import json

from fastapi import Request

from config.ai import DEFAULT_CHAT_MODEL
from service.ai._dashscope_common import get_dashscope_client

_MODEL = DEFAULT_CHAT_MODEL
_client = get_dashscope_client(timeout=120.0)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import PyPDF2
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        pass
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    except Exception as e:
        raise ValueError(f"PDF 解析失败: {e}")


async def resume_match_api(request: Request):
    form = await request.form()
    pdf_file = form.get("resume_pdf")
    job_description = (form.get("job_description") or "").strip()

    if not pdf_file or not job_description:
        return {"code": 400, "msg": "Missing resume_pdf or job_description"}

    pdf_bytes = await pdf_file.read()
    try:
        resume_text = _extract_pdf_text(pdf_bytes)
    except ValueError as e:
        return {"code": 400, "msg": str(e)}

    if not resume_text:
        return {"code": 400, "msg": "无法从 PDF 中提取文字，请确认为文字型 PDF"}

    prompt = f"""请分析以下简历与职位描述的匹配程度，以 JSON 格式返回分析结果。

简历内容：
{resume_text[:3000]}

职位描述：
{job_description[:2000]}

请以如下 JSON 格式返回（不要有其他内容）：
{{
  "score": <0-100 的整数匹配分>,
  "strengths": ["优势1", "优势2", ...],
  "gaps": ["差距1", "差距2", ...],
  "suggestions": ["建议1", "建议2", ...],
  "resume_summary": "简历摘要（100字以内）"
}}"""

    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content or ""
        # 提取 JSON 部分
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
        else:
            raise ValueError("LLM 返回格式错误")
    except json.JSONDecodeError:
        return {"code": 500, "msg": "LLM 返回格式解析失败，请重试"}
    except Exception as e:
        return {"code": 500, "msg": f"分析失败: {e}"}

    return {"code": 0, "msg": "success", "data": result}
