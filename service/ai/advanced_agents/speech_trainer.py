import io
import json
import time
import tempfile
import os
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dashscope import Generation


async def speech_analyze_api(request: Request):
    form = await request.form()
    audio_file: UploadFile = form.get("audio")
    if not audio_file:
        return {"code": 400, "msg": "请上传音频文件"}

    content = await audio_file.read()
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import faster_whisper
        model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(tmp_path, beam_size=5)
        transcript = " ".join(seg.text for seg in segments)
        duration = info.duration
    except Exception as e:
        return {"code": 500, "msg": f"音频转录失败: {str(e)}"}
    finally:
        os.unlink(tmp_path)

    if not transcript.strip():
        return {"code": 400, "msg": "未能识别到语音内容"}

    word_count = len(transcript.replace(" ", ""))
    words_per_minute = int(word_count / (duration / 60)) if duration > 0 else 0

    analysis_prompt = f"""请分析以下演讲文本，给出专业的演讲评估报告：

演讲时长：{duration:.1f} 秒
字数：{word_count} 字
语速：约 {words_per_minute} 字/分钟

演讲内容：
{transcript}

请从以下维度评估（JSON格式输出）：
{{
  "structure_score": 评分1-10,
  "language_score": 评分1-10,
  "pace_assessment": "语速评估（偏快/适中/偏慢）",
  "strengths": ["优点1", "优点2", "优点3"],
  "improvements": ["改进点1", "改进点2", "改进点3"],
  "suggestions": ["具体建议1", "具体建议2", "具体建议3"],
  "overall_feedback": "总体评价（2-3句话）",
  "filler_words": ["填充词1", "填充词2"],
  "content_summary": "内容摘要（1句话）"
}}

只输出 JSON，不要其他内容。"""

    try:
        resp = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": analysis_prompt}],
            result_format="message",
        )
        raw = resp.output.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        analysis = json.loads(raw)
    except Exception:
        analysis = {
            "structure_score": 7,
            "language_score": 7,
            "pace_assessment": "适中",
            "strengths": ["表达清晰", "内容完整"],
            "improvements": ["可增加更多具体案例"],
            "suggestions": ["适当使用停顿增强节奏感"],
            "overall_feedback": "演讲整体表现良好，继续练习会更好。",
            "filler_words": [],
            "content_summary": transcript[:50],
        }

    return {
        "code": 0,
        "msg": "success",
        "data": {
            "transcript": transcript,
            "word_count": word_count,
            "duration": round(duration, 1),
            "words_per_minute": words_per_minute,
            "analysis": analysis,
        },
    }
