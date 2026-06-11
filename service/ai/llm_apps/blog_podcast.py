import json
import asyncio
import tempfile
import os
from fastapi import Request
from fastapi.responses import StreamingResponse, Response
from dashscope import Generation
import requests as req_lib


async def blog_script_api(request: Request):
    body = await request.json()
    url = body.get("url", "")
    if not url:
        return {"code": 400, "msg": "请输入博客 URL"}

    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
        text = article.text[:4000]
    except Exception as e:
        return {"code": 500, "msg": f"网页抓取失败: {str(e)}"}

    if not text:
        return {"code": 400, "msg": "无法获取文章内容"}

    prompt = f"""请将以下博客文章改写成一段播客脚本（800-1200字）：

文章标题：{title}
文章内容：
{text}

要求：
1. 用口语化、自然的语气
2. 开头吸引听众注意
3. 逻辑清晰，有节奏感
4. 结尾有总结和行动呼吁
5. 适合 TTS 朗读（不要 Markdown 标记符）

直接输出播客脚本，不要任何前言。"""

    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
    )
    script = resp.output.choices[0].message.content.strip()

    return {
        "code": 0, "msg": "success",
        "data": {"title": title, "original_summary": text[:300] + "...", "podcast_script": script},
    }


async def blog_to_podcast_api(request: Request):
    body = await request.json()
    url = body.get("url", "")
    voice = body.get("voice", "zh-CN-XiaoxiaoNeural")

    if not url:
        return {"code": 400, "msg": "请输入博客 URL"}

    # Get script
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
        text = article.text[:4000]
    except Exception as e:
        return {"code": 500, "msg": f"网页抓取失败: {str(e)}"}

    prompt = f"""将以下博客改写成播客脚本（600字以内，适合TTS朗读，无Markdown）：\n{text[:2000]}"""
    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
    )
    script = resp.output.choices[0].message.content.strip()[:600]

    # TTS
    import edge_tts
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(script, voice=voice)
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'attachment; filename="podcast.mp3"'},
    )
