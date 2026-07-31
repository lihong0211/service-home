"""
Gmail 智能助手 - 需要 Google OAuth 2.0 配置
环境变量：
  GOOGLE_CLIENT_ID     - Google Cloud Console 的 OAuth 2.0 客户端 ID
  GOOGLE_CLIENT_SECRET - Google Cloud Console 的 OAuth 2.0 客户端密钥
  GOOGLE_REDIRECT_URI  - OAuth 回调地址（默认 http://localhost:3001/ai/gmail/callback）

配置步骤：
1. 前往 https://console.cloud.google.com/apis/credentials
2. 创建 OAuth 2.0 客户端 ID（类型：Web 应用）
3. 添加授权回调 URI：http://localhost:3001/ai/gmail/callback
4. 启用 Gmail API：https://console.cloud.google.com/apis/library/gmail.googleapis.com
5. 将 client_id 和 client_secret 写入 .env 文件
"""
import os
from fastapi import Request
from fastapi.responses import RedirectResponse

from service.ai._dashscope_common import call_dashscope_text

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:3001/ai/gmail/callback")
GMAIL_SCOPES = "https://www.googleapis.com/auth/gmail.readonly"

_token_store: dict = {}


async def gmail_auth_api(request: Request):
    if not GOOGLE_CLIENT_ID:
        return {"code": 500, "msg": "未配置 GOOGLE_CLIENT_ID，请在 .env 文件中设置"}

    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_config(
        {"web": {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
                 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                 "token_uri": "https://oauth2.googleapis.com/token"}},
        scopes=[GMAIL_SCOPES],
        redirect_uri=GOOGLE_REDIRECT_URI,
    )
    auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true")
    return {"code": 0, "msg": "success", "data": {"auth_url": auth_url}}


async def gmail_callback_api(request: Request):
    code = request.query_params.get("code", "")
    if not code:
        return {"code": 400, "msg": "缺少授权码"}

    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_config(
        {"web": {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
                 "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                 "token_uri": "https://oauth2.googleapis.com/token"}},
        scopes=[GMAIL_SCOPES],
        redirect_uri=GOOGLE_REDIRECT_URI,
    )
    flow.fetch_token(code=code)
    credentials = flow.credentials
    session_id = "gmail_session"
    _token_store[session_id] = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
    }
    return {"code": 0, "msg": "success", "data": {"session_id": session_id}}


def _get_gmail_service(session_id: str):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    tokens = _token_store.get(session_id, {})
    if not tokens:
        raise ValueError("未登录，请先完成 Google OAuth 授权")
    creds = Credentials(
        token=tokens["token"], refresh_token=tokens.get("refresh_token"),
        client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
    )
    return build("gmail", "v1", credentials=creds)


async def gmail_list_api(request: Request):
    session_id = request.query_params.get("session_id", "gmail_session")
    try:
        service = _get_gmail_service(session_id)
        results = service.users().messages().list(userId="me", maxResults=20, labelIds=["INBOX"]).execute()
        messages = results.get("messages", [])
        emails = []
        for msg in messages[:10]:
            detail = service.users().messages().get(userId="me", id=msg["id"], format="metadata",
                                                    metadataHeaders=["Subject", "From", "Date"]).execute()
            headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
            emails.append({"id": msg["id"], "subject": headers.get("Subject", ""), "from": headers.get("From", ""), "date": headers.get("Date", ""), "snippet": detail.get("snippet", "")})
        return {"code": 0, "msg": "success", "data": {"emails": emails}}
    except Exception as e:
        return {"code": 500, "msg": str(e)}


async def gmail_summarize_api(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "gmail_session")
    message_id = body.get("message_id", "")
    try:
        service = _get_gmail_service(session_id)
        detail = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        import base64
        payload = detail.get("payload", {})
        body_data = payload.get("body", {}).get("data", "")
        if not body_data and payload.get("parts"):
            body_data = payload["parts"][0].get("body", {}).get("data", "")
        text = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="ignore") if body_data else detail.get("snippet", "")

        summary = call_dashscope_text(f"请用3-5句话总结以下邮件内容：\n{text[:2000]}")
        return {"code": 0, "msg": "success", "data": {"summary": summary, "full_text": text[:500]}}
    except Exception as e:
        return {"code": 500, "msg": str(e)}


async def gmail_reply_draft_api(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "gmail_session")
    message_id = body.get("message_id", "")
    intent = body.get("intent", "友好回复")
    try:
        service = _get_gmail_service(session_id)
        detail = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        snippet = detail.get("snippet", "")

        draft = call_dashscope_text(f"根据以下邮件内容，写一封{intent}的回复邮件（200字以内）：\n{snippet}")
        return {"code": 0, "msg": "success", "data": {"draft": draft}}
    except Exception as e:
        return {"code": 500, "msg": str(e)}
