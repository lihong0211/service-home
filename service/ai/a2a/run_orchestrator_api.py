"""
编排器 HTTP 接口：供前端直接调用，返回 chain + artifacts + final_artifact。
先启动三个 Agent（8001/8002/8003），再启动本服务（如 8010）。
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from service.ai.a2a.orchestrator import get_result_for_frontend

app = FastAPI(title="内容生成链编排器")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RunChainRequest(BaseModel):
    topic: str


@app.post("/ai/a2a/chain")
def run_chain_api(body: RunChainRequest):
    """
    执行 大纲→正文→摘要 链，返回结构：
    - chain: 调用链（每步 agent、状态、时间等），供前端展示流水线
    - artifacts: 每步产出（outline/document/summary），供前端展示每步输出
    - final_artifact: 最后一步产出，供前端主区域展示
    （与主程序路由 POST /ai/a2a/chain 一致，可单独跑本服务时使用）
    """
    try:
        return get_result_for_frontend(body.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
