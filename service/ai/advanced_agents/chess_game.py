import json
import uuid
from fastapi import Request
from dashscope import Generation

# In-memory game store
_games: dict = {}


async def chess_new_api(request: Request):
    try:
        import chess
        import chess.svg
    except ImportError:
        return {"code": 500, "msg": "请安装 python-chess: pip install python-chess"}

    game_id = str(uuid.uuid4())[:8]
    board = chess.Board()
    _games[game_id] = board

    svg = chess.svg.board(board, size=400)
    return {
        "code": 0, "msg": "success",
        "data": {
            "game_id": game_id,
            "fen": board.fen(),
            "board_svg": svg,
            "turn": "white",
        },
    }


async def chess_move_api(request: Request):
    try:
        import chess
        import chess.svg
    except ImportError:
        return {"code": 500, "msg": "请安装 python-chess: pip install python-chess"}

    body = await request.json()
    game_id = body.get("game_id", "")
    move_uci = body.get("move", "")

    board = _games.get(game_id)
    if board is None:
        return {"code": 404, "msg": "游戏不存在，请重新开始"}

    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return {"code": 400, "msg": f"非法走法：{move_uci}，请重新输入"}
        board.push(move)
    except Exception:
        return {"code": 400, "msg": f"走法格式错误：{move_uci}，请使用 UCI 格式（如 e2e4）"}

    if board.is_game_over():
        svg = chess.svg.board(board, size=400)
        result = board.result()
        _games.pop(game_id, None)
        return {
            "code": 0, "msg": "success",
            "data": {
                "fen": board.fen(), "board_svg": svg,
                "game_over": True, "result": result,
                "ai_move": None,
            },
        }

    # Ask AI for its move
    legal_moves = [m.uci() for m in board.legal_moves]
    prompt = f"""你是一个国际象棋引擎。当前棋盘 FEN：{board.fen()}
合法走法（UCI格式）：{', '.join(legal_moves[:30])}

请选择一个合法走法，只输出 UCI 格式的走法，不要任何解释。例如：e7e5"""

    ai_move_uci = None
    try:
        resp = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
        )
        raw = resp.output.choices[0].message.content.strip().split()[0]
        ai_move = chess.Move.from_uci(raw)
        if ai_move in board.legal_moves:
            board.push(ai_move)
            ai_move_uci = raw
        else:
            # fallback: pick first legal move
            ai_move = next(iter(board.legal_moves))
            board.push(ai_move)
            ai_move_uci = ai_move.uci()
    except Exception:
        if list(board.legal_moves):
            ai_move = next(iter(board.legal_moves))
            board.push(ai_move)
            ai_move_uci = ai_move.uci()

    game_over = board.is_game_over()
    svg = chess.svg.board(board, size=400)

    if game_over:
        _games.pop(game_id, None)

    return {
        "code": 0, "msg": "success",
        "data": {
            "fen": board.fen(),
            "board_svg": svg,
            "ai_move": ai_move_uci,
            "game_over": game_over,
            "result": board.result() if game_over else None,
            "turn": "white" if board.turn == chess.WHITE else "black",
        },
    }
