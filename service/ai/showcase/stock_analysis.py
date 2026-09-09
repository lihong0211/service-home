#!/usr/bin/env python3
"""
AI 股票分析：独立功能模块，不依赖 text2sql。
数据源为 Tushare（实时拉取 A 股历史日线，落地到 ai 库 stock_price 表做缓存），
在此基础上提供 ARIMA 预测、Prophet 周期分解、布林带异常点检测、自动配图查询。
"""

import base64
import io
import os
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from fastapi import Request
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from utils.http_body import read_json_optional, query_dict
from config.db import DB_AI_CONFIG

import matplotlib
matplotlib.use("Agg")  # 无 GUI 的服务器环境，Agg 后端只渲染到内存/文件，不弹窗
import matplotlib.pyplot as plt

# matplotlib 默认字体不含中文，图表标题/坐标轴是中文会显示成方块，这里换成常见中文字体；
# 同时修正负号（默认会被中文字体渲染成方块）。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------
# 常量：表名、各分析方法所需的最少历史数据行数（不够就直接报错，不做无意义的建模）
# ------------------------------------------------------------------
_TABLE_NAME = "stock_price"
_MIN_ARIMA_ROWS = 30
_MIN_BOLL_ROWS = 21     # 布林带用 20 日均线，至少要有 20+1 行才能算出第一个有效值
_MIN_PROPHET_ROWS = 30

# ------------------------------------------------------------------
# 模块级缓存/状态。全部用 global 读写，因为 uvicorn 是多 worker 进程模型，
# 缓存只在单个进程内有效——这也是为什么下面还要做磁盘缓存兜底。
# ------------------------------------------------------------------
_engine: Engine | None = None          # SQLAlchemy engine，进程内单例，避免每次请求都重新建连接池
_table_ready = False                    # stock_price 表是否已确认存在，避免每次请求都发一次 CREATE TABLE IF NOT EXISTS

_stock_basic_cache: pd.DataFrame | None = None   # 全市场股票代码/名称表的内存缓存
_stock_basic_cache_ts = 0.0                       # 上面缓存的写入时间戳
_STOCK_BASIC_TTL = 24 * 3600                      # 缓存有效期 24 小时（股票名称/代码基本不会天天变）

_stock_basic_last_failure_ts = 0.0     # 上一次调用 Tushare stock_basic 失败（一般是限频）的时间戳
_STOCK_BASIC_RETRY_COOLDOWN = 300      # 失败后 5 分钟内不再重试真实接口，直接用磁盘缓存/兜底名单
                                        # ——否则用户在搜索框里敲几个字，就会对着一个已知会限频的接口
                                        # 反复重试，既没用又会在前端控制台刷一堆错误日志。

# 全市场股票列表落盘缓存路径：进程重启后不用重新打一次 Tushare，也方便跨 worker 共享
_STOCK_BASIC_DISK_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "stock_analysis", "stock_basic.csv"
)

# Tushare 免费额度下 stock_basic 限频很紧（实测 1 次/小时，短时间内多次触发甚至退化到 1 次/分钟），
# 内存/磁盘缓存都没命中前，先靠这份常见蓝筹股静态兜底名单让搜索可用；一旦某次 stock_basic 调用成功，
# 磁盘缓存写入后就不再依赖这份名单。名单跟前端 stock-analysis.ts 里的 POPULAR_STOCKS（"常用股票"
# 快捷按钮）保持一致，方便对照。
_FALLBACK_STOCKS = pd.DataFrame([
    ("600519.SH", "贵州茅台"), ("000858.SZ", "五粮液"), ("601211.SH", "国泰君安"), ("688981.SH", "中芯国际"),
    ("000001.SZ", "平安银行"), ("000002.SZ", "万科A"), ("600036.SH", "招商银行"), ("601318.SH", "中国平安"),
    ("600000.SH", "浦发银行"), ("601398.SH", "工商银行"), ("601988.SH", "中国银行"), ("601288.SH", "农业银行"),
    ("600030.SH", "中信证券"), ("601688.SH", "华泰证券"), ("300750.SZ", "宁德时代"), ("002594.SZ", "比亚迪"),
    ("600276.SH", "恒瑞医药"), ("600887.SH", "伊利股份"), ("000651.SZ", "格力电器"), ("000333.SZ", "美的集团"),
    ("601857.SH", "中国石油"), ("600028.SH", "中国石化"), ("601899.SH", "紫金矿业"), ("600809.SH", "山西汾酒"),
    ("000568.SZ", "泸州老窖"), ("601888.SH", "中国中免"), ("600690.SH", "海尔智家"), ("002415.SZ", "海康威视"),
    ("300059.SZ", "东方财富"), ("601668.SH", "中国建筑"),
], columns=["ts_code", "name"])


def _get_engine() -> Engine:
    """懒加载单例 SQLAlchemy engine，连到项目共用的 ai 库（config.db.DB_AI_CONFIG）。"""
    global _engine
    if _engine is None:
        # 密码可能包含 @、: 等特殊字符，必须做 URL 编码，否则会把连接串解析错
        pwd = quote_plus(DB_AI_CONFIG["password"])
        url = (
            f"mysql+pymysql://{DB_AI_CONFIG['user']}:{pwd}"
            f"@{DB_AI_CONFIG['host']}:{DB_AI_CONFIG['port']}/{DB_AI_CONFIG['database']}"
            f"?charset={DB_AI_CONFIG.get('charset', 'utf8mb4')}"
        )
        _engine = create_engine(url, pool_pre_ping=True)  # pool_pre_ping：用前先探活，避免用到已断开的连接
    return _engine


def _ensure_table() -> Engine:
    """确保 stock_price 表存在（进程内只执行一次 DDL，后续请求直接跳过）。"""
    global _table_ready
    engine = _get_engine()
    if not _table_ready:
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    stock_name VARCHAR(20) NOT NULL,
                    ts_code VARCHAR(20) NOT NULL,
                    trade_date VARCHAR(10) NOT NULL,
                    open DECIMAL(15,2),
                    high DECIMAL(15,2),
                    low DECIMAL(15,2),
                    close DECIMAL(15,2),
                    vol DECIMAL(20,2),
                    amount DECIMAL(20,2),
                    UNIQUE KEY uniq_stock_date (ts_code, trade_date)
                ) DEFAULT CHARSET=utf8mb4
            """))
        _table_ready = True
    return engine


def _pro():
    """构造一个 Tushare pro_api 客户端。每次调用都新建（Tushare SDK 本身很轻，没必要缓存实例）。"""
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("缺少 TUSHARE_TOKEN 环境变量")
    import tushare as ts  # 延迟 import：只有真正用到股票分析功能时才加载这个依赖
    return ts.pro_api(token)


def _stock_basic() -> pd.DataFrame:
    """全市场股票代码/名称。内存缓存 24 小时；同时落盘持久化，因为 Tushare 免费额度下
    stock_basic 限频很紧（1 次/小时，密集调用时会退化到 1 次/分钟），重启进程或短时间内
    触发限频时都要能从磁盘/静态名单兜底，而不是直接报错。
    失败后 5 分钟内不再重试真实接口，直接走磁盘缓存或兜底名单——避免每次搜索请求
    （比如用户输入框每敲一个字触发一次搜索）都再打一次必然失败的限流接口。
    """
    global _stock_basic_cache, _stock_basic_cache_ts, _stock_basic_last_failure_ts

    # 第一层：内存缓存命中且未过期，直接返回，零开销
    if _stock_basic_cache is not None and time.time() - _stock_basic_cache_ts <= _STOCK_BASIC_TTL:
        return _stock_basic_cache

    # 第二层：最近 5 分钟内已经失败过一次，大概率还在限频窗口里，不再浪费一次网络请求，
    # 直接退化到磁盘缓存（如果有）或者静态兜底名单
    if time.time() - _stock_basic_last_failure_ts <= _STOCK_BASIC_RETRY_COOLDOWN:
        if os.path.exists(_STOCK_BASIC_DISK_CACHE):
            return pd.read_csv(_STOCK_BASIC_DISK_CACHE, dtype=str)
        return _FALLBACK_STOCKS

    # 第三层：真正去调 Tushare。成功了就顺手把内存缓存和磁盘缓存都刷新一遍。
    try:
        df = _pro().stock_basic(exchange="", list_status="L", fields="ts_code,name")
        os.makedirs(os.path.dirname(_STOCK_BASIC_DISK_CACHE), exist_ok=True)
        df.to_csv(_STOCK_BASIC_DISK_CACHE, index=False)
        _stock_basic_cache = df
        _stock_basic_cache_ts = time.time()
        return df
    except Exception:
        # 大概率是限频报错。记下失败时间，进入上面第二层的冷却期；
        # 本次请求本身仍然要有结果返回，所以继续走磁盘缓存/兜底名单，不向上抛异常。
        _stock_basic_last_failure_ts = time.time()
        if os.path.exists(_STOCK_BASIC_DISK_CACHE):
            return pd.read_csv(_STOCK_BASIC_DISK_CACHE, dtype=str)
        return _FALLBACK_STOCKS


async def search_stock_api(request: Request):
    """GET /ai/stock-analysis/search?keyword=茅台 —— 按名称/代码模糊搜索，返回 ts_code 供后续接口使用。"""
    q = query_dict(request)
    keyword = (q.get("keyword") or "").strip()
    if not keyword:
        return {"code": 400, "msg": "缺少 keyword"}
    df = _stock_basic()
    # 名称按子串匹配（中文股票名），代码按大写子串匹配（ts_code 形如 600519.SH，用户可能只输入数字部分）
    mask = df["name"].str.contains(keyword, na=False) | df["ts_code"].str.contains(keyword.upper(), na=False)
    hits = df[mask].head(20)  # 最多返回 20 条，避免关键字太短（比如输入"银行"）命中一长串
    return {
        "code": 0, "msg": "success",
        "data": {"list": [{"ts_code": r.ts_code, "name": r.name} for r in hits.itertuples()]},
    }


def _default_range(days: int = 365) -> tuple[str, str]:
    """默认取「今天往前 N 天」到「今天」，各分析接口在前端没传日期区间时用这个兜底。"""
    today = datetime.now().date()
    return (today - timedelta(days=days)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _stock_name(ts_code: str) -> str:
    """根据 ts_code 查中文股票名称，仅用于写入 stock_price.stock_name 这一展示性字段。
    非致命：查不到（比如 stock_basic 也限频失败且没有任何缓存/兜底命中该代码）就直接
    用 ts_code 本身占位，不能因为一个展示字段查不到就把整个价格同步流程搞挂。
    """
    try:
        df = _stock_basic()
    except Exception:
        return ts_code
    row = df[df["ts_code"] == ts_code]
    return row.iloc[0]["name"] if len(row) else ts_code


def _sync_prices(ts_code: str, start_date: str, end_date: str) -> None:
    """确保 stock_price 表内 [start_date, end_date] 区间有数据，缺失才向 Tushare 拉取补齐。

    先查表里这只股票在该区间的最早/最晚交易日；如果已覆盖到接近 end_date（留 3 天容差，
    因为最近几天可能还没有交易数据/非交易日），就认为数据已经是新的，直接跳过，不重复拉取
    ——这是整个模块避免把 Tushare daily 接口打爆的关键机制（stock_basic 限频严重，daily
    接口额度虽然宽松得多，但同样没必要每次分析请求都重新拉一遍历史）。
    """
    engine = _ensure_table()
    with engine.connect() as conn:
        existing = conn.execute(
            text(f"SELECT COUNT(*) c, MIN(trade_date) mn, MAX(trade_date) mx FROM {_TABLE_NAME} "
                 f"WHERE ts_code = :ts_code AND trade_date BETWEEN :start AND :end"),
            {"ts_code": ts_code, "start": start_date, "end": end_date},
        ).mappings().first()
    covers = bool(existing and existing["c"] and existing["mn"] and existing["mn"] <= start_date and existing["mx"] >= (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=3)
    ).strftime("%Y-%m-%d"))
    if covers:
        return  # 已有数据覆盖请求区间，不用再拉

    # 向 Tushare 拉取该区间的日线行情。注意 Tushare daily 接口要求日期格式是 YYYYMMDD（无短横线），
    # 跟我们表里存的 YYYY-MM-DD 不一样，需要转换。
    df = _pro().daily(
        ts_code=ts_code,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if df is None or df.empty:
        return  # 可能是新股/停牌等原因该区间确实没有数据，不报错，让上层按"数据不足"处理
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    df["stock_name"] = _stock_name(ts_code)

    # upsert：同一支股票同一天的记录已存在就更新，不存在就插入。用 ON DUPLICATE KEY UPDATE
    # 配合表上的 UNIQUE KEY(ts_code, trade_date) 实现，一次 executemany 搞定整批数据，
    # 不用先查再判断插入/更新。
    rows = df[["stock_name", "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]].to_dict("records")
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {_TABLE_NAME} (stock_name, ts_code, trade_date, open, high, low, close, vol, amount)
            VALUES (:stock_name, :ts_code, :trade_date, :open, :high, :low, :close, :vol, :amount)
            ON DUPLICATE KEY UPDATE
                stock_name=VALUES(stock_name), open=VALUES(open), high=VALUES(high),
                low=VALUES(low), close=VALUES(close), vol=VALUES(vol), amount=VALUES(amount)
        """), rows)


def _query_prices(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从本地 stock_price 表读取某只股票在指定区间内的行情，按交易日升序排列。
    调用前应该先跑一次 _sync_prices 确保数据已经拉取到本地。
    """
    engine = _ensure_table()
    sql = text(f"""
        SELECT trade_date, open, high, low, close, vol, amount FROM {_TABLE_NAME}
        WHERE ts_code = :ts_code AND trade_date BETWEEN :start AND :end
        ORDER BY trade_date ASC
    """)
    return pd.read_sql(sql, engine, params={"ts_code": ts_code, "start": start_date, "end": end_date})


def _fig_to_base64() -> str:
    """把当前 matplotlib 画布保存成 PNG 并转成 base64 字符串，直接嵌进 JSON 响应里返回给前端
    （沿用项目里 data_viz.py 的既有约定：图片走 base64 内嵌，不是落盘再返回一个静态 URL）。
    调用后会关闭所有画布（plt.close("all")），避免 matplotlib 全局状态在多次请求之间串号。
    """
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close("all")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _parse_range(body: dict) -> tuple[str, str]:
    """从请求体里取 start_date/end_date，留空则用默认区间（近一年到今天）。"""
    default_start, default_end = _default_range()
    start_date = (body.get("start_date") or default_start).strip()
    end_date = (body.get("end_date") or default_end).strip()
    return start_date, end_date


async def query_prices_api(request: Request):
    """POST /ai/stock-analysis/query {ts_code, start_date?, end_date?}
    自动配图：查询历史行情并智能选图——数据点超过 20 个时抽样 10 个点画折线图（避免柱状图
    太密看不清），否则直接画柱状图。
    """
    body = await read_json_optional(request) or {}
    ts_code = (body.get("ts_code") or "").strip()
    if not ts_code:
        return {"code": 400, "msg": "缺少 ts_code"}
    start_date, end_date = _parse_range(body)

    _sync_prices(ts_code, start_date, end_date)   # 先确保本地有数据（缺了就去 Tushare 补）
    df = _query_prices(ts_code, start_date, end_date)
    if df.empty:
        return {"code": 404, "msg": "该区间内无数据"}

    # 智能选图：数据点多（>20）用折线图并均匀抽样成 10 个点，避免柱状图挤成一片看不出走势；
    # 数据点少就直接用柱状图，更直观。
    if len(df) > 20:
        idx = np.linspace(0, len(df) - 1, 10, dtype=int)
        df_plot = df.iloc[idx]
        chart_type = "line"
    else:
        df_plot = df
        chart_type = "bar"
    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
        plt.bar(df_plot["trade_date"], df_plot["close"], label="收盘价")
    else:
        plt.plot(df_plot["trade_date"], df_plot["close"], marker="o", label="收盘价")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.title(f"{ts_code} 股票行情")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_b64 = _fig_to_base64()

    return {
        "code": 0, "msg": "success",
        "data": {
            "columns": list(df.columns),
            "rows": df.fillna("").values.tolist(),  # NaN 转空字符串，避免 JSON 序列化报错
            "image_base64": img_b64,
        },
    }


async def arima_forecast_api(request: Request):
    """POST /ai/stock-analysis/arima {ts_code, n}
    用 ARIMA(5,1,5) 模型对收盘价建模，预测未来 n 天的价格。
    (5,1,5) 里 p=5（跟过去 5 天自身值的线性组合）、d=1（一次差分，把非平稳的价格序列变平稳）、
    q=5（跟过去 5 天噪声项的线性组合）——固定参数不做自动调参，图省事也图稳定可复现。
    """
    body = await read_json_optional(request) or {}
    ts_code = (body.get("ts_code") or "").strip()
    n = body.get("n")
    if not ts_code or not n:
        return {"code": 400, "msg": "缺少 ts_code 或 n"}
    n = int(n)
    start_date, end_date = _default_range()  # ARIMA 固定用近一年数据训练，不支持自定义区间

    _sync_prices(ts_code, start_date, end_date)
    df = _query_prices(ts_code, start_date, end_date)
    if len(df) < _MIN_ARIMA_ROWS:
        return {"code": 400, "msg": "历史数据不足，无法进行 ARIMA 建模预测"}

    from statsmodels.tsa.arima.model import ARIMA  # 延迟 import，只有真正建模时才加载

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    model = ARIMA(close, order=(5, 1, 5))
    fitted = model.fit()
    forecast = fitted.forecast(steps=n)

    # 预测日期：直接按自然日往后推（不区分交易日/非交易日），跟 ARIMA 模型本身按等间隔序列
    # 建模的假设保持一致，够用即可，不做交易日历对齐。
    last_date = pd.to_datetime(df["trade_date"].iloc[-1])
    pred_dates = [(last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(n)]
    pred_values = [float(v) for v in forecast]

    # 画图：历史收盘价折线 + 预测收盘价折线拼在一张图上，方便直观看出预测是延续原趋势还是转向
    plt.figure(figsize=(10, 6))
    plt.plot(df["trade_date"], close, label="历史收盘价")
    plt.plot(pred_dates, pred_values, marker="o", label="预测收盘价")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.title(f"{ts_code} 收盘价 ARIMA 预测")
    plt.legend()
    # 横坐标日期太多会挤成一团看不清，超过 12 个就等间隔抽样约 10 个刻度显示
    all_dates = list(df["trade_date"]) + pred_dates
    total_len = len(all_dates)
    if total_len > 12:
        step = max(1, total_len // 10)
        show_idx = list(range(0, total_len, step))
        plt.xticks(show_idx, [all_dates[i] for i in show_idx], rotation=45)
    else:
        plt.xticks(rotation=45)
    plt.tight_layout()
    img_b64 = _fig_to_base64()

    # 用预测的最后一天相对当前最新收盘价的涨跌幅，生成一句自然语言的趋势说明
    # （±1% 以内算"相对平稳"，避免把正常波动也说成"走高/走低"）
    last_close = float(close.iloc[-1])
    last_pred = pred_values[-1]
    pct = (last_pred - last_close) / last_close * 100 if last_close else 0.0
    if pct > 1:
        trend = "预计持续走高"
    elif pct < -1:
        trend = "预计持续走低"
    else:
        trend = "预计相对平稳"

    return {
        "code": 0, "msg": "success",
        "data": {
            "forecast": [{"date": d, "predicted_close": v} for d, v in zip(pred_dates, pred_values)],
            "trend_summary": f"{trend}，较最近收盘价 {last_close:.2f} 变动约 {pct:.2f}%",
            "image_base64": img_b64,
        },
    }


async def boll_detection_api(request: Request):
    """POST /ai/stock-analysis/boll {ts_code, start_date?, end_date?}
    布林带（Bollinger Bands）超买超卖检测：20 日均线 ± 2 倍标准差作为上下轨，
    收盘价突破上轨记为"超买"，跌破下轨记为"超卖"。
    """
    body = await read_json_optional(request) or {}
    ts_code = (body.get("ts_code") or "").strip()
    if not ts_code:
        return {"code": 400, "msg": "缺少 ts_code"}
    start_date, end_date = _parse_range(body)

    _sync_prices(ts_code, start_date, end_date)
    df = _query_prices(ts_code, start_date, end_date)
    if len(df) < _MIN_BOLL_ROWS:
        return {"code": 400, "msg": "历史数据不足，无法进行布林带检测"}

    # 计算布林带三条线：中轨（20 日均线）、上轨（中轨 + 2σ）、下轨（中轨 - 2σ）
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["STD20"] = df["close"].rolling(window=20).std()
    df["UPPER"] = df["MA20"] + 2 * df["STD20"]
    df["LOWER"] = df["MA20"] - 2 * df["STD20"]
    overbought = df[df["close"] > df["UPPER"]][["trade_date", "close"]]  # 收盘价突破上轨：超买
    oversold = df[df["close"] < df["LOWER"]][["trade_date", "close"]]     # 收盘价跌破下轨：超卖

    # 画图：收盘价 + 三条布林带线，上下轨之间填充灰色阴影表示"正常波动区间"，
    # 超买/超卖的点用红/蓝散点单独标出，一眼能看出异常点分布在哪个时间段
    plt.figure(figsize=(12, 6))
    plt.plot(df["trade_date"], df["close"], label="收盘价")
    plt.plot(df["trade_date"], df["MA20"], label="MA20")
    plt.plot(df["trade_date"], df["UPPER"], label="上轨+2σ")
    plt.plot(df["trade_date"], df["LOWER"], label="下轨-2σ")
    plt.fill_between(df["trade_date"], df["UPPER"], df["LOWER"], color="gray", alpha=0.1)
    plt.scatter(overbought["trade_date"], overbought["close"], color="red", label="超买", zorder=5)
    plt.scatter(oversold["trade_date"], oversold["close"], color="blue", label="超卖", zorder=5)
    total_len = len(df)
    if total_len > 12:
        step = max(1, total_len // 10)
        show_idx = list(range(0, total_len, step))
        plt.xticks(show_idx, [df["trade_date"].iloc[i] for i in show_idx], rotation=45)
    else:
        plt.xticks(rotation=45)
    plt.xlabel("日期")
    plt.ylabel("价格")
    plt.title(f"{ts_code} 布林带异常点检测")
    plt.legend()
    plt.tight_layout()
    img_b64 = _fig_to_base64()

    return {
        "code": 0, "msg": "success",
        "data": {
            "overbought": overbought.values.tolist(),
            "oversold": oversold.values.tolist(),
            "image_base64": img_b64,
        },
    }


async def prophet_analysis_api(request: Request):
    """POST /ai/stock-analysis/prophet {ts_code, start_date?, end_date?}
    用 Facebook Prophet 对收盘价做时间序列分解，拆出 trend（长期趋势）、weekly（周内规律）、
    yearly（年度季节性）三条分量曲线，帮助观察股价除了随机波动之外有没有周期性规律。
    这里只做"分解画图"，不做未来预测（make_future_dataframe(periods=0) 表示不外推）。
    """
    body = await read_json_optional(request) or {}
    ts_code = (body.get("ts_code") or "").strip()
    if not ts_code:
        return {"code": 400, "msg": "缺少 ts_code"}
    start_date, end_date = _parse_range(body)

    _sync_prices(ts_code, start_date, end_date)
    df = _query_prices(ts_code, start_date, end_date)
    if len(df) < _MIN_PROPHET_ROWS:
        return {"code": 400, "msg": "历史数据不足，无法进行 Prophet 周期性分析"}

    from prophet import Prophet  # 延迟 import：这个库比较重，只有真正用到时才加载

    # Prophet 要求输入的两列必须严格命名为 ds（日期）和 y（数值）
    pdf = pd.DataFrame({
        "ds": pd.to_datetime(df["trade_date"]),
        "y": pd.to_numeric(df["close"], errors="coerce"),
    }).dropna()

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(pdf)
    # periods=0：只对历史区间做分解，不外推预测未来
    forecast = m.predict(m.make_future_dataframe(periods=0))
    # plot_components 是 Prophet 自带的分解图（trend/weekly/yearly 各占一个子图），直接用它生成的
    # figure 存图，不用再自己手写画图逻辑
    fig = m.plot_components(forecast)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    return {
        "code": 0, "msg": "success",
        "data": {"image_base64": img_b64},
    }
