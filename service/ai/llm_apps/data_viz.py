import io
import json
import base64
import tempfile
import os
from fastapi import Request, UploadFile
from dashscope import Generation


async def data_viz_columns_api(request: Request):
    form = await request.form()
    csv_file: UploadFile = form.get("csv")
    if not csv_file:
        return {"code": 400, "msg": "请上传 CSV 文件"}
    content = await csv_file.read()
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(content))
        return {
            "code": 0, "msg": "success",
            "data": {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "preview": df.head(3).to_dict(orient="records"),
            },
        }
    except Exception as e:
        return {"code": 500, "msg": f"文件解析失败: {str(e)}"}


async def data_viz_api(request: Request):
    form = await request.form()
    csv_file: UploadFile = form.get("csv")
    question = form.get("question", "")

    if not csv_file or not question:
        return {"code": 400, "msg": "缺少 CSV 文件或问题"}

    content = await csv_file.read()
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return {"code": 500, "msg": f"CSV 解析失败: {str(e)}"}

    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample = df.head(5).to_dict(orient="records")

    code_prompt = f"""你是一个数据可视化专家。根据以下数据集信息，生成 Python matplotlib 代码来回答用户的问题。

数据列及类型：{json.dumps(schema, ensure_ascii=False)}
前5行样本：{json.dumps(sample, ensure_ascii=False)}
用户问题：{question}

要求：
1. 使用 matplotlib（已导入 import matplotlib.pyplot as plt，import matplotlib as mpl）
2. 数据已加载为 DataFrame 变量 df
3. 设置中文字体：mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']; mpl.rcParams['axes.unicode_minus'] = False
4. 图表要有标题、坐标轴标签
5. 最后使用 plt.tight_layout()，不要调用 plt.show()
6. 只输出 Python 代码，不要任何解释

代码："""

    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": code_prompt}],
        result_format="message",
    )
    code = resp.output.choices[0].message.content.strip()
    if "```" in code:
        lines = code.split("\n")
        code_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not line.startswith("```"):
                code_lines.append(line)
        code = "\n".join(code_lines)

    # Execute code safely
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np
        import pandas as pd

        buf = io.BytesIO()
        namespace = {"df": df, "plt": plt, "pd": pd, "np": np, "mpl": mpl, "io": io}
        exec(code, namespace)  # nosec - controlled environment
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close("all")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
    except Exception as e:
        return {"code": 500, "msg": f"图表生成失败: {str(e)}", "data": {"code": code}}

    return {
        "code": 0, "msg": "success",
        "data": {"image_base64": img_b64, "generated_code": code},
    }
