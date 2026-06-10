#!/bin/bash

# =============================================================================
# run-automation.sh - 全栈 AI 功能自动化执行器
# =============================================================================
# 每个功能拆成两个独立 session（后端 + 前端），避免单 session token 爆炸：
#   Session 1（后端）：只做 service-home 的代码 + 路由 + curl 测试 + commit
#   Session 2（前端）：只做 ai-dashboard 的页面 + 路由 + build 测试 + commit
#
# 使用方式：./run-automation.sh <执行次数>
# 示例：./run-automation.sh 7
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

LOG_DIR="./automation-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/automation-$(date +%Y%m%d_%H%M%S).log"

log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" >> "$LOG_FILE"
    case $level in
        INFO)    echo -e "${BLUE}[INFO]${NC} ${message}" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} ${message}" ;;
        WARNING) echo -e "${YELLOW}[WARNING]${NC} ${message}" ;;
        ERROR)   echo -e "${RED}[ERROR]${NC} ${message}" ;;
        PROGRESS)echo -e "${CYAN}[PROGRESS]${NC} ${message}" ;;
    esac
}

count_remaining_tasks() {
    if [ -f "task.json" ]; then
        grep -c '"passes": false' task.json 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# 获取当前最小 passes:false 的 task id（用于前端 session 知道做哪个任务）
get_current_task_id() {
    python3 -c "
import json
with open('task.json') as f:
    data = json.load(f)
for t in sorted(data['tasks'], key=lambda x: x['id']):
    if not t['passes']:
        print(t['id'])
        break
" 2>/dev/null || echo "0"
}

if [ -z "$1" ]; then
    echo "用法: $0 <执行次数>"
    echo "示例: $0 7"
    exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "错误：参数必须是正整数"
    exit 1
fi

TOTAL_RUNS=$1

echo ""
echo "========================================"
echo "  全栈 AI 功能自动化执行器"
echo "  每个功能 = 后端 session + 前端 session"
echo "========================================"
echo ""

log "INFO" "启动自动化，计划执行 $TOTAL_RUNS 个功能"
log "INFO" "日志文件：$LOG_FILE"

if [ ! -f "task.json" ]; then
    log "ERROR" "task.json 不存在！请在 service-home 目录运行此脚本。"
    exit 1
fi

INITIAL_TASKS=$(count_remaining_tasks)
log "INFO" "初始待完成功能数：$INITIAL_TASKS"

for ((run=1; run<=TOTAL_RUNS; run++)); do
    echo ""
    echo "========================================"
    log "PROGRESS" "第 $run 个功能（共 $TOTAL_RUNS 个）"
    echo "========================================"

    REMAINING=$(count_remaining_tasks)
    if [ "$REMAINING" -eq 0 ]; then
        log "SUCCESS" "所有功能已全部完成！"
        break
    fi

    TASK_ID=$(get_current_task_id)
    log "INFO" "当前任务 ID：$TASK_ID，剩余：$REMAINING 个"

    RUN_START=$(date +%s)

    # ------------------------------------------------------------------
    # Session 1：后端（service-home）
    # ------------------------------------------------------------------
    log "INFO" "▶ [Session 1/2] 后端实现 - Task $TASK_ID"
    BACKEND_LOG="$LOG_DIR/run-${run}-task${TASK_ID}-backend-$(date +%H%M%S).log"

    BACKEND_PROMPT=$(mktemp)
    cat > "$BACKEND_PROMPT" << PROMPT_EOF
读取 /Users/lihong/Desktop/personal/code/service-home/task.json，找到 id 为 ${TASK_ID} 的任务。

只完成【后端部分】，按照 backend_steps 逐步实现：
1. 安装需要的 Python 依赖（如有）
2. 在 service/ai/ 目录创建服务模块
3. 在 routes/ai.py 顶部添加 import，在 register_ai() 末尾注册路由（使用 _ai_route，不用装饰器）
4. 运行导入检查：python -c "from routes.ai import register_ai; print('OK')"
5. curl 测试每个新增 API endpoint（服务器已运行在 http://localhost:3000）
6. 在 /Users/lihong/Desktop/personal/code/service-home/progress.txt 末尾追加后端实现记录
7. git add . && git commit -m "feat: [Task ${TASK_ID}] $(python3 -c "import json; t=[x for x in json.load(open('/Users/lihong/Desktop/personal/code/service-home/task.json'))['tasks'] if x['id']==${TASK_ID}][0]['name']" 2>/dev/null || echo "Task ${TASK_ID}") - 后端实现"

注意：
- 不要修改 ai-dashboard 的任何文件
- 不要将 task.json 的 passes 改为 true（等前端完成后再改）
- 遇到阻塞立即停止，不要提交
PROMPT_EOF

    if claude -p \
        --dangerously-skip-permissions \
        --max-turns 40 \
        --allowed-tools "Bash Edit Read Write Glob Grep Task WebFetch" \
        < "$BACKEND_PROMPT" 2>&1 | tee "$BACKEND_LOG"; then
        log "SUCCESS" "后端 session 完成"
    else
        log "WARNING" "后端 session 结束（退出码 $?）"
    fi
    rm -f "$BACKEND_PROMPT"

    # ------------------------------------------------------------------
    # Session 2：前端（ai-dashboard）
    # ------------------------------------------------------------------
    log "INFO" "▶ [Session 2/2] 前端实现 - Task $TASK_ID"
    FRONTEND_LOG="$LOG_DIR/run-${run}-task${TASK_ID}-frontend-$(date +%H%M%S).log"

    FRONTEND_PROMPT=$(mktemp)
    cat > "$FRONTEND_PROMPT" << PROMPT_EOF
读取 /Users/lihong/Desktop/personal/code/service-home/task.json，找到 id 为 ${TASK_ID} 的任务。

只完成【前端部分】，按照 frontend_steps 逐步实现：
1. 在 /Users/lihong/Desktop/personal/code/ai-dashboard/src/service/ 创建 API 调用层文件
2. 在 /Users/lihong/Desktop/personal/code/ai-dashboard/src/pages/ 创建页面组件
3. 修改 /Users/lihong/Desktop/personal/code/ai-dashboard/src/config/routes.tsx，在 skillsRoutes 末尾（coze 之前）添加路由
4. 修改 /Users/lihong/Desktop/personal/code/ai-dashboard/src/layouts/MainLayout.tsx，在 skillsMenuItems children 中（Portal 之前）添加菜单项
5. cd /Users/lihong/Desktop/personal/code/ai-dashboard && npm run build（必须无 TypeScript 错误）
6. 【端到端验收】读取 task.json 中该任务的 e2e_test 字段，用 Playwright MCP 工具按照步骤操作浏览器：
   - 打开 http://localhost:5173，导航到新页面
   - 按 e2e_test 描述执行真实操作（上传文件/输入内容/点击按钮）
   - 验证后端返回数据正确显示在页面上
   - 每个关键步骤截图保存到 /Users/lihong/Desktop/personal/code/service-home/test-fixtures/screenshots/task${TASK_ID}_*.png
   - 如果 e2e_test 操作失败（后端报错/页面无响应），记录失败原因，但仍可提交（前端实现本身正确即可）
7. 将 /Users/lihong/Desktop/personal/code/service-home/task.json 中 id 为 ${TASK_ID} 的任务 passes 改为 true
8. cd /Users/lihong/Desktop/personal/code/ai-dashboard && git add . && git commit -m "feat: [Task ${TASK_ID}] $(python3 -c "import json; t=[x for x in json.load(open('/Users/lihong/Desktop/personal/code/service-home/task.json'))['tasks'] if x['id']==${TASK_ID}][0]['name']" 2>/dev/null || echo "Task ${TASK_ID}") - 前端实现"

注意：
- 不要修改 service-home 的代码文件
- 后端 API 已实现，可以 curl 验证：curl http://localhost:3000/ai/...
- 测试夹具文件路径：test-fixtures/sample_data.csv 和 test-fixtures/sample_resume.pdf
- 遇到阻塞立即停止，不要将 passes 改为 true，不要提交
PROMPT_EOF

    if claude -p \
        --dangerously-skip-permissions \
        --max-turns 40 \
        --allowed-tools "Bash Edit Read Write Glob Grep Task WebFetch mcp__playwright__*" \
        < "$FRONTEND_PROMPT" 2>&1 | tee "$FRONTEND_LOG"; then
        log "SUCCESS" "前端 session 完成"
    else
        log "WARNING" "前端 session 结束（退出码 $?）"
    fi
    rm -f "$FRONTEND_PROMPT"

    # ------------------------------------------------------------------
    # 检查任务是否完成 + push
    # ------------------------------------------------------------------
    REMAINING_AFTER=$(count_remaining_tasks)
    COMPLETED=$((REMAINING - REMAINING_AFTER))
    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))

    if [ "$COMPLETED" -gt 0 ]; then
        log "SUCCESS" "Task $TASK_ID 完成（前后端均已提交），耗时 ${RUN_DURATION} 秒"
        log "INFO" "推送到 GitHub..."
        cd /Users/lihong/Desktop/personal/code/service-home
        git push 2>&1 && log "SUCCESS" "service-home 推送成功" || log "WARNING" "service-home 推送失败"
        cd /Users/lihong/Desktop/personal/code/ai-dashboard
        git push 2>&1 && log "SUCCESS" "ai-dashboard 推送成功" || log "WARNING" "ai-dashboard 推送失败"
        cd /Users/lihong/Desktop/personal/code/service-home
    else
        log "WARNING" "Task $TASK_ID 未能完成（可能遇到阻塞），耗时 ${RUN_DURATION} 秒"
    fi

    log "INFO" "剩余待完成功能数：$REMAINING_AFTER"
    echo "----------------------------------------" >> "$LOG_FILE"

    if [ $run -lt $TOTAL_RUNS ] && [ "$REMAINING_AFTER" -gt 0 ]; then
        log "INFO" "等待 3 秒后继续..."
        sleep 3
    fi
done

echo ""
echo "========================================"
log "SUCCESS" "自动化执行完毕！"
echo "========================================"

FINAL_REMAINING=$(count_remaining_tasks)
TOTAL_COMPLETED=$((INITIAL_TASKS - FINAL_REMAINING))

log "INFO" "汇总："
log "INFO" "  功能总数：$TOTAL_RUNS"
log "INFO" "  完成功能：$TOTAL_COMPLETED 个"
log "INFO" "  剩余功能：$FINAL_REMAINING 个"
log "INFO" "  日志文件：$LOG_FILE"

if [ "$FINAL_REMAINING" -eq 0 ]; then
    log "SUCCESS" "所有功能已全部实现！"
else
    log "WARNING" "仍有 $FINAL_REMAINING 个功能未完成，可继续运行。"
fi
