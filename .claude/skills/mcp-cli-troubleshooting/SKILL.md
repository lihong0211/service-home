---
name: mcp-cli-troubleshooting
description: "Diagnosing and resolving MCP server and CLI tool integration issues with Codex — tool comparison, permission bottlenecks, diagnostic interpretation, installation workflows, and next-step guidance. Use this skill when installing MCP servers or CLI tools in Codex, troubleshooting integration issues (permission errors, diagnostic warnings, missing configuration), or interpreting health-check output."
trigger: "Use this skill when installing MCP servers or CLI tools in Codex, troubleshooting integration issues (permission errors, diagnostic warnings, missing configuration), or interpreting health-check output."
author: lihong0211yao
source_sessions:
  - lihong0211yao_lihong0211yao's Organization_default_019f5efd-bc64-7230-9fe2-649badcf7daa
  - lihong0211yao_lihong0211yao's Organization_default_019f5efd-d196-7d01-a561-ddb4b0e3d0bb
  - lihong0211yao_lihong0211yao's Organization_default_019f5ebc-caa1-7300-93cc-ccb2aca5661b
  - lihong0211yao_lihong0211yao's Organization_default_019f5ebc-d643-7fa2-96d6-5b227bb755bb
  - lihong0211yao_lihong0211yao's Organization_default_019f5e92-0ad4-7101-a967-30f829c98768
  - lihong0211yao_lihong0211yao's Organization_default_019f5e92-0a21-7c53-bcb7-04b80dcbe099
  - lihong0211yao_lihong0211yao's Organization_default_019ede51-216d-71b3-b33d-0f5328f6bf59
  - lihong0211yao_lihong0211yao's Organization_default_019f06e7-0b3d-7c80-88d5-64230705121a
  - lihong0211yao_lihong0211yao's Organization_default_019f06dd-75e4-7e10-825e-5a73322ebe8e
  - lihong0211yao_lihong0211yao's Organization_default_019f06d3-ecd8-7753-9a93-8a35a76548ce
  - lihong0211yao_lihong0211yao's Organization_default_0edbf6de-1746-4688-a9fc-9a4a346cf52a
contributors:
  - lihong0211yao
version: 3
created_by_agent: claude_code
created_at: 2026-07-14T19:23:32.378Z
updated_at: 2026-07-14T23:55:17.881Z
---

# MCP & CLI Tool Troubleshooting

## When to Use

Use this skill when:
- Installing or configuring MCP servers (e.g., codebase-memory-mcp, Headroom) for Codex/Claude Code
- Deciding between similar MCP/CLI tools with different tradeoffs
- Troubleshooting tool integration issues: permission errors, missing setup, tool not recognized
- Interpreting diagnostic output (e.g., `headroom doctor`, health checks with pass/warn/fail columns)
- Distinguishing sandbox/permission issues from actual tool problems
- Deciding between CLI tool options or installation modes

## Tool Comparison (when user is choosing between alternatives)

If user needs to decide between similar tools:
- **Create decision matrix** covering: core purpose, problem solved, optimization stage, mechanism, integration modes, official metrics, best use case
- **Example:** `codebase-memory-mcp` vs `Headroom`
  - `codebase-memory-mcp`: Knowledge graph → helps AI understand code structure (reduces exploration). Best for: large/unfamiliar codebases, impact analysis, call chains.
  - `Headroom`: Output compression → shrinks logs/RAG/tool responses (reduces transmission). Best for: high tool-call volume, long logs, large context.
  - Both are compatible; not mutually exclusive
- **Recommendation:** "For large codebases, prioritize codebase-memory-mcp; for high tool-call volume/long logs, prioritize Headroom. Start with whichever bottleneck you hit first."

## Troubleshooting Workflow

### 1. Identify the Blocker

When user reports "stuck" or "not working":
- **Gather info:** Ask for error message, diagnostic output, or describe what failed
- **Distinguish issues:**
  - Permission/sandbox bottleneck: "awaiting approval," system dialog hanging, permission denied on `~/.config` or `~/.codex`
  - Network/DNS: connection timeouts, DNS resolution failure
  - Tool state: missing configuration, incomplete install, tool output malformed
  - Process state: service not running, wrong port/PID
- **Recognize patterns:** If permission prompt appeared, user may need to click *button* in GUI context, not type in chat. Chat input alone cannot approve system-level file writes. If in headless/CLI context, offer terminal workaround to bypass Codex sandbox.

### 2. Diagnose Current State

- **Run diagnostic command** if tool provides one (e.g., `headroom doctor`, `codebase-memory-mcp index_status`, `npm ls`)
- **Parse output:** Extract pass/warn/fail checks. Identify which are critical vs. informational.
  - *Critical:* Core feature doesn't work (e.g., proxy not running → Headroom can't compress, codex not routed → integration failed)
  - *Warning:* Nice-to-have not configured (e.g., budget limits, shell env fallback, persistence setup)
  - *Failure on non-critical subsystem:* Non-blocking (e.g., `deployments init-user down` means no auto-start on reboot, but manual proxy works)
- **Key insight:** Many warnings don't block your workflow. Example from `headroom doctor`:
  ```
  proxy     ✓ pass  running at http://127.0.0.1:8787
  codex     ✓ pass  routed (/Users/...)
  claude    ⚠ warn  not routed (no ANTHROPIC_KEY)
  shell env ⚠ warn  ANTHROPIC_KEY unset — shell bypasses proxy
  deployments ✗ fail  init-user down
  ```
  - `proxy ✓` and `codex ✓` = Headroom works in Codex now. Restart Codex to activate.
  - `claude ⚠` only affects Claude Code CLI, not Codex.
  - `shell env ⚠` only affects direct terminal; Codex already routed.
  - `deployments ✗` means auto-start not set up; manual proxy still running.
  - **Verdict:** Ready to use now. Restart Codex. Auto-start is optional.

### 3. Suggest Next Step

**For permission blockers:**
- Explain what's being written and why (e.g., "writing global config to ~/.codex")
- **If in Codex GUI:** "Click the allow button in the system permission dialog" (required; chat input alone won't approve)
- **If in headless/CLI:** "Run directly in terminal to bypass sandbox" + exact command
- Confirm by asking for diagnostic output after

**For incomplete setup:**
- Point to exact config file and what fields are missing
- Provide copy-paste commands or sample config block
- Suggest verification step (diagnostic command or curl test)

**For diagnostic failures:**
- Explain impact on user's goal ("affects X, doesn't affect Y")
- Prioritize: "Critical to fix first: X. Optional later: Y, Z."
- Provide specific fix command or manual steps

**For unclear errors:**
- Suggest running tool's own health/status command
- Recommend checking logs (e.g., `~/.headroom/logs`, `npm debug logs`)
- Ask user to share tool's troubleshooting section from README

### 4. Verify Readiness

- Guide user through verification: "After you do X, run `[diagnostic]` and paste output"
- Parse results against readiness criteria (e.g., "When you see ✓ on proxy and ✓ on codex, you're ready; deployments fail is OK for now")
- Identify next milestone: full persistent setup, additional configuration, ready to use now, etc.
- Avoid silent assumptions; explicitly confirm what's working and what's not

## Common Patterns

### Permission Bottleneck (Sandbox vs System)

When Codex/system requests approval to write global config:
- User in *interactive Codex GUI:* "Click the permission dialog button that appears" (critical; chat input doesn't approve)
- User in *headless/terminal:* "Run this command in your local terminal (bypasses sandbox): `[exact command]`"
- Confirm success: "After, run `[diagnostic]` again and share output"
- Key: Chat approval ≠ system permission. Always verify by re-running diagnostic.

### Diagnostic Output with Pass/Warn/Fail

When tool outputs a table with status columns:
- **Separate tiers:** Critical (must pass for primary use) vs. Optional (nice-to-have)
- **Explain each warn/fail:** What does it mean in practice? Does it block your workflow?
  - Example: `claude ⚠ not routed` doesn't affect Codex; only affects Claude Code CLI
  - Example: `deployments ✗ fail` means Headroom won't auto-start on reboot, but manual proxy works
  - Example: `shell env ⚠ unset` means direct terminal requests bypass proxy, but Codex already configured
  - Example: `savings ⚠ no tokens saved yet` is normal at first; just "no requests processed yet"
- **Priority order:** Fix critical issues first (proxy must pass, main client must be routed); defer optional config (persistence, shell env, budget)

## Anti-Patterns

- **Don't assume permission prompt = user must click button.** Ask whether they're in CLI vs. GUI context first. In headless environments, offer terminal workaround.
- **Don't treat all warnings as blockers.** Parse which checks affect the user's goal. A warn on "budget not configured" doesn't prevent basic usage. A warn on "not routed" only blocks that specific client.
- **Don't ignore diagnostic output.** Tools like `headroom doctor`, `npm ls`, `python -m pip check` are authoritative; use them to verify state.
- **Don't recommend "uninstall and reinstall" without root-cause analysis.** Most issues are config/permission/state, not corrupt install.
- **Don't skip verification steps.** After suggesting a fix, ask user to re-run diagnostic and confirm the status change.

## Example Workflow

**User:** "I installed Headroom but Codex doesn't use it."

**You:**
1. **Gather:** "Run `headroom doctor` and paste the output."
2. **Diagnose:** Output shows `codex ✓ routed` and `proxy ✓ pass`. User hasn't restarted Codex since install.
3. **Suggest:** "Codex was running before Headroom started. Restart Codex, then run `headroom doctor` again."
4. **Verify:** After restart, `savings` shows token count. Confirm: "Headroom is now active in Codex."
