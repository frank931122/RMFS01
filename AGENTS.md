# 仓库规则（Repository Guidelines）

## 1. 项目结构
- 仓库根目录：`pythonProject15/`
- 主代码在：`pythonProject5/`
- 场景输入（必须保留、可复现）：`pythonProject5/scenario/**`
- 求解/导出产物（大多可再生成）：`pythonProject5/solution_exports/`、`pythonProject5/solution_samples/`
- 基准评测：
  - 基线文件：`pythonProject5/bench_baseline_demo01.json`（用于 PASS/FAIL 对比）
  - 运行输出：`bench_quick_*`（默认不提交到 git）

## 2. 关键运行命令
- 快速基准（建议日常迭代用，几分钟内完成）：
  - `powershell -ExecutionPolicy Bypass -File .\pythonProject5\bench.ps1 -Gammas "0" -Iters 200 -Seeds "0" -Tol 0.001`
- 全量验收（提交前/里程碑用，较慢）：
  - `powershell -ExecutionPolicy Bypass -File .\pythonProject5\bench.ps1 -Gammas "0,5,10" -Iters 2000 -Seeds "0" -Tol 0.001`

## 3. Codex 必须遵守的规则（非常重要）
- 只在 `pythonProject5/` 内修改代码，除非我明确要求改其它目录。
- 严禁修改：
  - `pythonProject5/bench_baseline_demo01.json`
  - `pythonProject5/scenario/**`（场景输入）
- 每次改完任何代码，必须先跑“快速基准”，并在回复中报告：
  - 本次 SCORE
  - 与 baseline 的 PASS/FAIL
  - 运行耗时（wall time）
- 如果 PASS/FAIL 为 FAIL，或者 SCORE 变差：必须撤回改动，换更小/更稳的优化。
- 禁止把生成物加入 git（例如：`bench_quick_*`、大日志、导出目录）。如有必要，先更新 `.gitignore`。
- 当前任务（本次优化目标）：
  - 优先只修改 `pythonProject5/alns_min.py`，将其完善为“可复现、可收敛、可配置”的自适应大邻域搜索（ALNS）实现。
  - 如确实需要修改其它文件（例如仅为接入参数、打印/日志、或修复导入路径），必须先说明原因，并保持改动最小化。
  - 验收：每次修改后必须跑快速基准并报告 SCORE 与 PASS/FAIL；准备合并前再跑全量验收。
## 4. 代码风格
- Python：4 空格缩进
- 函数/变量：snake_case；类名：CamelCase
- 只做小改动、可验证、可回滚：一轮优化对应一个小 commit