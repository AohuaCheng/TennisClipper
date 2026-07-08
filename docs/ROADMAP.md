# 开发路线图

> 本文档列出 Tennis Rally Clipper 的未来开发阶段。
>
> 标记说明：`#P0`=必须完成，`#P1`=重要，`#P2`=可选/后续优化

---

## 状态总览

| 阶段 | 目标 | 状态 |
|---|---|---|
| Phase 1 | 最小可运行管线：输入视频 → 自动切分 → 导出 | ✅ 已完成（已集成音频+运动融合） |
| Phase 2 | ROI 与复核机制：提升可用性，降低误检 | ⏳ 待开始 |
| Phase 3 | 球员检测与姿态特征：让切分不只依赖运动 | ⏳ 待开始 |
| Phase 4 | 回合标签与筛选：long_rally / highlight 等 | ⏳ 待开始 |
| Phase 5 | 自然语言筛选与 AI 精筛 | ⏳ 待开始 |

---

## Phase 2: ROI 与复核机制

**目标**：提升可用性，降低误检

| # | 任务 | 优先级 | 对应模块 |
|---|---|---|---|
| 2.1 | 手动/半自动 ROI 配置 | #P0 | `vision/roi.py` |
| 2.2 | 保存/加载不同场地配置 | #P0 | `config.py` |
| 2.3 | 生成 HTML 复核报告 | #P0 | `review/html_report.py` |
| 2.4 | 每个片段附上缩略图/GIF | #P1 | `review/html_report.py` |
| 2.5 | 支持人工修改 keep/drop | #P0 | `review/html_report.py` |
| 2.6 | 支持修正 start/end 时间 | #P0 | `review/html_report.py` |
| 2.7 | 修改后重新导出 | #P0 | `export/` |
| 2.8 | court_config.json 持久化 | #P1 | `config.py` |

---

## Phase 3: 球员检测与姿态特征

**目标**：让切分不只依赖运动量

| # | 任务 | 优先级 | 对应模块 | 备注 |
|---|---|---|---|---|
| 3.1 | 集成 YOLO 人体检测 | #P0 | `vision/players.py` | 需安装 ultralytics |
| 3.2 | 区分近端/远端球员 | #P0 | `vision/players.py` | 基于位置和大小 |
| 3.3 | 计算球员运动轨迹 | #P0 | `vision/players.py` | 连续帧跟踪 |
| 3.4 | 可选集成 MediaPipe 姿态 | #P1 | `vision/pose.py` | 轻量方案 |
| 3.5 | 提取挥拍候选特征 | #P1 | `vision/pose.py` | 手腕速度、角度变化 |
| 3.6 | 更新 active score 权重 | #P0 | `segmentation/active_score.py` | 加入球员信号 |
| 3.7 | 与 Phase 1 baseline 对比 | #P0 | `scripts/eval_baseline.py` | 量化提升 |

---

## Phase 4: 回合标签与筛选

**目标**：从“剪掉废片”升级为“按需求挑片”

| # | 任务 | 优先级 | 对应模块 |
|---|---|---|---|
| 4.1 | 计算回合级特征 | #P0 | `features/extract.py` |
| 4.2 | 自动标签: long_rally | #P0 | `labeling/heuristics.py` |
| 4.3 | 自动标签: high_motion | #P0 | `labeling/heuristics.py` |
| 4.4 | 自动标签: short_rally | #P1 | `labeling/heuristics.py` |
| 4.5 | 自动标签: highlight_candidate | #P1 | `labeling/heuristics.py` |
| 4.6 | 支持结构化 filter 导出 | #P0 | `export/` |
| 4.7 | 标签质量评估 | #P1 | `scripts/eval_baseline.py` |
| 4.8 | 导出不同版本对比 | #P1 | `export/` |

---

## Phase 5: 自然语言与 AI 精筛

**目标**：支持“说想剪什么”

| # | 任务 | 优先级 | 对应模块 | 备注 |
|---|---|---|---|---|
| 5.1 | 定义 filter JSON schema | #P0 | `schemas/filter.schema.json` | |
| 5.2 | LLM 把自然语言转成 filter | #P0 | `labeling/llm_filter.py` | 本地或 API |
| 5.3 | 低置信度片段二次判断 | #P1 | `labeling/llm_filter.py` | 只处理少量片段 |
| 5.4 | explain 字段 | #P1 | `labeling/llm_filter.py` | 告诉用户为什么保留 |
| 5.5 | 导出 EDL 剪辑列表 | #P2 | `export/edl.py` | Premiere/DaVinci |
| 5.6 | 轻量可视化 UI | #P2 | `review/ui.py` | 可选，远期 |

---

> 使用本文档时，建议按 Phase 创建 GitHub Project 看板，并将任务转为 Issue 分配。
