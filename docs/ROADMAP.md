# 开发路线图

> 本文档列出 Tennis Rally Clipper 的开发阶段与当前状态。
>
> 标记说明：`#P0`=必须完成，`#P1`=重要，`#P2`=可选/后续优化

---

## 状态总览

| 阶段 | 目标 | 状态 |
|---|---|---|
| Phase 1 | 最小可运行管线：输入视频 → 自动切分 → 导出 | ✅ 已完成 |
| Phase 2 | ROI、复核、球场标定 | 🔄 部分完成 |
| Phase 3 | 视觉击球：球员/姿态/球轨迹 | 🔄 原型已实现，精度不足 |
| Phase 3.5 | 回合 lifecycle + ball refine | ✅ 已实现，待感知层加强 |
| Phase 4 | ML 数据集 + 专用模型 + VLM 基线 | 🔄 进行中 |
| Phase 5 | 回合标签筛选 + LLM 精筛 | ⏳ 待开始 |

---

## Phase 2: ROI 与复核机制

**目标**：提升可用性，降低误检

| # | 任务 | 优先级 | 状态 |
|---|---|---|---|
| 2.1 | 手动/半自动 ROI 配置 | #P0 | ✅ `vision/roi.py` |
| 2.2 | 球场点击标定 | #P0 | ✅ `calibrate-court` |
| 2.3 | 生成 HTML 复核报告 | #P0 | ⏳ |
| 2.4 | 支持人工修改 keep/drop | #P0 | ⏳ |
| 2.5 | court_config.json 持久化 | #P1 | ⏳ |

> 自动 Hough 球场线检测已放弃。

---

## Phase 3: 视觉击球与球轨迹

**目标**：用视觉信号替代音频/纯运动切分

| # | 任务 | 优先级 | 状态 |
|---|---|---|---|
| 3.1 | YOLO 人体检测 | #P0 | ✅ `vision/players.py` |
| 3.2 | 区分近端/远端球员 | #P0 | ✅ |
| 3.3 | MediaPipe 挥拍检测 | #P0 | ✅ `hit_detection_visual.py` |
| 3.4 | 球颜色+运动联合检测 | #P0 | ✅ `vision/ball.py` |
| 3.5 | 球轨迹跟踪 | #P1 | ✅ 覆盖率有限 |
| 3.6 | rally_lifecycle 回合推断 | #P0 | ✅ |
| 3.7 | ball_refine 边界修正 | #P1 | ✅ |
| 3.8 | benchmark IoU 评估 | #P0 | ✅ `eval_baseline.py` |

**已知瓶颈**：挥拍误检（转身≈击球）、球轨迹稀疏 → 需 Phase 4 ML 模型。

---

## Phase 4: ML 数据集与专用模型

**目标**：提升击球动作与球检测的基础识别质量

| # | 任务 | 优先级 | 状态 |
|---|---|---|---|
| 4.1 | `datasets/` 脚手架 + Clipper 扫描 | #P0 | ✅ |
| 4.2 | 球员动作标注 + 导出 crop | #P0 | ⏳ |
| 4.3 | Qwen-VL-Chat-Int4 基线评估 | #P0 | ⏳ |
| 4.4 | 球检测标注 + YOLO 微调 | #P1 | ⏳ |
| 4.5 | 接入 pipeline 下游 IoU 验证 | #P0 | ⏳ |
| 4.6 | 回合标签: long_rally / highlight | #P1 | ⏳ |

详见 [`datasets/README.md`](../datasets/README.md)。

---

## Phase 5: 自然语言与 AI 精筛

**目标**：支持“说想剪什么”

| # | 任务 | 优先级 | 对应模块 |
|---|---|---|---|
| 5.1 | 定义 filter JSON schema | #P0 | `schemas/filter.schema.json` |
| 5.2 | LLM 自然语言 → filter | #P0 | `labeling/llm_filter.py` |
| 5.3 | 导出 EDL 剪辑列表 | #P2 | `export/edl.py` |
| 5.4 | 轻量可视化 UI | #P2 | `review/ui.py` |

---

> 使用本文档时，建议按 Phase 创建 GitHub Project 看板，并将任务转为 Issue 分配。
