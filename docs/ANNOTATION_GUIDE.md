# 数据标注指南

> 本文档说明如何为 Tennis Rally Clipper 标注训练数据。
>
> ⚠️ 标注数据**不上传至 GitHub**，仅本地使用。

---

## 标注目标

为模型提供"黄金标准"回合切分，用于：
1. 验证自动切分 baseline 的准确性
2. 训练后续的标签分类器
3. 评估不同模块的改进效果

---

## 标注粒度

| 字段 | 精度 | 说明 | 示例 |
|---|---|---|---|
| `rally_start` | 0.5–1 秒 | 回合开始（发球准备/第一击前） | 83.5 |
| `rally_end` | 0.5–1 秒 | 回合结束（球落地/得分/失误后） | 101.9 |
| `keep` | 片段级 | 1=保留, 0=丢弃 | 1 |
| `label` | 片段级 | 可多选 | `long_rally`, `highlight` |

### 标签定义

| 标签 | 定义 | 用途 |
|---|---|---|
| `rally` | 正常回合 | 基础保留标签 |
| `serve_practice` | 发球练习 | 训练场景 |
| `warmup` | 热身 | 通常丢弃 |
| `dead_time` | 捡球/等待/聊天 | 必须丢弃 |
| `long_rally` | 多拍（≥6 拍或≥8秒） | 筛选好球 |
| `highlight` | 精彩回合候选 | 精选导出 |
| `error` | 失误候选 | 教练回看 |
| `uncertain` | 无法确定 | 给模型保留模糊空间 |

---

## 标注工具

### 方法 1: CSV 手动标注（推荐 MVP 阶段）

使用简单的 CSV 格式，配合视频播放器手动记录时间：

```csv
video_id,start,end,keep,label,notes
session_001,83.42,101.88,1,long_rally,"good rally"
session_001,145.20,151.70,0,dead_time,"picking balls"
session_001,210.50,225.30,1,highlight,"amazing point"
```

**推荐工具**：
- VLC 播放器（按 `Ctrl + T` 查看精确时间）
- QuickTime（macOS 精确到 0.01 秒）
- 任何能显示当前时间的视频播放器

### 方法 3: 从人工剪辑视频反推 benchmark（推荐有参考集锦时）

若你已有「原片 + 人工剪辑集锦」，可用视觉帧对齐自动生成 ground truth，无需逐段标注 original 时间：

```bash
python scripts/extract_benchmark.py \
  --original /path/to/raw.MOV \
  --result   /path/to/result.mp4 \
  --output   sessions/my_session/benchmark.json \
  --result-cuts 51 70 139 213 267   # result 内的切分点（秒）
```

**你需要提供：**
- 原片路径 + 剪辑集锦路径
- result 视频的切分点（`--result-cuts`），例如已知 6 段：`0–51, 51–70, 70–139, 139–213, 213–267, 267–332`

**你不需要提供：**
- 每段在原片的起止时间（由视觉匹配自动计算）

**对齐逻辑简述：**
1. 按切分点把 result 拆成若干段
2. 对 original 建帧指纹索引（dHash + HSV，可缓存）
3. 每段取 8 帧 probe，在原片中滑动匹配最佳起点
4. 假设 1:1 播放，`original_end = original_start + 段时长`

自动硬切检测（不传 `--result-cuts`）也可运行，但在网球视频中可能把回合内镜头变化误判为切点；硬切算法优化留待后续，**当前建议手动指定切分点**。

生成后用 `scripts/eval_baseline.py` 与 `work/timeline.json` 对比。

---

### 方法 2: 轻量 HTML 标注页（ML 数据集，进行中）

球员动作与球检测标注将使用 `datasets/` 下的 HTML 标注工具（见 [`datasets/README.md`](../datasets/README.md)），支持：
- 球员 crop 逐条分类（hit_serve / hit_rally / move 等）
- 球帧 bbox 审核
- 按 session split 导出 train/val/test

---

## 标注策略

### 1. 先标完整 session，不要只标精选片段

模型需要学习"捡球/等待"长什么样，也需要知道"正常回合"长什么样。

**推荐**：
- 标注 5–10 个完整 session（每个 30–60 分钟）
- 覆盖不同场景：白天/夜间、不同场地、不同对手

### 2. 保留 `uncertain` 标签

如果无法确定是回合还是捡球，不要强行判断：
- 标记为 `uncertain`
- 让模型学习"模糊样本"
- 避免把错误信号灌给模型

### 3. 按 session 切分数据集

⚠️ **绝不按片段随机切分**。

正确做法：
- **Train**: 60% 的 sessions（完整场次）
- **Validation**: 20% 的 sessions
- **Test**: 20% 的 sessions（全新场景）

原因：同一场视频中光照、机位、球员衣服、场地背景高度相似。随机按片段切分会导致模型高估泛化能力。

---

## 质量检查

标注完成后，建议进行以下检查：

1. **时间连续性检查**：标注的回合之间不应该有大段重叠或遗漏
2. **时长合理性**：正常回合通常 3–30 秒，过长（>60s）可能包含多个回合
3. **keep/drop 比例**：确保模型能看到足够的 `dead_time` 样本
4. **交叉验证**：找另一个球友标注同一段视频，对比一致性

---

## 标注示例流程

```bash
# 1. 准备一个完整视频
# 2. 用视频播放器逐段观看
# 3. 每遇到一个回合/片段，记录：
#    - start 时间（按空格暂停，记录时间）
#    - end 时间
#    - 判断 keep/drop
#    - 选择标签
# 4. 保存为 CSV
# 5. 用 scripts/eval_baseline.py 对比自动结果

# 或者：从人工剪辑集锦反推 benchmark
python scripts/extract_benchmark.py \
  --original raw.MOV --result result.mp4 \
  --output benchmark.json --result-cuts 51 70 139 213 267
python scripts/eval_baseline.py \
  --predicted work/timeline.json --ground-truth benchmark.json
```

---

> 📌 更多标注工具开发中。如有问题，请在 GitHub Issue 中讨论。
