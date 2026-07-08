# 架构设计

> 详细说明 Tennis Rally Clipper 的分层架构、模块职责和数据流。

---

## 架构总览

```text
┌─────────────────────────────────────────────────────┐
│  用户层: CLI / HTML Report / 未来可选 UI              │
├─────────────────────────────────────────────────────┤
│  筛选层: 规则引擎 / 启发式标签 / LLM 语义筛选            │
├─────────────────────────────────────────────────────┤
│  切分层: 回合切分 / 开始结束检测 / 后处理               │
├─────────────────────────────────────────────────────┤
│  特征层: 运动 / 球员 / 姿态 / 球 / 音频 特征聚合         │
├─────────────────────────────────────────────────────┤
│  视觉层: OpenCV / YOLO / MediaPipe / 自定义检测器        │
├─────────────────────────────────────────────────────┤
│  视频层: FFmpeg / 代理视频 / 元信息 / 裁剪拼接           │
└─────────────────────────────────────────────────────┘
```

---

## 设计原则

### 1. 分层渐进

每一层都建立在前一层之上，可独立运行：

- **Layer 1 (视频)**: 只要有 FFmpeg 就能工作
- **Layer 2 (运动 + 音频)**: 纯 CV + 音频信号，不需要 AI 模型
- **Layer 3 (球员)**: 叠加 YOLO 检测，提升精度
- **Layer 4 (姿态/球)**: 更精细的特征，需要更多计算
- **Layer 5 (LLM)**: 可选的高级语义筛选

### 2. 结构化优先

高维视频 → 低维结构化特征 → 轻量模型/规则，而非直接把视频喂给大模型。

### 3. 本地优先

- 默认所有处理在本地完成
- 不强制上传原视频
- 云端 LLM 仅用于可选精筛，且只上传结构化摘要

### 4. 偏召回

宁可多保留，不要误删好球。输出保留复核和重新导出能力。

---

## 数据流（MVP 已实现）

```text
input.mp4
  ├─→ video.ingest (元信息)
  ├─→ video.proxy (540p, 可选)
  ├─→ vision.motion (5fps 帧差 motion energy)
  └─→ audio.onset (22.05kHz WAV → 击球 onset)
         ↓
  features.extract (聚合运动 + 音频特征)
         ↓
  features.fuse_hit_events (运动峰值 + 音频 onset 确认)
         ↓
  segmentation.rules (击球事件聚类)
         ↓
  segmentation.postprocess (pre-roll / post-roll / 过滤)
         ↓
  export.concat (FFmpeg 裁剪原视频并拼接)
         ↓
  timeline.json + timeline.csv + hit_events.json
```

---

## 模块职责

### 视频层 (`tenniscut/video/`)

| 模块 | 文件 | 职责 |
|---|---|---|
| 导入 | `ingest.py` | 读取视频元信息，按指定 fps 读取帧 |
| 代理 | `proxy.py` | 生成低分辨率代理视频 |
| FFmpeg | `ffmpeg.py` | 裁剪、拼接、音频提取 |

### 视觉层 (`tenniscut/vision/`)

| 模块 | 文件 | 状态 |
|---|---|---|
| 运动检测 | `motion.py` | ✅ 已实现 |
| ROI | `roi.py` | Phase 2 |
| 球员检测 | `players.py` | Phase 3 |
| 姿态估计 | `pose.py` | Phase 3 |
| 球检测 | `ball.py` | Phase 4 |

### 特征层 (`tenniscut/features/`)

| 模块 | 文件 | 职责 |
|---|---|---|
| 提取 | `extract.py` | 聚合运动、音频特征，音频-运动融合 |
| Schema | `schema.py` | 定义特征数据结构 |

### 切分层 (`tenniscut/segmentation/`)

| 模块 | 文件 | 职责 |
|---|---|---|
| Active Score | `active_score.py` | 多信号加权活跃分数 |
| 规则切分 | `rules.py` | 阈值切分 + 击球事件聚类 |
| 后处理 | `postprocess.py` | 前后余量 + 过滤短片段 |

**MVP 切分策略**：以确认的击球事件为锚点，相邻击球间隔 > 6s 判定为新的回合。

### 筛选层 (`tenniscut/labeling/`)

| 模块 | 文件 | 状态 |
|---|---|---|
| 启发式标签 | `heuristics.py` | Phase 4 |
| LLM 筛选 | `llm_filter.py` | Phase 5 |

### 导出层 (`tenniscut/export/`)

| 模块 | 文件 | 输出 |
|---|---|---|
| 片段 | `clips.py` | 单个片段 |
| 拼接 | `concat.py` | 完整拼接视频 |
| EDL | `edl.py` | 专业剪辑软件格式（Phase 5） |

### 复核层 (`tenniscut/review/`)

| 模块 | 文件 | 输出 |
|---|---|---|
| HTML 报告 | `html_report.py` | `report.html`（Phase 2） |

---

## 扩展新检测信号

1. 在 `tenniscut/vision/` 或 `tenniscut/audio/` 创建新模块
2. 输出标准化 JSON，每行包含 `t` 时间戳
3. 在 `features.extract` 中注册新的特征列
4. 在 `segmentation.rules` 或 `active_score` 中使用

---

> 更多技术细节见各模块源码文档字符串。
