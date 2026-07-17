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
│  切分层: 回合切分 / lifecycle / ball_refine / 后处理   │
├─────────────────────────────────────────────────────┤
│  特征层: 运动 / 视觉击球 / 球轨迹 / 球员姿态             │
├─────────────────────────────────────────────────────┤
│  视觉层: OpenCV / YOLO / MediaPipe / 球检测跟踪        │
├─────────────────────────────────────────────────────┤
│  视频层: FFmpeg / 代理视频 / 元信息 / 裁剪拼接           │
└─────────────────────────────────────────────────────┘
```

---

## 设计原则

### 1. 分层渐进

- **Layer 1 (视频)**: FFmpeg 导入、代理、裁剪
- **Layer 2 (运动)**: 5fps 帧差扫描，提供粗粒度活跃信号
- **Layer 3 (视觉击球)**: YOLO 球员 + MediaPipe 挥拍 + 稀疏球确认（**当前主路径**）
- **Layer 4 (回合逻辑)**: lifecycle 推断、球轨迹 refine、球场几何
- **Layer 5 (ML)**: 专用动作/球检测 CNN 与小模型（进行中，见 `datasets/`）

### 2. 结构化优先

高维视频 → 低维结构化特征（hit 时间戳、轨迹点）→ 规则/轻量模型，而非直接把视频喂给大模型。

### 3. 本地优先

默认所有处理在本地完成；Layer1 使用自训 CNN，不依赖云端 VLM。

### 4. 偏召回

宁可多保留，不要误删好球。输出保留复核和重新导出能力。

---

## 数据流（当前主路径）

```text
input.mp4
  ├─→ video.ingest (元信息)
  ├─→ vision.motion (5fps 帧差 motion energy)
  ├─→ calibration.court (手动标定，推荐)
  └─→ hit_detection_visual.VisualHitDetector
         ├─ YOLO 球员跟踪
         ├─ MediaPipe 挥拍峰值
         └─ 稀疏球轨迹确认
         ↓
  segmentation.rules.segment_by_hit_events
         ↓
  segmentation.rally_lifecycle (回合起止推断)
         ↓
  segmentation.ball_refine (轨迹辅助修正，可选)
         ↓
  segmentation.postprocess (过滤、合并)
         ↓
  export.concat
         ↓
  timeline.json + hit_events.json + ball_trajectory.jsonl
```

### Legacy 音频路径（`--legacy-audio`，不推荐）

```text
audio.onset → features.fuse_hit_events(motion_peaks) → segment_by_hit_events
```

音频击球声在风噪/业余收音条件下误检率高，仅保留作对照实验。

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
| 运动检测 | `motion.py` | 已实现 |
| ROI | `roi.py` | 已实现 |
| 球员检测 | `players.py` | 已实现（YOLO） |
| 姿态估计 | `pose.py` | 已实现（MediaPipe） |
| 球检测/跟踪 | `ball.py`, `ball_track.py`, `ball_pipeline.py` | 已实现（CV，覆盖率有限） |
| 球场几何 | `court_lines.py` | 数据模型 + 加载；自动 Hough 已移除 |
| 标定 | `calibration/` | 手动点击标定（推荐） |

### 特征层 (`tenniscut/features/`)

| 模块 | 文件 | 职责 |
|---|---|---|
| 视觉击球 | `hit_detection_visual.py` | **主击球检测**（pose + ball） |
| 提取/融合 | `extract.py` | 每秒特征、回合事件融合 |
| Schema | `schema.py` | 特征数据结构 |

### 切分层 (`tenniscut/segmentation/`)

| 模块 | 文件 | 职责 |
|---|---|---|
| Active Score | `active_score.py` | 多信号加权（阈值 fallback） |
| 规则切分 | `rules.py` | 击球事件聚类 |
| 球事件 | `ball_rally.py` | 球轨迹事件分析（出界/触网等） |
| Lifecycle | `rally_lifecycle.py` | 回合起止推断（核心） |
| Ball refine | `ball_refine.py` | 轨迹辅助边界修正 |
| Refine | `refine.py` | 捡球/走动 trim |
| 后处理 | `postprocess.py` | 过滤短片段、合并 |

### ML 数据层 (`datasets/`, `tenniscut/ml/`)

| 模块 | 职责 |
|---|---|
| `corpus.py` | Clipper 视频库扫描、session registry |
| `scan_clipper_corpus.py` | CLI：生成 benchmark + registry |
| `datasets/schemas/` | 球员动作、球检测标注 schema |

### 评估层 (`tenniscut/benchmark/`)

| 模块 | 文件 | 职责 |
|---|---|---|
| 视觉对齐 | `align.py` | result 切分 + 原片帧指纹匹配 |
| CLI | `scripts/extract_benchmark.py` | 提取 benchmark |
| 评估 | `scripts/eval_baseline.py` | IoU / MAE 对比 |

---

## 已放弃的方案

| 方案 | 原因 | 处置 |
|---|---|---|
| 音频 onset 主击球检测 | 风噪/收音不稳定 | 保留 `--legacy-audio`，默认关闭 |
| 纯球轨迹击球检测 | 检测率 ~8%，轨迹断裂 | 删除 `hit_detection.py` |
| Hough 自动球场线 | 业余视频误检率高 | 删除自动检测，改 `calibrate-court` |
| 300–360s 一次性调试脚本 | 参数 sweep 已完成 | 已删除，用 `debug-ball` 替代 |

---

> 更多技术细节见各模块源码文档字符串及 [`datasets/README.md`](../datasets/README.md)。
