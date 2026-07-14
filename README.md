# Tennis Rally Clipper

> **本地优先的开源网球长视频自动回合切分工具。**
>
> 把 1–2 小时固定机位的打球视频，自动剪掉捡球和等待，保留每一回合，并导出结构化时间轴。

---

## 一句话介绍

Tennis Rally Clipper 专注做一件事：把**固定机位网球长视频 → 切分成一个个回合 → 删除无效时间 → 按需求导出**。

- **输入**：1–2 小时、固定背后视角的打球视频（mp4 / mov）
- **中间**：自动识别每一分的开始与结束，输出 timeline.json / timeline.csv
- **输出**：剪掉无效片段后的完整视频、每个回合单独片段、可人工复核的时间轴

**目标：让你打球 1 小时，剪辑只需几分钟。**

---

## 为什么需要这个工具？

| 痛点 | 现状 |
|---|---|
| 每次打球都有 1–2 小时长视频 | 手动剪辑极其耗时，几乎“打多久，剪多久” |
| 现有自动剪辑工具面向 Vlog | 不支持网球场景，更不支持“按回合筛选” |
| 球友想发小红书/B站/抖音 | 需要从海量素材中挑出好球、多拍、失误 |
| 教练/学员需要回看 | 没有结构化时间轴，定位特定回合困难 |

---

## 当前技术路线

当前主路径为 **视觉击球检测 + 回合生命周期推断**，已验证音频方案和纯球轨迹方案在业余固定机位场景下不够可靠。

```text
原始视频
  ├─→ 5fps 运动扫描 (OpenCV 帧差) → 每秒 motion energy
  └─→ 视觉击球通道 (默认开启)
           ├─ YOLO 球员框 + MediaPipe 挥拍峰值 (主信号)
           ├─ 稀疏球轨迹确认 (辅信号)
           └─ 可选球场线标定 (calibrate-court)
           ↓
  击球事件聚类 → rally_lifecycle / ball_refine 边界修正
           ↓
  按 --min-rally 过滤 → FFmpeg 裁剪原视频并拼接
```

> 旧版「音频 + 运动融合」仍可通过 `--legacy-audio` 启用，但不推荐。

### 关键设计

| 设计 | 说明 |
|---|---|
| **本地优先** | 视频默认本地处理，不强制上传 |
| **原视频导出** | 分析可用 proxy，导出用原视频，保留画质 |
| **偏召回** | 宁可多保留，不切断正在进行的回合 |
| **可解释** | 输出 timeline.json / hit_events.json，便于复核 |
| **视觉主信号** | 挥拍动作比击球声/纯球轨迹更稳定 |
| **人工标定球场** | `tenniscut calibrate-court` 替代不可靠的自动线检测 |

---

## 快速开始

### 环境要求

- **操作系统**: macOS 10.15+ / Linux / Windows 10+
- **Python**: 3.10 或更高版本
- **内存**: 建议 16GB
- **FFmpeg**: 必须预先安装（macOS 可下载静态二进制文件放入 `~/.local/bin`）

### 安装

```bash
# 1. 安装 uv（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/AohuaCheng/TennisClipper.git
cd TennisClipper

# 3. 创建虚拟环境并安装依赖
uv pip install -r requirements.txt
source .venv/bin/activate

# 4. 验证 FFmpeg
ffmpeg -version
```

### 一条命令跑起来

```bash
# 1. 初始化项目
tenniscut init my_session

# 2. 导入视频
tenniscut add my_session input.mp4

# 3. 一键处理（默认视觉击球检测）
tenniscut process my_session

# 可选：标定球场线（推荐，提升出界/触网判断）
tenniscut calibrate-court my_session --time 330

# 测试长片段验证（只看 >= 20s 的回合）
tenniscut process my_session --min-rally 20
```

### 输出结构

```text
my_session/
├── work/
│   ├── features.jsonl      # 每秒运动特征
│   ├── hit_events.json     # 确认的击球事件
│   ├── ball_trajectory.jsonl  # 球轨迹（视觉路径）
│   ├── timeline.json       # 结构化时间轴
│   └── timeline.csv        # 可编辑时间轴
└── export/
    ├── trimmed_full_video.mp4  # 拼接后的完整视频
    └── clips/                  # 单独片段（导出 clips 模式时）
```

---

## 当前项目进度

### 已实现

| 模块 | 文件 | 状态 |
|---|---|---|
| CLI 入口 | `tenniscut/cli.py` | init / add / process / export / calibrate-court / debug-ball |
| 视觉击球 | `tenniscut/features/hit_detection_visual.py` | YOLO + MediaPipe 挥拍检测 |
| 球轨迹 | `tenniscut/vision/ball*.py` | 颜色+运动联合检测、轨迹跟踪 |
| 回合边界 | `tenniscut/segmentation/rally_lifecycle.py` | 击球间隔 + 回合结束推断 |
| 球轨迹 refine | `tenniscut/segmentation/ball_refine.py` | 轨迹辅助边界修正 |
| 球场标定 | `tenniscut/calibration/` | 点击标定球场线 |
| Benchmark | `tenniscut/benchmark/align.py` | 人工剪辑 → 原片对齐 |
| ML 数据集 | `datasets/`, `scripts/ml/` | session registry + benchmark 脚手架 |
| 评估 | `scripts/eval_baseline.py` | IoU / MAE 对比 |

### 进行中 / 下一步

| 方向 | 说明 |
|---|---|
| ML 感知层 | 球员动作分类 + 球检测专用模型（见 `datasets/README.md`） |
| 复核 UI | HTML 标注与 timeline 编辑 |
| 自动球场线 | 已放弃 Hough 自动检测，改用手动标定 |

### 测试验证（IMG_7252，42 分钟）

| 指标 | 当前自动切分 | 人工剪辑 |
|---|---|---|
| 原始时长 | 2558.1s | 2558.1s |
| 303–354s 锚点段 | 仍易过度切分（3 段 vs 1 段） | 1 段 |
| 根因 | 击球/轨迹基础识别不足 | — |

完整 benchmark 评估见下方「Benchmark 评估」章节。

---

## Benchmark 评估（人工剪辑 → 原片对齐）

当你已经有一份**人工剪辑好的集锦视频**（`result.mp4`）和对应的**原始长视频**（`raw.MOV`）时，可以用视觉帧匹配自动反推每段在原片中的起止时间，作为评估 ground truth。

### 工作原理

```text
result.mp4（人工剪辑集锦）
  ├─→ Step 1: 切分 result（自动硬切检测 或 手动 --result-cuts）
  └─→ Step 2: 每段视觉匹配到 original（dHash + HSV，全自动）
           ↓
  benchmark.json（original_start / original_end）
           ↓
  eval_baseline.py 对比 tenniscut 输出的 timeline.json
```

| 步骤 | 是否需人工 | 说明 |
|---|---|---|
| 切分 result | **视情况** | 默认自动检测硬切；网球场景易过切，建议用 `--result-cuts` 指定切点 |
| 对齐 original | 否 | 每段取 8 帧 probe，在原片索引中滑动匹配，带单调约束 |

### 提取 benchmark

```bash
python scripts/extract_benchmark.py \
  --original /path/to/IMG_7252_raw.MOV \
  --result   /path/to/IMG_7252_result.mp4 \
  --output   sessions/test_session_7252/benchmark_7252.json \
  --result-cuts 51 70 139 213 267 \
  --index-cache sessions/test_session_7252/work/original_frame_index.pkl
```

### 对比自动切分结果

```bash
python scripts/eval_baseline.py \
  --predicted sessions/test_session_7252/work/timeline.json \
  --ground-truth sessions/test_session_7252/benchmark_7252.json \
  --output sessions/test_session_7252/eval_report.json
```

### ML 数据集初始化

```bash
python scripts/ml/scan_clipper_corpus.py --skip-benchmarks
```

详见 [`datasets/README.md`](datasets/README.md)。

---

## 开发阶段

| 阶段 | 目标 | 状态 |
|---|---|---|
| Phase 1 | 最小可运行管线：输入视频 → 自动切分 → 导出 | 已完成 |
| Phase 2 | ROI、复核、球场标定 | 部分完成（标定已实现） |
| Phase 3 | 视觉击球：球员/姿态/球轨迹 | 已实现原型，精度待提升 |
| Phase 3.5 | 回合 lifecycle + ball refine | 已实现，待 ML 感知层加强 |
| Phase 4 | ML 数据集 + 专用模型 + VLM 基线 | 进行中 |
| Phase 5 | 回合标签筛选 + LLM 精筛 | 待开始 |

详细路线图见 [`docs/ROADMAP.md`](./docs/ROADMAP.md)。

---

## 技术栈

| 领域 | 组件 |
|---|---|
| 视频处理 | FFmpeg, OpenCV |
| 视觉检测 | Ultralytics YOLO, MediaPipe |
| 评估 | 视觉帧对齐 benchmark |
| 导出 | FFmpeg concat / clips / EDL |

---

## 已知限制

1. **固定机位**：优先支持固定背后视角，手持或剧烈晃动视频效果会下降
2. **击球识别**：挥拍峰值会把转身等非击球判为 hit；需 ML 动作分类改进
3. **球轨迹稀疏**：CV 球检测在 4K 业余视频中覆盖率低，不能单独支撑回合切分
4. **球场线**：需手动 `calibrate-court`；自动 Hough 检测已移除
5. **非实时**：面向离线长视频，不处理直播流

---

## 贡献与隐私

- 欢迎提交 Issue 反馈误检、漏检、切断回合等问题
- 不默认公开任何球友的原始视频，详见 [`docs/DATA_PRIVACY.md`](./docs/DATA_PRIVACY.md)
- 标注标准见 [`docs/ANNOTATION_GUIDE.md`](./docs/ANNOTATION_GUIDE.md)

---

> **Make long tennis practice videos watchable in minutes.**
>
> 把一小时网球长视频，快速变成只保留回合的可观看版本。
