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

当前实现已从「运动能量阈值切分」升级为 **「音频 + 运动融合的击球事件聚类」**，以解决“对拉中途被切断”的核心问题。

```text
原始视频
  ├─→ 5fps 运动扫描 (OpenCV 帧差) → 每秒 motion energy
  └─→ FFmpeg 提取音频 (22.05kHz 单声道 WAV)
           ↓
  500-3000Hz 带通滤波 → RMS 能量 → 自适应阈值 → 击球 onset
           ↓
  运动峰值 + 音频 onset 时间对齐 → 确认真实击球事件
           ↓
  相邻击球事件 > 6s 拆分 → 一个回合
  start = 首击球 - 2s, end = 末击球 + 0.5s
           ↓
  按 --min-rally 过滤 → FFmpeg 裁剪原视频并拼接
```

### 关键设计

| 设计 | 说明 |
|---|---|
| **本地优先** | 视频默认本地处理，不强制上传 |
| **原视频导出** | 分析可用 proxy，导出用原视频，保留画质 |
| **偏召回** | 宁可多保留，不切断正在进行的回合 |
| **可解释** | 输出 timeline.json / timeline.csv / hit_events.json，便于复核 |
| **音频融合** | 用击球声确认运动峰值，减少误检和漏切 |

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

# 3. 一键处理（默认最小片段 5s）
tenniscut process my_session

# 测试长片段验证（只看 >= 20s 的回合）
tenniscut process my_session --min-rally 20
```

### 输出结构

```text
my_session/
├── work/
│   ├── features.jsonl      # 每秒运动特征
│   ├── hit_events.json     # 确认的击球事件
│   ├── timeline.json       # 结构化时间轴
│   ├── timeline.csv        # 可编辑时间轴
│   └── audio.wav           # 提取的音频（中间文件）
└── export/
    ├── trimmed_full_video.mp4  # 拼接后的完整视频
    └── clips/                  # 单独片段（导出 clips 模式时）
```

---

## 当前项目进度

### 已实现 ✅

| 模块 | 文件 | 状态 |
|---|---|---|
| CLI 入口 | `tenniscut/cli.py` | ✅ 支持 `init`, `add`, `proxy`, `scan`, `segment`, `process`, `export`, `review` |
| 配置管理 | `tenniscut/config.py` | ✅ YAML 配置读写 |
| 视频导入 | `tenniscut/video/ingest.py` | ✅ OpenCV 读取元信息与帧 |
| 代理视频 | `tenniscut/video/proxy.py` | ✅ FFmpeg 540p，支持 macOS VideoToolbox 硬件编码 |
| FFmpeg 封装 | `tenniscut/video/ffmpeg.py` | ✅ 裁剪、拼接、音频提取 |
| 运动检测 | `tenniscut/vision/motion.py` | ✅ 帧差 + 显著变化像素比例 |
| 特征提取 | `tenniscut/features/extract.py` | ✅ 每秒特征 + 音频/运动融合函数 |
| 音频处理 | `tenniscut/audio/onset.py` | ✅ 击球声检测 |
| 切分规则 | `tenniscut/segmentation/rules.py` | ✅ 阈值切分 + 击球事件聚类 |
| 后处理 | `tenniscut/segmentation/postprocess.py` | ✅ 前后余量、过滤、合并 |
| 导出 | `tenniscut/export/concat.py` | ✅ 拼接导出 |
| 复核 | `tenniscut/review/` | ✅ 基础 CLI review（HTML 报告待完善） |
| 时间轴 Schema | `schemas/timeline.schema.json` | ✅ 已定义 |
| 特征 Schema | `schemas/features.schema.json` | ✅ 已定义 |
| 测试 | `tests/` | ✅ 基础单元测试 |

### 测试验证

| 指标 | 当前结果（IMG_7252_raw，42 分钟） | 人工剪辑 |
|---|---|---|
| 原始时长 | 2558.1s | 2558.1s |
| 输出时长（≥20s 片段） | 213.9s | 332.1s |
| 检出 ≥20s 回合数 | 9 | — |
| 检出覆盖率 | 约 64% | 基准 |

---

## 开发阶段

| 阶段 | 目标 | 状态 |
|---|---|---|
| Phase 1 | 最小可运行管线：输入视频 → 自动切分 → 导出 | ✅ 已完成（含音频融合） |
| Phase 2 | ROI 与复核机制：提升可用性，降低误检 | ⏳ 待开始 |
| Phase 3 | 球员检测与姿态特征：让切分不只依赖运动 | ⏳ 待开始 |
| Phase 4 | 回合标签与筛选：long_rally / highlight 等 | ⏳ 待开始 |
| Phase 5 | 自然语言筛选与 AI 精筛 | ⏳ 待开始 |

详细路线图见 [`docs/ROADMAP.md`](./docs/ROADMAP.md)。

---

## 技术栈

| 领域 | 组件 |
|---|---|
| 视频处理 | FFmpeg, OpenCV |
| 基础视觉 | NumPy, Pandas |
| 音频处理 | SciPy |
| 检测与姿态（可选） | Ultralytics YOLO, MediaPipe（Phase 3） |
| 导出 | FFmpeg concat / clips / EDL |

---

## 已知限制

1. **固定机位**：优先支持固定背后视角，手持或剧烈晃动视频效果会下降
2. **下手发球**：挥拍动作小、声音小，开始击球可能漏检
3. **非击球结束**：球落地/对方空挥时可能没有声音，结束边界可能偏晚
4. **音频质量**：风噪、环境声会干扰击球声检测
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
