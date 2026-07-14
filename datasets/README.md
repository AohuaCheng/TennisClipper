# ML 数据集目录

本目录存放球员动作与球检测的训练/评估数据。**媒体文件（裁剪图、抽帧）不上传 Git**，仅提交 schema、manifest 与 benchmark JSON。

## 目录结构

```text
datasets/
├── README.md
├── sessions_metadata.csv      # 人工维护：场地/单双打元数据
├── sessions_registry.json     # 自动生成：视频路径 + split
├── benchmarks/                # 各 session 剪辑对齐 benchmark
├── schemas/                   # JSON Schema
├── player_actions/
│   ├── raw_crops/             # YOLO 裁剪图（gitignore）
│   └── manifests/             # train/val/test jsonl
├── ball_detection/
│   ├── frames/                # 抽帧（gitignore）
│   ├── labels/                # 人工标注
│   ├── prelabels/             # CV 预标注
│   └── manifests/
└── eval/                      # VLM / 下游评估报告
```

## Split 规则

**按 session 切分，绝不随机切帧**（见 [docs/ANNOTATION_GUIDE.md](../docs/ANNOTATION_GUIDE.md)）。

| Split | Sessions |
|-------|----------|
| test | `7252`, `7559` |
| val | `7255` |
| train | `7125_7126`, `7515`, `7521` |

`7252` 为 303–354s 下游评估锚点。

## 元数据字段

| 字段 | 说明 |
|------|------|
| `court_id` | 场地标识 |
| `court_type` | `indoor_hard` / `outdoor_hard` / `outdoor_clay` |
| `match_type` | `singles` / `doubles` |
| `split` | `train` / `val` / `test`（由 registry 脚本注入） |

## 初始化

```bash
python scripts/ml/scan_clipper_corpus.py
```

可选参数：

```bash
python scripts/ml/scan_clipper_corpus.py \
  --clipper-dir /Users/aohuacheng/Downloads/Clipper \
  --skip-benchmarks          # 仅生成 registry，不对齐 benchmark
```

## 评估指标（通过标准）

| 任务 | 目标 |
|------|------|
| 球员 `hit_rally` vs `move` F1 | ≥ 0.80（test） |
| 球 IoU@0.5 | ≥ 0.50 |
| 303–354s 轨迹帧覆盖率 | ≥ 40% |
