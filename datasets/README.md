# ML 数据集目录

本目录存放球员动作与球检测的训练/评估数据。**媒体文件（裁剪图、全画面、抽帧）与本地标注进度不上传 Git**，仅提交 schema、`*_unlabeled.jsonl` manifest 与 benchmark JSON。

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
│   ├── full_frame/            # 全画面抽帧（gitignore）
│   └── manifests/             # *_unlabeled.jsonl 入 Git；*_labeled.jsonl 本地
├── ball_detection/
│   ├── frames/                # 抽帧（gitignore）
│   ├── labels/                # 人工标注
│   ├── prelabels/             # CV 预标注
│   └── manifests/
└── eval/                      # VLM / 下游评估报告（gitignore，保留 .gitkeep）
```

### Git 跟踪说明

| 路径 | Git |
|------|-----|
| `schemas/*.json` | ✅ |
| `benchmarks/*.json` | ✅ |
| `manifests/*_unlabeled.jsonl` | ✅ |
| `manifests/*_labeled.jsonl` | ❌ 本地标注进度 |
| `eval/**` | ❌ 评估产物 |
| `raw_crops/**`、`full_frame/**` | ❌ 媒体 |

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

## M1：导出球员裁剪图

```bash
# 锚点段 303–354s（7252 test session）
python scripts/ml/export_player_crops.py \
  --session 7252 \
  --time-range 303 354 \
  --max-samples 300

# 全 split 批量导出（见计划 max-samples 表）
python scripts/ml/export_player_crops.py --session 7252 --max-samples 450
python scripts/ml/export_player_crops.py --session 7559 --max-samples 400
python scripts/ml/export_player_crops.py --session 7255 --max-samples 300
python scripts/ml/export_player_crops.py --session 7515 --max-samples 600
python scripts/ml/export_player_crops.py --session 7521 --max-samples 600
python scripts/ml/export_player_crops.py --session 7125_7126 --max-samples 800

# 全量重建 crop + 全画面（推荐，保证帧对齐）
python scripts/ml/export_player_crops.py --rebuild-all

# 合并 split manifest（--rebuild-all 已自动合并；单独合并用下面命令）
python scripts/ml/export_player_crops.py --merge-split train --merge-only
python scripts/ml/export_player_crops.py --merge-split val --merge-only
python scripts/ml/export_player_crops.py --merge-split test --merge-only
```

## M1：规则预填 + 高速交互标注（推荐）

```bash
# 预填 Layer 2 + QA（无需模型）；你主要标 Layer 1 action_state
python scripts/ml/prefill_annotation_defaults.py --all --relabel

python scripts/ml/annotate_player_actions.py \
  --manifest datasets/player_actions/manifests/train_unlabeled.jsonl \
  --serve --port 8765
# 浏览器: http://127.0.0.1:8765
```

**双层标签（v2）**：预填后主要填写 `action_state`；`rally_phase` / `label_confidence` / QA 已有默认值，可按需修改。

| Layer | 字段 | 标签 |
|-------|------|------|
| Layer 1 | `action_state` | `serving` · `hitting` · `moving` · `pick_ball` · `rest` · `unsure` |
| Layer 2 | `rally_phase` | `in_play` · `dead_time` · `unsure` |

**快捷键**：`1`–`6` 姿态 · `I`/`O` 回合内/外 · `9`–`5` 置信度 · `Space` 跳过 · `Z` 撤销 · `Shift+1`–`6` 批量标 track

标签实时写入 `manifests/{split}_labeled.jsonl`（无需手动导出）。旧 v1 单标签（`hit_rally` 等）已废弃，请重标。

合并多份标注文件：

```bash
python scripts/ml/import_labels.py \
  datasets/player_actions/manifests/train_labeled.jsonl \
  ~/Downloads/extra_labeled.jsonl \
  --output datasets/player_actions/manifests/train_labeled.jsonl --stats
```

## 标注 Sprint 建议顺序

1. Pilot 50 条：`7252_unlabeled.jsonl`，filter=in_rally
2. Train 500 条：`train_unlabeled.jsonl`
3. Test 150 条：`test_unlabeled.jsonl`（7252 为主，留作最终评估）
4. Val 100 条：`val_unlabeled.jsonl`

## M2：VLM / 小模型评估

```bash
# 1) 构建分层 VLM 测试集（50% dead_time，50% in_play；按 pose 子类分层）
python scripts/ml/build_vlm_eval_manifest.py --size 200

# 2) Qwen3-VL 零样本基线（player crop，主指标 pose+rally 双层一致）
python scripts/ml/eval_qwen_vl.py \
  --manifest datasets/player_actions/manifests/vlm_eval_stratified.jsonl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --task dual \
  --output-dir datasets/eval/qwen3_vl_2b

# 错判样本 HTML 图库（manifest 中的 full_frame 仅作 QA 对照）
python scripts/ml/build_vlm_error_gallery.py \
  --report datasets/eval/qwen3_vl_2b/qwen3_vl.json \
  --output-dir datasets/eval/qwen3_vl_2b

# 3) ResNet18 小模型
python scripts/ml/train_action_classifier.py \
  --train-manifest datasets/player_actions/manifests/train_labeled.jsonl \
  --output datasets/eval/resnet18_action_classifier.pt
```

VLM 输入：**player crop**（`crop_path`），每条样本对应一个 `track_id` 的球员裁切图。同一视频帧可有多条样本（不同球员各一条）；评估时必须按 `sample_id` / `track_id` 匹配，不要用 full_frame 里其他球员的动作推断当前 crop。

标注 UI 中的 `full_frame_path`（红框全场图）仅用于人工 QA，不参与 VLM 推理。

评估任务（`--task`）：
- `dual`（默认）：pose 与 rally_phase 同时正确
- `in_play_vs_dead`：回合二分类 F1
- `all_poses`：6 类姿态 exact match

模型输出 JSON：`{"action_state":"...","rally_phase":"...","confidence":0.0-1.0}`（解析仍兼容旧字段 `pose`）

## 评估指标（通过标准）

| 任务 | 目标 |
|------|------|
| 球员 **in_play vs dead_time** F1 | ≥ 0.80（VLM 基线参考） |
| 球员 **pose + rally 双层一致** accuracy | ≥ 0.60（VLM 基线参考） |
| 球 IoU@0.5 | ≥ 0.50 |
| 303–354s 轨迹帧覆盖率 | ≥ 40% |
