# 标注 Sprint 操作指南（v2 双层标签）

目标：train ≥500 · test ≥150 · val ≥100（含完整 **action_state + rally_phase + 置信度**）

## 标签体系

### Layer 1：动作状态 `action_state`（单帧快照，非完整挥拍时序）

| 标签 | 含义 |
|------|------|
| `serving` | 发球相关（含准备、抛球、击球、随挥、落地） |
| `hitting` | 对拉击球相关（含准备、挥拍、随挥） |
| `moving` | 移动/走位/碎步（本时刻未在发球或击球） |
| `pick_ball` | 弯腰/蹲下捡球 |
| `rest` | 休息、等待、缓慢走动 |
| `unsure` | 看不清 |

### Layer 2：回合状态 `rally_phase`

| 标签 | 含义 |
|------|------|
| `in_play` | 回合中（发球流程、对拉、接发球准备等） |
| `dead_time` | 回合外（局间、换边、休息、捡球等） |
| `unsure` | 不确定 |

### 附加字段

- `label_confidence`：标注者对 action_state + rally_phase 的置信度（100%/80%/60%/40%/20%）
- QA：`frame_align`（crop 与 full_frame 是否同一时刻）、`is_target_player`（是否场上目标球员）

> 旧 v1 标签（`hit_rally`、`hit_serve`、`idle` 等）已废弃，请全部按 v2 重标。

## 0. 规则预填（推荐，无需 VLM）

导出 crop 后，一键预填 Layer 2 + QA，你只需标 Layer 1 `action_state`：

```bash
# 全部 manifest（含 session 级 7252 等）
python scripts/ml/prefill_annotation_defaults.py --all --relabel

# 仅 train/val/test 合并 manifest
python scripts/ml/prefill_annotation_defaults.py --splits-only --relabel
```

默认预填值：

| 字段 | 值 |
|------|-----|
| `rally_phase` | `in_play`（`in_rally=true`）或 `dead_time`（`in_rally=false`） |
| `label_confidence` | `1.0`（100%） |
| `frame_align` | `same`（同一帧） |
| `is_target_player` | `yes`（场上球员） |
| `action_state` | `unsure`（留给你标注） |

写入 `*_labeled.jsonl`；`--relabel` 会清除旧 VLM 预标注并重置 Layer 1。

**重标前建议**：备份现有 `*_labeled.jsonl`，或直接用 `--relabel` 覆盖。

## 1. 启动标注服务

```bash
# Pilot：7252 锚点 session
python scripts/ml/annotate_player_actions.py \
  --manifest datasets/player_actions/manifests/7252_unlabeled.jsonl \
  --serve --port 8765

# 正式 train
python scripts/ml/annotate_player_actions.py \
  --manifest datasets/player_actions/manifests/train_unlabeled.jsonl \
  --serve --port 8765
```

浏览器打开 http://127.0.0.1:8765

界面为双栏：**YOLO crop** + **full_frame（红框）**。预填后主要选 **Layer 1 action_state**（`1`–`6`）；Layer 2 / 置信度 / QA 已默认填好，可按需改。

## 2. 推荐顺序

| 阶段 | Manifest | 数量 | Filter |
|------|----------|------|--------|
| Pilot | `7252_unlabeled.jsonl` | 50 | in_rally + near |
| Train | `train_unlabeled.jsonl` | 500 | in_rally 优先 |
| Train 负样本 | `train_unlabeled.jsonl` | 150 | 非 in_rally |
| Test | `test_unlabeled.jsonl` | 150 | 7252 为主 |
| Val | `val_unlabeled.jsonl` | 100 | — |

## 3. 快捷键

| 键 | 动作 |
|----|------|
| `1`–`6` | action_state：serving / hitting / moving / pick_ball / rest / unsure |
| `I` | 回合中 `in_play` |
| `O` | 回合外 `dead_time` |
| `9`/`8`/`7`/`6`/`5` | 置信度 100% / 80% / 60% / 40% / 20%（选置信度后自动跳下一条） |
| `Space` | 跳过，跳下一条未完成 |
| `Z` | 撤销上一条 |
| `Shift+1`–`6` | 当前标签批量应用到 track 后续 5 帧 |
| `←` / `→` | 上一条 / 下一条 |

## 4. 进度查看

标注自动保存到同 split 的 `*_labeled.jsonl`。查看统计：

```bash
python scripts/ml/import_labels.py \
  datasets/player_actions/manifests/train_labeled.jsonl \
  --output /tmp/stats.jsonl --stats
```

## 5. 达标后

```bash
# 构建 50/50 分层 VLM 测试集（dead_time vs in_play，200 条）
python scripts/ml/build_vlm_eval_manifest.py --size 200

# Qwen3-VL 零样本基线：crop vs full_frame，主指标 pose+rally 双层一致
python scripts/ml/eval_qwen_vl.py \
  --manifest datasets/player_actions/manifests/vlm_eval_stratified.jsonl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --task dual \
  --compare-all \
  --output-dir datasets/eval/qwen3_vl_2b

# 错判图库
python scripts/ml/build_vlm_error_gallery.py \
  --report-crop datasets/eval/qwen3_vl_2b/qwen3_vl_crop.json \
  --report-full datasets/eval/qwen3_vl_2b/qwen3_vl_full_frame.json \
  --output-dir datasets/eval/qwen3_vl_2b

python scripts/ml/train_action_classifier.py \
  --train-manifest datasets/player_actions/manifests/train_labeled.jsonl \
  --val-manifest datasets/player_actions/manifests/val_labeled.jsonl \
  --output datasets/eval/resnet18_action_classifier.pt
```
