# 示例

此目录用于存放示例数据、配置文件和输出样例。

> ⚠️ **注意**：原始视频不上传至 GitHub。

---

## 预期内容

| 文件 | 说明 | 来源 |
|---|---|---|
| `sample_config.yaml` | 示例项目配置文件 | 手动创建 |
| `sample_timeline.json` | 示例输出时间轴 | 运行 `tenniscut process` 生成 |
| `sample_report.html` | 示例复核报告 | 运行 `tenniscut review` 生成 |
| `sample_project/` | 完整示例项目目录 | 运行 `tenniscut init` 生成 |

---

## 示例项目目录结构

```text
sample_project/
├── config.yaml              # 项目配置
├── input.mp4                # 原始视频 (不上传)
├── proxy.mp4                # 代理视频
├── work/
│   ├── features.jsonl       # 结构化特征
│   ├── hit_events.json      # 确认的击球事件
│   ├── timeline.json        # 自动切分结果
│   ├── timeline.csv         # 可编辑时间轴
│   └── audio.wav            # 中间音频文件
└── export/
    ├── trimmed_full_video.mp4
    └── clips/
        ├── segment_0001.mp4
        └── ...
```

---

> 示例数据将在 Phase 2 完成后添加。
