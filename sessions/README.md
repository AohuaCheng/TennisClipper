# 本地测试会话

此目录用于存放本地测试产生的会话目录，例如 `sessions/test_session_7252/`。

这些目录包含生成的中间文件（`work/`）和导出视频（`export/`），不会被提交到 Git。

## 用法

```bash
# 在项目根目录下创建会话
mkdir -p sessions
tenniscut init sessions/my_session
tenniscut add sessions/my_session /path/to/video.mp4
tenniscut process sessions/my_session --min-rally 20
```
