"""Interactive HTTP server for player action labeling."""
from __future__ import annotations

import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from tenniscut.ml.corpus import load_registry
from tenniscut.ml.frame_io import render_full_frame_jpg
from tenniscut.ml.manifest_io import LabelStore, load_jsonl

from tenniscut.ml.labels import (
    ACTION_STATE_DISPLAY,
    ACTION_STATE_LABELS,
    CONFIDENCE_PRESETS,
    RALLY_PHASE_DISPLAY,
    RALLY_PHASE_LABELS,
)

ACTION_STATE_UI: List[Tuple[str, str, str]] = [
    (state, str(i + 1), ACTION_STATE_DISPLAY[state])
    for i, state in enumerate(ACTION_STATE_LABELS)
]
# backward compat alias
POSE_UI = ACTION_STATE_UI
RALLY_UI: List[Tuple[str, str, str]] = [
    ("in_play", "I", RALLY_PHASE_DISPLAY["in_play"]),
    ("dead_time", "O", RALLY_PHASE_DISPLAY["dead_time"]),
    ("unsure", "U", RALLY_PHASE_DISPLAY["unsure"]),
]
CONFIDENCE_UI: List[Tuple[float, str, str]] = [
    (1.0, "9", "100%"),
    (0.8, "8", "80%"),
    (0.6, "7", "60%"),
    (0.4, "6", "40%"),
    (0.2, "5", "20%"),
]

INTERACTIVE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>球员动作标注</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #0f0f0f; color: #eee; }
    header { padding: 10px 14px; background: #1a1a1a; border-bottom: 1px solid #333; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    main { display: grid; grid-template-columns: 1fr 360px; gap: 12px; padding: 12px; min-height: calc(100vh - 120px); }
    .left { display: flex; flex-direction: column; gap: 10px; }
    #viewer { background: #000; border: 1px solid #333; border-radius: 8px; min-height: 360px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; }
    .viewPane { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 280px; }
    .viewPane img { max-width: 100%; max-height: 38vh; object-fit: contain; border: 1px solid #333; border-radius: 4px; }
    .viewLabel { font-size: 12px; color: #888; margin-bottom: 4px; }
    #videoWrap { background: #000; border: 1px solid #333; border-radius: 8px; padding: 8px; }
    #ctxVideo { width: 100%; max-height: 28vh; background: #000; }
    .panel { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px; overflow-y: auto; max-height: calc(100vh - 100px); }
    .meta { font-size: 13px; line-height: 1.55; color: #bbb; margin-bottom: 8px; }
    .labels { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    button, .chip { background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 6px; padding: 7px 9px; cursor: pointer; font-size: 13px; }
    button:hover { background: #333; }
    button.active { background: #0a5; border-color: #0c6; }
    .chip { display: inline-block; margin: 2px 4px 2px 0; }
    .chip.on { background: #035; border-color: #09f; }
    .progress { font-weight: 600; }
    .hint { color: #888; font-size: 12px; }
    #trackStrip { display: flex; gap: 6px; overflow-x: auto; padding: 6px 0; }
    #trackStrip img { height: 72px; border: 2px solid transparent; border-radius: 4px; cursor: pointer; }
    #trackStrip img.sel { border-color: #0af; }
    #trackStrip img.labeled { border-color: #0a5; }
    footer { padding: 10px 14px; border-top: 1px solid #333; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .toggle { accent-color: #0a5; }
    .stat { font-size: 12px; color: #9aa; }
    .qaGroup { margin-top: 12px; padding-top: 8px; border-top: 1px solid #333; }
    .qaRow { margin-bottom: 8px; }
    .qaBtns { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }
    .qaBtns button.sel { background: #035; border-color: #09f; }
  </style>
</head>
<body>
  <header>
    <span class="progress" id="progress">加载中…</span>
    <span id="sampleId" class="hint"></span>
    <span class="hint">1-6 姿态 · I/O/U 回合 · 5-9 置信度 · Space 跳过 · Z 撤销</span>
  </header>
  <main>
    <section class="left">
      <div id="viewer"><span>加载中…</span></div>
      <div id="videoWrap">
        <div class="hint">原视频上下文 (±1s)</div>
        <video id="ctxVideo" muted playsinline preload="metadata"></video>
      </div>
      <div>
        <div class="hint">同 track 上下文</div>
        <div id="trackStrip"></div>
      </div>
    </section>
    <aside class="panel">
      <div class="meta" id="meta"></div>
      <div style="margin-bottom:8px">
        <span class="hint">过滤:</span>
        <span class="chip on" data-filter="all">全部</span>
        <span class="chip" data-filter="unlabeled">未标注</span>
        <span class="chip" data-filter="in_rally">in_rally</span>
        <span class="chip" data-filter="near">near</span>
        <span class="chip" data-filter="far">far</span>
        <span class="chip" data-filter="qa_unset">QA未填</span>
        <span class="chip" data-filter="invalid_player">非球员</span>
        <span class="chip" data-filter="frame_mismatch">帧不一致</span>
      </div>
      <label class="hint"><input type="checkbox" id="autoAdvance" class="toggle" checked /> 标注完成后自动跳下一条</label>
      <div class="hint" style="margin-top:8px"><b>Layer 1 · action_state</b>（动作状态）</div>
      <div class="labels" id="actionStateButtons"></div>
      <div class="hint" style="margin-top:8px"><b>Layer 2 · rally_phase</b>（回合阶段）</div>
      <div class="labels" id="rallyButtons"></div>
      <div class="hint" style="margin-top:8px"><b>标注置信度</b></div>
      <div class="qaBtns" id="confidenceBtns"></div>
      <div class="qaGroup">
        <div class="qaRow">
          <div class="hint">crop 与全帧是否同一时刻？</div>
          <div class="qaBtns" id="frameAlignBtns"></div>
        </div>
        <div class="qaRow">
          <div class="hint">框定是否为场上目标球员？</div>
          <div class="qaBtns" id="targetPlayerBtns"></div>
        </div>
      </div>
      <div style="margin-top:10px">
        <button id="bulkTrackBtn">批量: 当前标注 → 该 track 后续 5 帧</button>
      </div>
      <div class="stat" id="stats" style="margin-top:12px"></div>
    </aside>
  </main>
  <footer>
    <button id="prevBtn">← 上一条</button>
    <button id="nextBtn">下一条 →</button>
    <button id="unlabeledBtn">未标注 (U)</button>
    <button id="undoBtn">撤销 (Z)</button>
  </footer>
  <script>
    const UI = __UI_CONFIG_JSON__;
    let allSamples = [];
    let queue = [];
    let idx = 0;
    let ann = {};
    let history = [];
    let filterMode = "all";
    let videoMap = {};

    async function api(path, opts) {
      const r = await fetch(path, opts);
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }

    function getAnn(s) {
      return ann[s.sample_id] || {
        action_state: s.action_state || "unsure",
        rally_phase: s.rally_phase || "unsure",
        label_confidence: s.label_confidence ?? null,
        frame_align: s.frame_align || null,
        is_target_player: s.is_target_player || null,
      };
    }

    function isComplete(s) {
      const a = getAnn(s);
      return a.action_state && a.action_state !== "unsure" && a.rally_phase && a.rally_phase !== "unsure"
        && a.label_confidence != null
        && a.frame_align && a.frame_align !== "unsure"
        && a.is_target_player && a.is_target_player !== "unsure";
    }

    function applyFilter() {
      queue = allSamples.filter(s => {
        if (filterMode === "unlabeled") return !isComplete(s);
        if (filterMode === "in_rally") return s.in_rally === true;
        if (filterMode === "near") return s.role === "near";
        if (filterMode === "far") return s.role === "far";
        const a = getAnn(s);
        if (filterMode === "qa_unset") return !a.frame_align || !a.is_target_player;
        if (filterMode === "invalid_player") return a.is_target_player === "no";
        if (filterMode === "frame_mismatch") return a.frame_align === "different";
        return true;
      });
      if (!queue.length) queue = allSamples.slice();
      idx = Math.min(idx, Math.max(0, queue.length - 1));
    }

    function current() { return queue[idx]; }

    function labeledCount() {
      return allSamples.filter(s => isComplete(s)).length;
    }

    function syncVideo(s) {
      const vid = document.getElementById("ctxVideo");
      const src = videoMap[s.session_id];
      if (!src) { vid.removeAttribute("src"); return; }
      if (vid.dataset.session !== s.session_id) {
        vid.dataset.session = s.session_id;
        vid.src = src;
      }
      const start = Math.max(0, s.t - 1.0);
      const seek = () => { vid.currentTime = start; vid.play().catch(() => {}); };
      if (vid.readyState >= 1) seek();
      else vid.onloadedmetadata = seek;
    }

    function renderTrackStrip(s) {
      const strip = document.getElementById("trackStrip");
      const same = allSamples.filter(x => x.session_id === s.session_id && x.track_id === s.track_id)
        .sort((a,b) => a.t - b.t);
      const pos = same.findIndex(x => x.sample_id === s.sample_id);
      const window = same.slice(Math.max(0, pos - 3), pos + 4);
      strip.innerHTML = "";
      window.forEach(item => {
        const img = document.createElement("img");
        img.src = "/crop/" + encodeURIComponent(item.crop_path);
        img.title = item.t + "s";
        if (item.sample_id === s.sample_id) img.classList.add("sel");
        if (isComplete(item)) img.classList.add("labeled");
        img.onclick = () => {
          const qidx = queue.findIndex(q => q.sample_id === item.sample_id);
          if (qidx >= 0) { idx = qidx; render(); }
        };
        strip.appendChild(img);
      });
    }

    function renderSelectionButtons() {
      const s = current();
      const a = getAnn(s);
      document.querySelectorAll("#actionStateButtons button").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.actionState === a.action_state);
      });
      document.querySelectorAll("#rallyButtons button").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.rally === a.rally_phase);
      });
      document.querySelectorAll("#confidenceBtns button").forEach(btn => {
        btn.classList.toggle("sel", parseFloat(btn.dataset.conf) === a.label_confidence);
      });
      document.querySelectorAll("#frameAlignBtns button").forEach(btn => {
        btn.classList.toggle("sel", btn.dataset.val === a.frame_align);
      });
      document.querySelectorAll("#targetPlayerBtns button").forEach(btn => {
        btn.classList.toggle("sel", btn.dataset.val === a.is_target_player);
      });
    }

    function render() {
      if (!queue.length) return;
      const s = current();
      const a = getAnn(s);
      document.getElementById("progress").textContent =
        `${idx + 1}/${queue.length} · 总 ${allSamples.length} · 已完成 ${labeledCount()}`;
      document.getElementById("sampleId").textContent = s.sample_id;
      document.getElementById("viewer").innerHTML =
        `<div class="viewPane"><div class="viewLabel">YOLO crop</div>` +
        `<img src="/crop/${encodeURIComponent(s.crop_path)}" alt="crop"></div>` +
        `<div class="viewPane"><div class="viewLabel">全帧 @ t=${s.t}s (bbox)</div>` +
        `<img src="/api/frame/${encodeURIComponent(s.sample_id)}" alt="full frame"></div>`;
      document.getElementById("meta").innerHTML =
        `session: <b>${s.session_id}</b> · t: <b>${s.t}s</b><br>` +
        `track: <b>${s.track_id}</b> · role: ${s.role} · benchmark in_rally: ${s.in_rally}<br>` +
        `segment: ${s.segment_id || "-"}<br>` +
        `action_state: <b>${a.action_state}</b> · rally_phase: <b>${a.rally_phase}</b> · 置信: <b>${a.label_confidence ?? "-"}</b><br>` +
        `QA: frame=${a.frame_align || "-"} player=${a.is_target_player || "-"}`;
      renderSelectionButtons();
      syncVideo(s);
      renderTrackStrip(s);
    }

    async function refreshStats() {
      const st = await api("/api/stats");
      document.getElementById("stats").textContent =
        `已完成 ${st.labeled}/${st.total} · action ${JSON.stringify(st.action_state_counts || st.pose_counts)} · phase ${JSON.stringify(st.rally_phase_counts)}`;
    }

    async function persistAnnotation(sampleId, payload, { advance = false, bulkTrack = false } = {}) {
      history.push({ sample_id: sampleId, prev: { ...getAnn({ sample_id: sampleId }) } });
      await api("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId, ...payload }),
      });
      ann[sampleId] = { ...getAnn({ sample_id: sampleId }), ...payload };
      if (bulkTrack) {
        const s = current();
        const same = allSamples.filter(x =>
          x.session_id === s.session_id && x.track_id === s.track_id && x.t >= s.t
        ).sort((a,b) => a.t - b.t).slice(0, 6);
        const ids = same.map(x => x.sample_id);
        await api("/api/label/bulk", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sample_ids: ids, ...payload }),
        });
        ids.forEach(id => { ann[id] = { ...getAnn({ sample_id: id }), ...payload }; });
      }
      await refreshStats();
      render();
      if (advance && document.getElementById("autoAdvance").checked && isComplete({ sample_id: sampleId })) {
        goUnlabeled();
      }
    }

    async function setField(field, value, opts = {}) {
      const s = current();
      const cur = getAnn(s);
      const payload = {
        action_state: cur.action_state,
        rally_phase: cur.rally_phase,
        label_confidence: cur.label_confidence,
        frame_align: cur.frame_align,
        is_target_player: cur.is_target_player,
      };
      payload[field] = value;
      await persistAnnotation(s.sample_id, payload, opts);
    }

    function go(delta) {
      idx = Math.max(0, Math.min(queue.length - 1, idx + delta));
      render();
    }

    function goUnlabeled() {
      for (let i = 1; i <= queue.length; i++) {
        const j = (idx + i) % queue.length;
        if (!isComplete(queue[j])) { idx = j; render(); return; }
      }
      go(1);
    }

    async function undo() {
      const last = history.pop();
      if (!last) return;
      await api("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: last.sample_id, ...last.prev }),
      });
      ann[last.sample_id] = last.prev;
      await refreshStats();
      render();
    }

    function buildUI() {
      const actionBox = document.getElementById("actionStateButtons");
      UI.action_state.forEach(([id, key, title]) => {
        const b = document.createElement("button");
        b.dataset.actionState = id;
        b.textContent = `${key}: ${title}`;
        b.onclick = () => setField("action_state", id, { advance: false });
        actionBox.appendChild(b);
      });
      const rallyBox = document.getElementById("rallyButtons");
      UI.rally.forEach(([id, key, title]) => {
        const b = document.createElement("button");
        b.dataset.rally = id;
        b.textContent = `${key}: ${title}`;
        b.onclick = () => setField("rally_phase", id, { advance: false });
        rallyBox.appendChild(b);
      });
      const confBox = document.getElementById("confidenceBtns");
      UI.confidence.forEach(([val, key, title]) => {
        const b = document.createElement("button");
        b.dataset.conf = String(val);
        b.textContent = `${key}: ${title}`;
        b.onclick = () => setField("label_confidence", val, { advance: true });
        confBox.appendChild(b);
      });
      const frameOpts = [["same","同一帧"],["different","不一致"],["unsure","不确定"]];
      const playerOpts = [["yes","是球员"],["no","非球员"],["unsure","不确定"]];
      frameOpts.forEach(([val, title]) => {
        const b = document.createElement("button");
        b.dataset.val = val;
        b.textContent = title;
        b.onclick = () => setField("frame_align", val);
        document.getElementById("frameAlignBtns").appendChild(b);
      });
      playerOpts.forEach(([val, title]) => {
        const b = document.createElement("button");
        b.dataset.val = val;
        b.textContent = title;
        b.onclick = () => setField("is_target_player", val);
        document.getElementById("targetPlayerBtns").appendChild(b);
      });
      document.querySelectorAll(".chip[data-filter]").forEach(chip => {
        chip.onclick = () => {
          document.querySelectorAll(".chip[data-filter]").forEach(c => c.classList.remove("on"));
          chip.classList.add("on");
          filterMode = chip.dataset.filter;
          applyFilter();
          render();
        };
      });
      document.getElementById("prevBtn").onclick = () => go(-1);
      document.getElementById("nextBtn").onclick = () => go(1);
      document.getElementById("unlabeledBtn").onclick = goUnlabeled;
      document.getElementById("undoBtn").onclick = undo;
      document.getElementById("bulkTrackBtn").onclick = () => {
        const s = current();
        const a = getAnn(s);
        if (isComplete(s)) {
          persistAnnotation(s.sample_id, a, { advance: false, bulkTrack: true });
        }
      };
      document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT") return;
        if (e.key === "ArrowLeft") go(-1);
        if (e.key === "ArrowRight") go(1);
        if (e.key === "u" || e.key === "U") goUnlabeled();
        if (e.key === "z" || e.key === "Z") undo();
        if (e.key === " ") { e.preventDefault(); goUnlabeled(); }
        const n = parseInt(e.key, 10);
        if (n >= 1 && n <= 6) setField("action_state", UI.action_state[n - 1][0], { advance: false });
        if (e.key === "i" || e.key === "I") setField("rally_phase", "in_play", { advance: false });
        if (e.key === "o" || e.key === "O") setField("rally_phase", "dead_time", { advance: false });
        if (n >= 5 && n <= 9) {
          const conf = UI.confidence.find(c => c[1] === String(n));
          if (conf) setField("label_confidence", conf[0], { advance: true });
        }
      });
    }

    async function init() {
      const data = await api("/api/samples");
      allSamples = data.samples;
      videoMap = data.video_map || {};
      allSamples.forEach(s => {
        ann[s.sample_id] = getAnn(s);
      });
      applyFilter();
      buildUI();
      render();
      refreshStats();
    }
    init();
  </script>
</body>
</html>
"""


def _guess_video_path(session_id: str, registry_path: Path) -> Optional[Path]:
    registry = load_registry(registry_path)
    for session in registry.get("sessions", []):
        if session["session_id"] == session_id:
            videos = session.get("original_videos") or []
            if videos:
                path = Path(videos[0])
                if path.exists():
                    return path
    return None


def build_video_map(
    samples: List[Dict[str, Any]],
    registry_path: Path,
) -> Dict[str, str]:
    session_ids = sorted({s["session_id"] for s in samples})
    out: Dict[str, str] = {}
    for sid in session_ids:
        if _guess_video_path(sid, registry_path):
            out[sid] = f"/video/{sid}"
    return out


class AnnotateServer:
    """Threading HTTP server for interactive player action labeling."""

    def __init__(
        self,
        manifest_path: Path,
        datasets_root: Path,
        registry_path: Path,
        labeled_path: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.manifest_path = manifest_path.resolve()
        self.datasets_root = datasets_root.resolve()
        self.registry_path = registry_path.resolve()
        self.host = host
        self.port = port
        self.store = LabelStore(manifest_path, labeled_path)
        self._lock = threading.Lock()
        self._video_cache: Dict[str, Path] = {}

        for sample in self.store.get_samples():
            sid = sample["session_id"]
            if sid not in self._video_cache:
                path = _guess_video_path(sid, self.registry_path)
                if path:
                    self._video_cache[sid] = path

        handler_cls = self._make_handler()
        self.httpd = ThreadingHTTPServer((host, port), handler_cls)

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send_json(self, payload: Any, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                return json.loads(raw.decode("utf-8"))

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    ui_config = {
                        "action_state": ACTION_STATE_UI,
                        "rally": RALLY_UI,
                        "confidence": CONFIDENCE_UI,
                    }
                    html = INTERACTIVE_HTML.replace(
                        "__UI_CONFIG_JSON__",
                        json.dumps(ui_config, ensure_ascii=False),
                    )
                    body = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/api/samples":
                    samples = server.store.get_samples()
                    payload = {
                        "samples": samples,
                        "video_map": build_video_map(samples, server.registry_path),
                    }
                    self._send_json(payload)
                    return

                if path == "/api/stats":
                    self._send_json(server.store.stats())
                    return

                crop_match = re.match(r"^/crop/(.+)$", path)
                if crop_match:
                    rel = unquote(crop_match.group(1))
                    file_path = server.datasets_root / rel
                    server._serve_file(file_path, self)
                    return

                frame_match = re.match(r"^/api/frame/(.+)$", path)
                if frame_match:
                    sample_id = unquote(frame_match.group(1))
                    row = server.store._by_id.get(sample_id)
                    if row is None:
                        self.send_error(404)
                        return
                    if row.get("full_frame_path"):
                        file_path = server.datasets_root / row["full_frame_path"]
                        if file_path.is_file():
                            server._serve_file(file_path, self)
                            return
                    video_path = server._video_cache.get(row["session_id"])
                    if video_path is None or not video_path.exists():
                        self.send_error(404)
                        return
                    cache_dir = (
                        server.datasets_root
                        / "player_actions"
                        / "full_frame"
                        / row["session_id"]
                    )
                    cache_path = cache_dir / f"{sample_id}_bbox.jpg"
                    try:
                        out = render_full_frame_jpg(
                            video_path,
                            float(row["t"]),
                            cache_path,
                            frame_index=row.get("frame_index"),
                            bbox_norm=row.get("bbox"),
                            force=row.get("frame_index") is not None,
                        )
                        server._serve_file(out, self)
                    except (ValueError, OSError):
                        self.send_error(404)
                    return

                video_match = re.match(r"^/video/([A-Za-z0-9_]+)$", path)
                if video_match:
                    sid = video_match.group(1)
                    file_path = server._video_cache.get(sid)
                    if file_path and file_path.exists():
                        server._serve_file(file_path, self)
                    else:
                        self.send_error(404)
                    return

                self.send_error(404)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                try:
                    data = self._read_json()
                    if path == "/api/label":
                        row = server.store.set_annotation(
                            data["sample_id"],
                            action_state=data.get("action_state"),
                            rally_phase=data.get("rally_phase"),
                            label_confidence=data.get("label_confidence"),
                            frame_align=data.get("frame_align"),
                            is_target_player=data.get("is_target_player"),
                            notes=data.get("notes"),
                        )
                        self._send_json({"ok": True, "sample": row})
                        return
                    if path == "/api/qa":
                        row = server.store.update_qa(
                            data["sample_id"],
                            frame_align=data.get("frame_align"),
                            is_target_player=data.get("is_target_player"),
                        )
                        self._send_json({"ok": True, "sample": row})
                        return
                    if path == "/api/label/bulk":
                        count = server.store.set_labels_bulk(
                            data["sample_ids"],
                            action_state=data["action_state"],
                            rally_phase=data["rally_phase"],
                            label_confidence=float(data["label_confidence"]),
                        )
                        self._send_json({"ok": True, "updated": count})
                        return
                except (KeyError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                self.send_error(404)

        return Handler

    @staticmethod
    def _safe_write(handler: BaseHTTPRequestHandler, data: bytes) -> None:
        try:
            handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Browser cancelled the request (e.g. rapid sample switching).
            pass

    @staticmethod
    def _serve_file(file_path: Path, handler: BaseHTTPRequestHandler) -> None:
        if not file_path.is_file():
            handler.send_error(404)
            return
        file_size = file_path.stat().st_size
        mime, _ = mimetypes.guess_type(str(file_path))
        mime = mime or "application/octet-stream"
        range_header = handler.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                with open(file_path, "rb") as f:
                    f.seek(start)
                    chunk = f.read(length)
                handler.send_response(206)
                handler.send_header("Content-Type", mime)
                handler.send_header("Content-Length", str(len(chunk)))
                handler.send_header(
                    "Content-Range",
                    f"bytes {start}-{end}/{file_size}",
                )
                handler.send_header("Accept-Ranges", "bytes")
                handler.end_headers()
                AnnotateServer._safe_write(handler, chunk)
                return
        with open(file_path, "rb") as f:
            data = f.read()
        handler.send_response(200)
        handler.send_header("Content-Type", mime)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        AnnotateServer._safe_write(handler, data)

    def serve_forever(self) -> None:
        print(f"Annotation server: http://{self.host}:{self.port}")
        print(f"Manifest: {self.manifest_path}")
        print(f"Labeled:  {self.store.labeled_path}")
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()


def run_annotate_server(
    manifest_path: Path,
    datasets_root: Path,
    registry_path: Path,
    *,
    labeled_path: Optional[Path] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = AnnotateServer(
        manifest_path,
        datasets_root,
        registry_path,
        labeled_path=labeled_path,
        host=host,
        port=port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()
