"""Visual-only tennis hit detection prototype.

Uses player pose (wrist/elbow motion) as the primary signal because ball
detection is too sparse in the 300-360s test segment. Sparse ball candidates
and court geometry are used only for confirmation and end detection.

This is intentionally a prototype module so it can be tested independently
before integration into the main pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tenniscut.video.ingest import read_frames
from tenniscut.vision.ball import detect_ball_candidates
from tenniscut.vision.ball_color import load_profile_from_session
from tenniscut.vision.ball_track import track_ball_trajectory, filter_player_attached_trajectories
from tenniscut.segmentation.ball_rally import analyze_ball_events
from tenniscut.vision.court_lines import CourtGeometry
from tenniscut.vision.player_track import PlayerTracker
from tenniscut.vision.players import detect_players_in_frame, get_player_motion_mask
from tenniscut.vision.pose import COCO_KEYPOINTS, estimate_pose
from tenniscut.vision.roi import CourtROI, load_roi_from_session


class SwingDetector:
    """Detect tennis swings from pose keypoint sequences.

    A swing is a local peak in wrist speed, not just any frame where the wrist
    is moving. This avoids firing on continuous non-hit movements during a rally.
    """

    def __init__(
        self,
        wrist_speed_threshold: float = 250.0,  # px/s minimum for a peak
        elbow_speed_threshold: float = 150.0,  # px/s
        min_wrist_height_ratio: float = 0.65,  # wrist must be above hip level
        peak_window: float = 0.5,  # seconds around a peak to merge
        min_prominence: float = 50.0,  # px/s prominence vs neighbors
    ):
        self.wrist_speed_threshold = wrist_speed_threshold
        self.elbow_speed_threshold = elbow_speed_threshold
        self.min_wrist_height_ratio = min_wrist_height_ratio
        self.peak_window = peak_window
        self.min_prominence = min_prominence

    def detect(
        self,
        track_id: int,
        pose_history: List[Dict[str, Any]],
        t: float,
        frame_height: int,
    ) -> Optional[Dict[str, Any]]:
        """Return swing event if a speed peak is detected at time t for this track."""
        if len(pose_history) < 3:
            return None

        # Need a local peak: last point must be higher than previous and next.
        # Since we only have history, we detect a peak when the current point is
        # higher than the previous two and the last point is the local max so far.
        # A peak is "confirmed" when speed drops in the next sample.
        if len(pose_history) < 4:
            return None

        # Compute wrist speed for the last 4 samples
        left_wrist_idx = COCO_KEYPOINTS.index("left_wrist")
        right_wrist_idx = COCO_KEYPOINTS.index("right_wrist")
        left_elbow_idx = COCO_KEYPOINTS.index("left_elbow")
        right_elbow_idx = COCO_KEYPOINTS.index("right_elbow")

        speeds = []
        for i in range(1, len(pose_history)):
            prev = pose_history[i - 1]
            curr = pose_history[i]
            dt = curr.get("t", t) - prev.get("t", t - 0.2)
            if dt <= 0:
                dt = 0.2
            prev_kps = prev.get("keypoints", [])
            curr_kps = curr.get("keypoints", [])
            if len(prev_kps) < 17 or len(curr_kps) < 17:
                continue
            side, wrist_speed, elbow_speed = self._compute_dominant_limb_speed(
                prev_kps, curr_kps, dt,
                left_wrist_idx, right_wrist_idx,
                left_elbow_idx, right_elbow_idx,
            )
            if side is None:
                continue
            wrist = curr_kps[right_wrist_idx if side == "right" else left_wrist_idx]
            speeds.append({
                "t": curr.get("t", t),
                "wrist_speed": wrist_speed,
                "elbow_speed": elbow_speed,
                "wrist_y": wrist[1],
                "wrist_vis": wrist[2],
                "side": side,
            })

        if len(speeds) < 4:
            return None

        # Peak is at the second-to-last sample (we need the last sample to confirm drop)
        peak_idx = len(speeds) - 2
        peak = speeds[peak_idx]
        if peak["wrist_speed"] < self.wrist_speed_threshold:
            return None

        # Check prominence: peak must be higher than neighbors by min_prominence
        left_neighbor = max(speeds[i]["wrist_speed"] for i in range(max(0, peak_idx - 2), peak_idx))
        right_neighbor = speeds[peak_idx + 1]["wrist_speed"]
        if peak["wrist_speed"] - left_neighbor < self.min_prominence:
            return None
        if peak["wrist_speed"] - right_neighbor < self.min_prominence:
            return None

        # Height check: wrist must be above hip level
        if peak["wrist_y"] > frame_height * self.min_wrist_height_ratio:
            return None

        # Confidence
        confidence = 0.5
        if peak["wrist_speed"] >= self.wrist_speed_threshold:
            confidence += 0.2
        if peak["wrist_speed"] >= self.wrist_speed_threshold * 1.5:
            confidence += 0.1
        if peak["elbow_speed"] >= self.elbow_speed_threshold:
            confidence += 0.1
        if peak["wrist_vis"] > 0.5:
            confidence += 0.1
        confidence = min(1.0, confidence)

        return {
            "t": round(peak["t"], 3),
            "track_id": track_id,
            "type": "swing",
            "side": peak["side"],
            "wrist_speed": round(peak["wrist_speed"], 1),
            "elbow_speed": round(peak["elbow_speed"], 1),
            "confidence": round(confidence, 3),
        }

    def _compute_dominant_limb_speed(
        self,
        prev_kps: List[List[float]],
        curr_kps: List[List[float]],
        dt: float,
        left_wrist_idx: int,
        right_wrist_idx: int,
        left_elbow_idx: int,
        right_elbow_idx: int,
    ) -> Tuple[Optional[str], float, float]:
        def speed(idx: int) -> Tuple[float, float]:
            p = prev_kps[idx]
            c = curr_kps[idx]
            if p[2] < 0.2 or c[2] < 0.2:
                return 0.0, min(p[2], c[2])
            dist = np.hypot(c[0] - p[0], c[1] - p[1])
            return dist / dt, c[2]

        lw_speed, lw_vis = speed(left_wrist_idx)
        rw_speed, rw_vis = speed(right_wrist_idx)
        le_speed, _ = speed(left_elbow_idx)
        re_speed, _ = speed(right_elbow_idx)

        # Pick side with higher wrist speed, but require visibility
        if rw_speed >= lw_speed and rw_vis > 0.2:
            return "right", rw_speed, re_speed
        if lw_vis > 0.2:
            return "left", lw_speed, le_speed
        return None, 0.0, 0.0


class VisualHitDetector:
    """Detect tennis hits from player pose and sparse ball candidates."""

    def __init__(
        self,
        roi_cfg: CourtROI,
        color_profile: Dict[str, Any],
        court_geom: Optional[CourtGeometry] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        player_fps: float = 5.0,
        ball_fps: float = 30.0,
        ball_method: str = "combined",
        max_candidates: int = 10,
        require_motion: bool = False,
        racket_extend: bool = False,
        player_padding: int = 10,
        color_conf_threshold: float = 0.20,
        motion_conf_threshold: float = 0.20,
        motion_threshold: int = 12,
        max_area: int = 300,
        min_track_length: int = 2,
        max_gap_frames: int = 15,
        require_reversal: bool = False,
    ):
        self.roi_cfg = roi_cfg
        self.color_profile = dict(color_profile)
        self.color_profile["max_area"] = max_area
        self.court_geom = court_geom
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.player_fps = player_fps
        self.ball_fps = ball_fps
        self.ball_method = ball_method
        self.max_candidates = max_candidates
        self.require_motion = require_motion
        self.racket_extend = racket_extend
        self.player_padding = player_padding
        self.color_conf_threshold = color_conf_threshold
        self.motion_conf_threshold = motion_conf_threshold
        self.motion_threshold = motion_threshold
        self.min_track_length = min_track_length
        self.max_gap_frames = max_gap_frames
        self.require_reversal = require_reversal
        self.court_mask = roi_cfg.combined_court_mask()

        self.tracker = PlayerTracker(iou_threshold=0.25)
        self.swing_detector = SwingDetector()
        self.pose_history: Dict[int, List[Dict[str, Any]]] = {}
        self.ball_candidates: List[Dict[str, Any]] = []
        self.swing_events: List[Dict[str, Any]] = []
        self.hit_events: List[Dict[str, Any]] = []
        self.candidates_per_frame: List[List[Dict[str, Any]]] = []

    def run(
        self,
        video_path: Path,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run visual hit detection on a video segment.

        Returns dict with hit_times, hit_events, swing_events, player_timeline, ball_candidates.
        """
        player_interval = max(1, int(self.ball_fps / self.player_fps))
        prev_frame: Optional[np.ndarray] = None
        frames_processed = 0
        last_players: List[Dict[str, Any]] = []
        last_player_mask: Optional[np.ndarray] = None
        player_timeline: List[Dict[str, Any]] = []

        for frame in read_frames(
            video_path, fps=self.ball_fps, duration=duration, start_time=start_time
        ):
            current_time = start_time + frames_processed / self.ball_fps
            frames_processed += 1

            # Player detection at player_fps
            player_bboxes = None
            player_mask = None
            if frames_processed % player_interval == 0:
                try:
                    vision = detect_players_in_frame(
                        frame, roi=self.roi_cfg, conf_threshold=0.4
                    )
                    last_players = self.tracker.update(
                        vision["players"], current_time
                    )
                    if last_players:
                        last_player_mask = get_player_motion_mask(
                            frame, last_players, padding=30
                        )
                    player_timeline.append(
                        {"t": current_time, "players": list(last_players)}
                    )

                    # Pose estimation for each tracked player
                    for p in last_players:
                        pose = estimate_pose(frame, p["bbox"])
                        if pose:
                            pose["t"] = current_time
                            pose["track_id"] = p["track_id"]
                            self.pose_history.setdefault(p["track_id"], []).append(pose)
                            swing = self.swing_detector.detect(
                                p["track_id"],
                                self.pose_history[p["track_id"]],
                                current_time,
                                self.frame_height,
                            )
                            if swing:
                                self.swing_events.append(swing)
                except RuntimeError:
                    last_players = []
                    last_player_mask = None

            if last_players:
                player_bboxes = [p["bbox"] for p in last_players]
            player_mask = last_player_mask

            # Ball candidates at ball_fps (sparse, for confirmation only)
            cands = detect_ball_candidates(
                frame,
                prev_frame=prev_frame,
                roi_mask=self.court_mask,
                color_profile=self.color_profile,
                player_bboxes=player_bboxes,
                player_mask=player_mask,
                method=self.ball_method,
                max_candidates=self.max_candidates,
                net_line_y=self.roi_cfg.net_line_y or 0.5,
                require_motion=self.require_motion,
                racket_extend=self.racket_extend,
                player_padding=self.player_padding,
                color_conf_threshold=self.color_conf_threshold,
                motion_conf_threshold=self.motion_conf_threshold,
                motion_threshold=self.motion_threshold,
            )
            for c in cands:
                c["t"] = current_time
            self.ball_candidates.extend(cands)
            self.candidates_per_frame.append(cands)
            prev_frame = frame

            if progress_callback and frames_processed % 500 == 0:
                progress_callback(frames_processed)

        trajectory_result = self._track_ball(player_timeline, start_time=start_time)
        ball_events = self._analyze_ball_events(trajectory_result, player_timeline)
        self._build_hits(player_timeline, trajectory_result)

        return {
            "hit_times": sorted({e["t"] for e in self.hit_events}),
            "hit_events": self.hit_events,
            "swing_events": self.swing_events,
            "player_timeline": player_timeline,
            "ball_candidates": self.ball_candidates,
            "candidates_per_frame": self.candidates_per_frame,
            "trajectory_result": trajectory_result,
            "ball_events": ball_events,
            "tracks": self.tracker.tracks,
            "stats": self._compute_stats(trajectory_result, frames_processed),
        }

    def _track_ball(
        self,
        player_timeline: List[Dict[str, Any]],
        start_time: float = 0.0,
    ) -> Dict[str, Any]:
        trajectory_result = track_ball_trajectory(
            self.candidates_per_frame,
            fps=self.ball_fps,
            start_time=start_time,
            frame_height=self.frame_height,
            min_track_length=self.min_track_length,
            max_gap_frames=self.max_gap_frames,
            require_reversal=self.require_reversal,
        )
        trajectory_result = filter_player_attached_trajectories(
            trajectory_result,
            player_timeline,
            net_y_px=(self.roi_cfg.net_line_y or 0.5) * self.frame_height,
            frame_height=self.frame_height,
        )
        return trajectory_result

    def _analyze_ball_events(
        self, trajectory_result: Dict[str, Any], player_timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return analyze_ball_events(
            trajectory_result,
            player_timeline=player_timeline,
            roi_cfg=self.roi_cfg,
            fps=self.ball_fps,
            net_line_y=self.roi_cfg.net_line_y or 0.5,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            court_geometry=self.court_geom,
        )

    def _compute_stats(
        self, trajectory_result: Dict[str, Any], frames_processed: int
    ) -> Dict[str, Any]:
        points = trajectory_result.get("points", [])
        valid = sum(1 for p in points if p.get("track_id") is not None)
        total = frames_processed if frames_processed > 0 else len(points)
        detection_rate = valid / total if total else 0.0
        tracks = trajectory_result.get("trajectories", [])
        valid_tracks = len(tracks)
        total_reversals = sum(t.get("reversals", 0) for t in tracks)
        return {
            "detection_rate": round(detection_rate, 4),
            "valid_tracks": valid_tracks,
            "total_frames": total,
            "tracked_points": len(points),
            "total_reversals": total_reversals,
            "rejected_static_tracks": 0,
        }

    def _build_hits(
        self, player_timeline: List[Dict[str, Any]], trajectory_result: Dict[str, Any]
    ) -> None:
        """Convert swing events into hit events, using ball proximity to boost confidence.

        Also adds hits from ball trajectory direction changes when they coincide with
        a player swing or are near a player.
        """
        # Sort swings by time and merge nearby swings from different players
        swings = sorted(self.swing_events, key=lambda e: e["t"])
        merged_swings: List[Dict[str, Any]] = []
        for s in swings:
            if not merged_swings:
                merged_swings.append(s)
                continue
            last = merged_swings[-1]
            if s["t"] - last["t"] <= 0.25:
                # Keep higher confidence
                if s["confidence"] > last["confidence"]:
                    merged_swings[-1] = s
            else:
                merged_swings.append(s)

        # Check ball proximity
        for s in merged_swings:
            nearby_ball = self._nearest_ball(s["t"], s.get("track_id"), player_timeline)
            confidence = s["confidence"]
            if nearby_ball is not None and nearby_ball["dist"] < 200.0:
                confidence = min(1.0, confidence + 0.15)

            self.hit_events.append(
                {
                    "t": s["t"],
                    "type": "rally_hit",
                    "confidence": round(confidence, 3),
                    "source": "swing",
                    "track_id": s.get("track_id"),
                    "wrist_speed": s.get("wrist_speed"),
                    "ball_proximity": nearby_ball["dist"] if nearby_ball else None,
                }
            )

        # Add trajectory-based hits: direction changes that are near a player
        existing_times = {e["t"] for e in self.hit_events}
        for track in trajectory_result.get("trajectories", []):
            pts = track.get("points", [])
            if len(pts) < 3:
                continue
            for i in range(1, len(pts) - 1):
                p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
                v1 = np.array([p1["x"] - p0["x"], p1["y"] - p0["y"]])
                v2 = np.array([p2["x"] - p1["x"], p2["y"] - p1["y"]])
                if np.linalg.norm(v1) < 1 or np.linalg.norm(v2) < 1:
                    continue
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                if angle < 90:
                    continue
                # Direction change near a player -> likely a hit
                nearest_player = self._nearest_player(p1["t"], p1["x"], p1["y"], player_timeline)
                if nearest_player and nearest_player["dist"] < 250.0:
                    t_hit = round(p1["t"], 3)
                    if not any(abs(t_hit - e["t"]) <= 0.3 for e in self.hit_events):
                        self.hit_events.append(
                            {
                                "t": t_hit,
                                "type": "rally_hit",
                                "confidence": 0.6,
                                "source": "ball_direction_change",
                                "track_id": nearest_player.get("track_id"),
                                "ball_proximity": nearest_player["dist"],
                            }
                        )

    def _nearest_player(
        self, t: float, x: float, y: float, player_timeline: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return nearest player to a ball position at time t."""
        best: Optional[Tuple[float, int, Dict[str, Any]]] = None
        for rec in player_timeline:
            if abs(rec["t"] - t) > 0.2:
                continue
            for p in rec.get("players", []):
                cx, cy = p.get("center", [0, 0])
                dist = np.hypot(x - cx, y - cy)
                if best is None or dist < best[0]:
                    best = (dist, p.get("track_id", -1), p)
        if best is None:
            return None
        return {"dist": best[0], "track_id": best[1], "player": best[2]}

    def _nearest_ball(
        self, t: float, track_id: Optional[int], player_timeline: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return nearest ball candidate to the player at time t."""
        player_pos = None
        for rec in player_timeline:
            if abs(rec["t"] - t) < 0.15:
                for p in rec.get("players", []):
                    if p.get("track_id") == track_id:
                        player_pos = np.array(p["center"])
                        break
                if player_pos is not None:
                    break
        if player_pos is None:
            return None

        best: Optional[Tuple[float, float, float]] = None
        for c in self.ball_candidates:
            if abs(c["t"] - t) <= 0.2:
                dist = np.linalg.norm(np.array([c["x"], c["y"]]) - player_pos)
                if best is None or dist < best[2]:
                    best = (c["x"], c["y"], dist)
        return {"x": best[0], "y": best[1], "dist": best[2]} if best else None


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="sessions/test_session_7252")
    parser.add_argument("--start", type=float, default=300.0)
    parser.add_argument("--end", type=float, default=360.0)
    args = parser.parse_args()

    session = Path(args.session)
    with open(session / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    video_path = Path(cfg["videos"][0])

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    roi = load_roi_from_session(session)
    roi.set_frame_size(fw, fh)
    color_profile = load_profile_from_session(session)

    geom_path = session / "work" / "court_geometry.json"
    if geom_path.exists():
        court_geom = CourtGeometry.load(geom_path)
    else:
        court_geom = None

    detector = VisualHitDetector(
        roi, color_profile, court_geom=court_geom, frame_width=fw, frame_height=fh
    )
    result = detector.run(video_path, start_time=args.start, duration=args.end - args.start)

    print(f"Detected {len(result['hit_times'])} hits:")
    for h in result["hit_events"]:
        print(
            f"  t={h['t']:.2f} conf={h['confidence']:.2f} "
            f"speed={h.get('wrist_speed')} ball_dist={h.get('ball_proximity')}"
        )
