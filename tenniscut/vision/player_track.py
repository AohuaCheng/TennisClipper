"""Simple IOU-based player bounding-box tracker."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class PlayerTracker:
    """Match per-frame YOLO detections to stable track IDs."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.active_tracks: List[Dict[str, Any]] = []
        self.tracks: List[List[Dict[str, Any]]] = []

    def update(self, players: List[Dict[str, Any]], t: float) -> List[Dict[str, Any]]:
        """Match current players to tracks and return players with track_id."""
        if not players:
            self.active_tracks = []
            return []

        unmatched_tracks = list(self.active_tracks)
        matched: List[Optional[Dict[str, Any]]] = [None] * len(players)

        for tr in unmatched_tracks:
            best_iou = 0.0
            best_pi = -1
            for pi, p in enumerate(players):
                if matched[pi] is not None:
                    continue
                iou = self._iou(tr["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_iou >= self.iou_threshold and best_pi >= 0:
                matched[best_pi] = tr
                tr["bbox"] = players[best_pi]["bbox"]
                tr["center"] = players[best_pi]["center"]
                tr["role"] = players[best_pi].get("role", "unknown")
                tr["last_seen"] = t

        for pi, p in enumerate(players):
            if matched[pi] is None:
                new_track = {
                    "track_id": self.next_id,
                    "bbox": p["bbox"],
                    "center": p["center"],
                    "role": p.get("role", "unknown"),
                    "first_seen": t,
                    "last_seen": t,
                }
                self.next_id += 1
                self.active_tracks.append(new_track)
                self.tracks.append([new_track])
                matched[pi] = new_track
            else:
                track = matched[pi]
                hist = next(
                    (tr for tr in self.tracks if tr[0]["track_id"] == track["track_id"]),
                    None,
                )
                if hist is not None:
                    hist.append(
                        {
                            "t": t,
                            "bbox": track["bbox"],
                            "center": track["center"],
                            "role": track["role"],
                        }
                    )

        self.active_tracks = [
            tr for tr in self.active_tracks if t - tr.get("last_seen", t) < 2.0
        ]

        return [
            {**p, "track_id": matched[pi]["track_id"]}
            for pi, p in enumerate(players)
            if matched[pi] is not None
        ]

    def _iou(self, a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        union = a_area + b_area - inter_area
        return inter_area / union if union > 0 else 0.0
