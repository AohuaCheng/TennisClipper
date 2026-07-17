#!/usr/bin/env python3
"""Train BiGRU rally baseline on oracle Layer1 scene sequences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tenniscut.ml.rally_sequence import (  # noqa: E402
    build_session_sequences,
    load_scene_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="BiGRU rally baseline")
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "datasets/player_actions/scene_frames")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/eval/rally_bigru.pt")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    except ImportError:
        print("Install torch: pip install torch", file=sys.stderr)
        sys.exit(1)

    train_scenes = load_scene_frames(args.scene_dir / f"{args.train_split}_scene_frames.jsonl")
    test_scenes = load_scene_frames(args.scene_dir / f"{args.test_split}_scene_frames.jsonl")
    train_seqs, train_labels, train_weights, _ = build_session_sequences(train_scenes)
    test_seqs, test_labels, test_weights, _ = build_session_sequences(test_scenes)

    if not train_seqs:
        print("No trainable sequences", file=sys.stderr)
        sys.exit(1)

    player_dim = train_seqs[0].shape[-1]
    flat_dim = player_dim * train_seqs[0].shape[1]

    class BiGRURally(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(flat_dim, args.hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(args.hidden * 2, 1)

        def forward(self, x, lengths):
            b, t, p, f = x.shape
            flat = x.reshape(b, t, p * f)
            packed = pack_padded_sequence(flat, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
            return self.head(out).squeeze(-1)

    def batchify(seqs, labels, weights):
        lengths = [s.shape[0] for s in seqs]
        max_t = max(lengths)
        p, f = seqs[0].shape[1], seqs[0].shape[2]
        x = np.zeros((len(seqs), max_t, p, f), dtype=np.float32)
        y = np.full((len(seqs), max_t), -1.0, dtype=np.float32)
        w = np.zeros((len(seqs), max_t), dtype=np.float32)
        for i, (seq, lab, wt) in enumerate(zip(seqs, labels, weights)):
            x[i, : seq.shape[0]] = seq
            y[i, : len(lab)] = lab
            w[i, : len(wt)] = wt
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(w),
            torch.tensor(lengths, dtype=torch.long),
        )

    model = BiGRURally()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(args.epochs):
        model.train()
        X, Y, W, L = batchify(train_seqs, train_labels, train_weights)
        opt.zero_grad()
        logits = model(X, L)
        mask = Y >= 0
        loss = (bce(logits, Y.clamp(0, 1)) * W * mask.float()).sum() / mask.float().sum().clamp(min=1)
        loss.backward()
        opt.step()
        print(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for seq, lab, wt in zip(test_seqs, test_labels, test_weights):
            X = torch.from_numpy(seq).unsqueeze(0)
            L = torch.tensor([seq.shape[0]])
            logits = model(X, L).squeeze(0).numpy()
            prob = 1 / (1 + np.exp(-logits))
            all_pred.extend((prob >= 0.5).astype(int).tolist())
            all_true.extend(lab.astype(int).tolist())
    acc = float((np.array(all_pred) == np.array(all_true)).mean())
    report = {"model": "bigru", "accuracy": round(acc, 4), "support": len(all_true)}
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "player_dim": player_dim}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
