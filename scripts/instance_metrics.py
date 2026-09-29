"""Object-level metrics for building instance segmentation.

Pixel F1 is the wrong unit here. A model that merges every terrace into one
blob can score 84 pixel F1 while finding a fifth of the buildings - measured
on fold 0, exactly that happened. These metrics count objects.

  AP     COCO-style average precision, swept over IoU 0.50:0.95. Needs a
         confidence per predicted instance; the mean predicted probability
         inside it serves. This is what the detection literature reports.
  PQ     Panoptic Quality = SQ x RQ, and the decomposition is the useful part:
         RQ is an F1 over matched objects (did you find the right things) and
         SQ the mean IoU of those matches (are the outlines any good). A model
         can have high SQ and low RQ - good masks, wrong object count - which
         is precisely this failure.
  split/merge  how many ground-truth buildings were cut in two, and how many
         predictions swallowed several buildings, per 100 labelled buildings.
"""
from __future__ import annotations

import numpy as np


class InstanceMatcher:
    """Greedy IoU matching between predicted and ground-truth label images."""

    def __init__(self, min_px=30):
        self.min_px = min_px

    @staticmethod
    def _ids(lab, min_px):
        ids, counts = np.unique(lab, return_counts=True)
        return [int(i) for i, c in zip(ids, counts) if i > 0 and c >= min_px]

    def overlaps(self, pred_lab, gt_lab):
        """One intersection histogram, everything else derived from it.

        Per-pair mask operations are O(P x G) full-image ANDs - unusable at
        715 chips. A single bincount over the pixels where both are non-zero
        gives the same numbers.
        """
        pi = self._ids(pred_lab, self.min_px)
        gi = self._ids(gt_lab, self.min_px)
        n_p, n_g = len(pi), len(gi)
        if not n_p or not n_g:
            z = np.zeros((n_p, n_g))
            return z, z, z, pi, gi
        pmap = np.zeros(int(pred_lab.max()) + 1, np.int64) - 1
        gmap = np.zeros(int(gt_lab.max()) + 1, np.int64) - 1
        for k, v in enumerate(pi):
            pmap[v] = k
        for k, v in enumerate(gi):
            gmap[v] = k
        both = (pred_lab > 0) & (gt_lab > 0)
        pv, gv = pmap[pred_lab[both]], gmap[gt_lab[both]]
        ok = (pv >= 0) & (gv >= 0)
        inter = np.bincount(pv[ok] * n_g + gv[ok],
                            minlength=n_p * n_g).reshape(n_p, n_g).astype(np.float64)
        pa = np.array([(pred_lab == v).sum() for v in pi], np.float64)[:, None]
        ga = np.array([(gt_lab == v).sum() for v in gi], np.float64)[None, :]
        iou = inter / np.maximum(pa + ga - inter, 1)
        cover_gt = inter / np.maximum(ga, 1)      # share of each GT a pred covers
        cover_pred = inter / np.maximum(pa, 1)    # share of each pred inside a GT
        return iou, cover_gt, cover_pred, pi, gi

    @staticmethod
    def match(iou, thr):
        """Greedy highest-IoU-first, one prediction per ground truth."""
        pairs = []
        used_p, used_g = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for p, g in order:
            if iou[p, g] < thr:
                break
            if p in used_p or g in used_g:
                continue
            used_p.add(int(p)); used_g.add(int(g))
            pairs.append((int(p), int(g), float(iou[p, g])))
        return pairs


class InstanceScores:
    """Accumulates over chips, then reports AP, PQ and split/merge rates."""

    IOUS = np.round(np.arange(0.50, 0.96, 0.05), 2)

    def __init__(self, min_px=30):
        self.matcher = InstanceMatcher(min_px)
        self.min_px = min_px
        self.rows = []
        self.n_gt = 0
        self.pq = {t: [0, 0, 0, 0.0] for t in self.IOUS}
        self.merges = self.splits = 0
        self.n_pred = 0
        self.empty_chips = self.empty_chips_hit = self.empty_fp = 0

    def add(self, pred_lab, gt_lab, prob=None):
        iou, cover_gt, cover_pred, pi, gi = self.matcher.overlaps(pred_lab, gt_lab)
        self.n_gt += len(gi)
        self.n_pred += len(pi)
        if not len(gi):
            # A chip with no labelled building can only ever contribute false
            # positives, and the matching loop below needs a ground truth to run
            # against - so these used to return here and vanish from PQ entirely.
            # On a benchmark that is 25% verified-empty that silently hid every
            # hallucination. Counted separately because the two false-positive
            # populations mean different things: a blob bridging two roofs costs
            # real detections, a speck on bare ground costs only precision.
            self.empty_chips += 1
            if len(pi):
                self.empty_chips_hit += 1
                self.empty_fp += len(pi)
                for t in self.IOUS:
                    self.pq[t][1] += len(pi)
            return
        if not len(pi):
            for t in self.IOUS:
                self.pq[t][2] += len(gi)      # every building missed
            return
        scores = [float(prob[pred_lab == p].mean()) if prob is not None else 1.0
                  for p in pi]
        self.rows.extend(zip(scores, iou.max(axis=1).tolist()))

        for t in self.IOUS:
            pairs = self.matcher.match(iou, t)
            acc = self.pq[t]
            acc[0] += len(pairs)
            acc[1] += len(pi) - len(pairs)
            acc[2] += len(gi) - len(pairs)
            acc[3] += sum(x[2] for x in pairs)

        # merge: one prediction takes over half of two or more buildings
        self.merges += int(((cover_gt >= 0.5).sum(axis=1) >= 2).sum())
        # split: one building is carved up by two or more predictions
        self.splits += int(((cover_pred >= 0.5).sum(axis=0) >= 2).sum())

    def average_precision(self):
        """AP per IoU threshold plus the 0.50:0.95 mean, from ranked scores."""
        if not self.rows or self.n_gt == 0:
            return {}, 0.0
        arr = np.array(sorted(self.rows, key=lambda r: -r[0]))
        out = {}
        for t in self.IOUS:
            hit = arr[:, 1] >= t
            tp = np.cumsum(hit)
            fp = np.cumsum(~hit)
            rec = tp / self.n_gt
            prec = tp / np.maximum(tp + fp, 1)
            # 101-point interpolated, as COCO does
            p = np.maximum.accumulate(prec[::-1])[::-1]
            out[float(t)] = float(np.mean(np.interp(np.linspace(0, 1, 101), rec, p,
                                                    left=p[0] if len(p) else 0, right=0)))
        return out, float(np.mean(list(out.values())))

    def panoptic(self, t=0.5):
        """PQ = SQ x RQ, with false positives on empty chips included in fp.

        `empty_fp` and `empty_chip_fp_rate` are reported alongside so the two
        false-positive populations stay separable: bridging blobs inside a
        populated scene cost detections, specks on bare ground cost precision
        only, and a single fp count conflates them.
        """
        tp, fp, fn, iou_sum = self.pq[t]
        sq = iou_sum / max(tp, 1)
        rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)
        return {"PQ": sq * rq, "SQ": sq, "RQ": rq, "tp": tp, "fp": fp, "fn": fn,
                "empty_fp": self.empty_fp,
                "populated_fp": fp - self.empty_fp,
                "empty_chips": self.empty_chips,
                "empty_chip_fp_rate": self.empty_chips_hit /
                                      max(self.empty_chips, 1)}

    def report(self):
        ap, mean_ap = self.average_precision()
        p = self.panoptic(0.5)
        per100 = 100.0 / max(self.n_gt, 1)
        lines = [f"{'ground-truth buildings':32s} {self.n_gt:,}",
                 f"{'predicted instances':32s} {self.n_pred:,} "
                 f"({100*self.n_pred/max(self.n_gt,1):.0f}% of truth)", ""]
        if ap:
            lines += [f"{'AP @[.50:.95]':32s} {100*mean_ap:6.2f}",
                      f"{'AP @.50':32s} {100*ap[0.5]:6.2f}",
                      f"{'AP @.75':32s} {100*ap[0.75]:6.2f}", ""]
        lines += [f"{'PQ  (panoptic quality)':32s} {100*p['PQ']:6.2f}",
                  f"{'  SQ (mask quality of matches)':32s} {100*p['SQ']:6.2f}",
                  f"{'  RQ (F1 over objects)':32s} {100*p['RQ']:6.2f}",
                  f"{'  matched / spurious / missed':32s} {p['tp']:,} / {p['fp']:,} / {p['fn']:,}",
                  "",
                  f"{'merges per 100 buildings':32s} {self.merges*per100:6.1f}",
                  f"{'splits per 100 buildings':32s} {self.splits*per100:6.1f}"]
        return "\n".join(lines)
