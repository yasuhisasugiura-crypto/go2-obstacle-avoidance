# -*- coding: utf-8 -*-
"""Simple reactive obstacle avoider for small boxes (no gap-width judgement).

Idea
  1. Turn the 128-column depth row into world-frame obstacle points and remember them.
  2. Head to the goal. If the straight corridor (robot radius + margin) is blocked,
     pick the heading closest to the goal direction whose corridor is clear.
  3. NEVER output "stop and stare". If no heading is clear, relax the margin, then take the
     heading with the largest clearance. A stall watchdog forces a committed sidestep.

Conventions: yaw [rad], x forward, y left. Image column 0 = left.
Depth is axial (z-depth). Non-finite depth = nothing hit = free.
"""
import math
from collections import deque


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SimpleAvoider:
    def __init__(self, goal, radius=0.22, margin=0.10, lookahead=0.9,
                 tan_half=1.0, cam_off=0.3, max_range=3.0, goal_free=0.5,
                 stall_steps=200, stall_dist=0.15, escape_steps=60):
        self.goal = goal
        self.radius = radius
        self.margin = margin
        self.lookahead = lookahead
        self.tan_half = tan_half
        self.cam_off = cam_off
        self.max_range = max_range
        self.goal_free = goal_free
        self.stall_steps = stall_steps
        self.stall_dist = stall_dist
        self.escape_steps = escape_steps
        self.cells = {}
        self.prev = 0.0
        self.hist = deque(maxlen=stall_steps)
        self.escape_left = 0
        self.escape_dir = 1

    # ---- perception -------------------------------------------------
    def observe(self, px, py, yaw, fine):
        n = len(fine)
        cx = px + self.cam_off * math.cos(yaw)
        cy = py + self.cam_off * math.sin(yaw)
        for i, d in enumerate(fine):
            if not math.isfinite(d) or d <= 0.05 or d >= self.max_range:
                continue
            a = math.atan((1.0 - 2.0 * (i + 0.5) / n) * self.tan_half)
            fwd, lat = d, d * math.tan(a)
            wx = cx + fwd * math.cos(yaw) - lat * math.sin(yaw)
            wy = cy + fwd * math.sin(yaw) + lat * math.cos(yaw)
            if math.hypot(wx - self.goal[0], wy - self.goal[1]) < self.goal_free:
                continue
            self.cells[(round(wx / 0.04), round(wy / 0.04))] = (wx, wy)

    def _local_points(self, px, py, yaw, rng=2.0):
        c, s = math.cos(yaw), math.sin(yaw)
        pts = []
        for wx, wy in self.cells.values():
            dx, dy = wx - px, wy - py
            if dx * dx + dy * dy > rng * rng:
                continue
            x = dx * c + dy * s
            y = -dx * s + dy * c
            if x > -0.1:
                pts.append((x, y))
        return pts

    def _clearance(self, pts, a):
        c, s = math.cos(a), math.sin(a)
        best = 99.0
        for x, y in pts:
            t = max(0.0, min(self.lookahead, x * c + y * s))
            best = min(best, math.hypot(x - t * c, y - t * s))
        return best

    # ---- decision ---------------------------------------------------
    def decide(self, px, py, yaw):
        info = {}
        self.hist.append((px, py))
        bearing = wrap(math.atan2(self.goal[1] - py, self.goal[0] - px) - yaw)
        pts = self._local_points(px, py, yaw)
        info["n_pts"] = len(pts)

        # stall watchdog: committed sidestep
        if self.escape_left > 0:
            self.escape_left -= 1
            info["reason"] = "escape"
            return 0.15, 0.6 * self.escape_dir, self.prev, info
        if len(self.hist) == self.stall_steps:
            x0, y0 = self.hist[0]
            if math.hypot(px - x0, py - y0) < self.stall_dist:
                left = sum(1 for _, y in pts if y > 0)
                right = sum(1 for _, y in pts if y < 0)
                self.escape_dir = -1 if left > right else 1
                self.escape_left = self.escape_steps
                self.hist.clear()
                info["reason"] = "escape_start"
                return 0.0, 0.6 * self.escape_dir, self.prev, info

        limit = math.radians(75)
        cands = [math.radians(a) for a in range(-75, 76, 5)]
        cands.append(max(-limit, min(limit, bearing)))
        scored = [(a, self._clearance(pts, a)) for a in cands]

        chosen, level = None, "full"
        for need, name in ((self.radius + self.margin, "full"),
                           (self.radius + 0.5 * self.margin, "relaxed")):
            ok = [(abs(a - bearing) + 0.3 * abs(a - self.prev), a, cl)
                  for a, cl in scored if cl > need]
            if ok:
                _, a, cl = min(ok)
                chosen, level = (a, cl), name
                break
        if chosen is None:
            # nothing is clear: take the most open heading (still moving slowly)
            a, cl = max(scored, key=lambda z: (z[1], -abs(z[0] - bearing)))
            chosen, level = (a, cl), "best_effort"
        target, clr = chosen
        self.prev = target

        wz = max(-0.8, min(0.8, 1.5 * target))
        ad = abs(target)
        if level == "best_effort":
            vx = 0.0 if ad > math.radians(20) else 0.1
        elif ad > math.radians(40):
            vx = 0.0
        else:
            vx = 0.4 * max(0.25, 1.0 - ad / math.radians(45))
        info.update({"reason": level, "target_deg": round(math.degrees(target), 1),
                     "clearance": round(clr, 3), "bearing_deg": round(math.degrees(bearing), 1)})
        return vx, wz, target, info
