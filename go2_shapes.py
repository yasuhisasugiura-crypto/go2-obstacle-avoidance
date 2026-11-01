# -*- coding: utf-8 -*-
"""Obstacle shapes for the mixed course (pure Python; Genesis is imported only inside add_obstacle).

Each item is a dict:
  box : {"kind":"box","x","y","sx","sy","h"}      axis-aligned rectangle
  cyl : {"kind":"cyl","x","y","r","h"}            vertical cylinder
  tri : {"kind":"tri","x","y","R","yaw","h"}      equilateral triangular prism (R = circumradius, yaw in degrees)
"""
import math
import os

H_DEFAULT = 0.5


def box(x, y, sx, sy=None, h=H_DEFAULT):
    return {"kind": "box", "x": x, "y": y, "sx": sx, "sy": sx if sy is None else sy, "h": h}


def cyl(x, y, r, h=H_DEFAULT):
    return {"kind": "cyl", "x": x, "y": y, "r": r, "h": h}


def tri(x, y, R, yaw=0.0, h=H_DEFAULT):
    return {"kind": "tri", "x": x, "y": y, "R": R, "yaw": yaw, "h": h}


def tri_vertices(it):
    """World-frame (x, y) vertices, counter-clockwise."""
    out = []
    for k in range(3):
        a = math.radians(it["yaw"]) + k * 2.0 * math.pi / 3.0
        out.append((it["x"] + it["R"] * math.cos(a), it["y"] + it["R"] * math.sin(a)))
    return out


def label(it):
    if it["kind"] == "box":
        return "box %.2fx%.2f" % (it["sx"], it["sy"])
    if it["kind"] == "cyl":
        return "cyl r=%.2f" % it["r"]
    return "tri R=%.2f" % it["R"]


# ---------------- geometry (2-D) ----------------
def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _inside_poly(px, py, poly):
    s = None
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        c = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if c == 0:
            continue
        if s is None:
            s = c > 0
        elif (c > 0) != s:
            return False
    return True


def distance(it, px, py):
    """Distance from point to the obstacle surface (0 if inside)."""
    k = it["kind"]
    if k == "box":
        dx = max(abs(px - it["x"]) - it["sx"] / 2.0, 0.0)
        dy = max(abs(py - it["y"]) - it["sy"] / 2.0, 0.0)
        return math.hypot(dx, dy)
    if k == "cyl":
        return max(math.hypot(px - it["x"], py - it["y"]) - it["r"], 0.0)
    v = tri_vertices(it)
    if _inside_poly(px, py, v):
        return 0.0
    return min(_seg_dist(px, py, *v[i], *v[(i + 1) % 3]) for i in range(3))


def footprint_radius(it):
    """Radius of a circle that contains the obstacle."""
    if it["kind"] == "box":
        return math.hypot(it["sx"], it["sy"]) / 2.0
    if it["kind"] == "cyl":
        return it["r"]
    return it["R"]


def ray_hit(it, ox, oy, dx, dy, maxr=20.0):
    """Smallest t>0 where ray o + t*d (|d|=1) hits the obstacle, or None."""
    k = it["kind"]
    if k == "box":
        tmin, tmax = 0.0, maxr
        for o, d, lo, hi in ((ox, dx, it["x"] - it["sx"] / 2, it["x"] + it["sx"] / 2),
                             (oy, dy, it["y"] - it["sy"] / 2, it["y"] + it["sy"] / 2)):
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    return None
            else:
                t1, t2 = (lo - o) / d, (hi - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin, tmax = max(tmin, t1), min(tmax, t2)
                if tmin > tmax:
                    return None
        return tmin
    if k == "cyl":
        fx, fy = ox - it["x"], oy - it["y"]
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - it["r"] ** 2
        disc = b * b - c
        if disc < 0:
            return None
        s = math.sqrt(disc)
        for t in (-b - s, -b + s):
            if t > 0:
                return t
        return None
    v = tri_vertices(it)
    best = None
    for i in range(3):
        ax, ay = v[i]
        bx, by = v[(i + 1) % 3]
        ex, ey = bx - ax, by - ay
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((ax - ox) * ey - (ay - oy) * ex) / den
        u = ((ax - ox) * dy - (ay - oy) * dx) / den
        if t > 0 and 0.0 <= u <= 1.0 and (best is None or t < best):
            best = t
    return best


# ---------------- Genesis side ----------------
def _write_tri_obj(it, idx):
    h = it["h"]
    v = tri_vertices(it)
    # vertices relative to the entity position (pos = (x, y, 0))
    rel = [(vx - it["x"], vy - it["y"]) for vx, vy in v]
    path = "obstacle_tri_%d.obj" % idx
    with open(path, "w", newline="\n") as f:
        for x, y in rel:
            f.write("v %.6f %.6f 0.0\n" % (x, y))
        for x, y in rel:
            f.write("v %.6f %.6f %.6f\n" % (x, y, h))
        # bottom (1,2,3) faces down, top (4,5,6) faces up, 3 side quads as triangles
        f.write("f 1 3 2\n")
        f.write("f 4 5 6\n")
        for i in range(3):
            a, b = i + 1, (i + 1) % 3 + 1
            c, d = b + 3, a + 3
            f.write("f %d %d %d\n" % (a, b, c))
            f.write("f %d %d %d\n" % (a, c, d))
    return os.path.abspath(path)


def triangles_to_cylinders(items):
    """Fallback when triangular prisms cannot be created: replace each by a cylinder of similar size."""
    return [cyl(i["x"], i["y"], i["R"] * 0.85, i["h"]) if i["kind"] == "tri" else i for i in items]


def add_obstacle(scene, it, idx=0):
    """Add one fixed obstacle to a Genesis scene (must be called before scene.build)."""
    import genesis as gs
    h = it["h"]
    k = it["kind"]
    if k == "box":
        return scene.add_entity(gs.morphs.Box(pos=(it["x"], it["y"], h / 2.0), size=(it["sx"], it["sy"], h), fixed=True))
    if k == "cyl":
        return scene.add_entity(gs.morphs.Cylinder(pos=(it["x"], it["y"], h / 2.0), radius=it["r"], height=h, fixed=True))
    path = _write_tri_obj(it, idx)
    return scene.add_entity(gs.morphs.Mesh(file=path, pos=(it["x"], it["y"], 0.0), fixed=True))
