import math, random
from go2_shapes import box, cyl, tri, footprint_radius

GOAL = (12.0, 0.0)  # goal of the mixed courses

# ---- Course 1 (already run): 16 obstacles. The robot went around the outside. ----
MIXED_LONG = [
    box(1.0, 0.0, 0.15),
    cyl(1.9, 0.45, 0.25),
    tri(2.7, -0.40, 0.35, 90),
    box(3.5, 0.35, 0.50),
    cyl(4.3, -0.40, 0.08),
    tri(5.0, 0.30, 0.12, 0),
    box(5.8, -0.40, 0.50, 0.30),
    cyl(6.6, 0.40, 0.25),
    box(7.4, -0.35, 0.15),
    tri(8.2, 0.35, 0.35, 30),
    cyl(9.0, -0.40, 0.25),
    box(9.7, 0.40, 0.15),
    tri(10.3, -0.35, 0.12, 60),
    box(10.9, 0.30, 0.50),
    cyl(11.3, -0.30, 0.08),
]


def mirror(items):
    out = []
    for it in items:
        j = dict(it)
        j["y"] = -it["y"]
        if j["kind"] == "tri":
            j["yaw"] = -it["yaw"]
        out.append(j)
    return out


def random_layout(seed, n=16, goal=GOAL, x0=1.0, x1=None, ymax=1.1, sep=0.9, start_clear=1.0, goal_clear=1.5):
    rng = random.Random(seed)
    x1 = goal[0] - 1.2 if x1 is None else x1
    items, tries = [], 0
    while len(items) < n and tries < 20000:
        tries += 1
        k = rng.choice(["box", "cyl", "tri"])
        big = rng.random() < 0.5
        x = rng.uniform(x0, x1)
        y = rng.uniform(-ymax, ymax)
        if k == "box":
            it = box(round(x, 2), round(y, 2), 0.5 if big else 0.15, 0.3 if (big and rng.random() < 0.4) else None)
        elif k == "cyl":
            it = cyl(round(x, 2), round(y, 2), 0.25 if big else 0.08)
        else:
            it = tri(round(x, 2), round(y, 2), 0.35 if big else 0.12, rng.choice([0, 30, 60, 90]))
        if math.hypot(x, y) < start_clear or math.hypot(x - goal[0], y - goal[1]) < goal_clear:
            continue
        ok = True
        for o in items:
            d = math.hypot(o["x"] - it["x"], o["y"] - it["y"]) - footprint_radius(o) - footprint_radius(it)
            if d < sep:
                ok = False
                break
        if ok:
            items.append(it)
    items.sort(key=lambda i: i["x"])
    return items


# The previous submission course (10 small boxes, goal x = 8 m), kept so that its trajectory can be plotted.
LONG_MIRROR = [box(0.9, 0.0, 0.15)] + [box(round(1.6 + 0.7 * k, 2), 0.35 if k % 2 == 0 else -0.35, 0.15) for k in range(8)] + [box(7.3, 0.0, 0.15)]
GOALS = {"long_mirror": (8.0, 0.0)}

# ---- Course 2: "weave" courses, 22 obstacles each (boxes / cylinders / triangular prisms, small and large). ----
# Obstacles are arranged in columns that spread across the lateral (y) direction, and neighbouring
# columns are shifted left/right alternately. There is no one-sided path around them: the robot has
# to go BETWEEN obstacles. Level 1 = a little spread, level 2 = more, level 3 = the most.

WEAVE_1 = [  # level 1 (seed 3): 22 obstacles
    tri(1.59, -0.96, 0.35, 90),
    cyl(1.59, 0.44, 0.25),
    cyl(1.52, 1.74, 0.25),
    tri(2.78, -1.51, 0.35, 30),
    tri(2.69, -0.23, 0.12, 0),
    cyl(2.74, 0.84, 0.25),
    cyl(4.16, -0.79, 0.25),
    box(4.3, 0.41, 0.15, 0.15),
    cyl(4.15, 1.48, 0.25),
    tri(5.44, -1.68, 0.12, 60),
    box(5.36, -0.68, 0.5, 0.3),
    tri(5.35, 0.62, 0.35, 60),
    tri(6.7, -0.87, 0.35, 90),
    cyl(6.77, 0.58, 0.25),
    tri(6.69, 1.82, 0.12, 60),
    box(8.0, -1.88, 0.5, 0.5),
    tri(8.05, -0.47, 0.35, 90),
    tri(8.1, 1.05, 0.35, 90),
    cyl(9.45, -0.31, 0.08),
    cyl(9.31, 0.84, 0.25),
    box(10.36, -0.79, 0.5, 0.5),
    tri(10.46, 0.36, 0.12, 30),
]

WEAVE_2 = [  # level 2 (seed 3): 22 obstacles
    tri(1.4, -1.35, 0.35, 60),
    tri(1.35, 0.34, 0.35, 0),
    tri(1.27, 1.94, 0.35, 60),
    box(1.42, 3.36, 0.15, 0.15),
    cyl(3.46, -2.45, 0.08),
    tri(3.44, -1.31, 0.12, 90),
    tri(3.31, -0.21, 0.12, 0),
    box(3.37, 1.16, 0.15, 0.15),
    box(5.21, -1.51, 0.15, 0.15),
    cyl(5.11, -0.19, 0.08),
    box(5.28, 1.23, 0.5, 0.5),
    cyl(5.24, 2.75, 0.08),
    box(7.01, -2.95, 0.5, 0.3),
    box(7.01, -1.54, 0.5, 0.3),
    box(7.01, 0.02, 0.5, 0.5),
    cyl(7.14, 1.49, 0.08),
    tri(8.88, -1.28, 0.12, 60),
    cyl(8.91, -0.04, 0.25),
    box(8.83, 1.18, 0.15, 0.15),
    tri(8.83, 2.46, 0.12, 90),
    cyl(10.98, -1.4, 0.08),
    cyl(10.83, -0.25, 0.08),
]

WEAVE_3 = [  # level 3 (seed 4): 22 obstacles
    box(1.52, -2.15, 0.15, 0.15),
    tri(1.54, -0.62, 0.12, 0),
    box(1.56, 0.63, 0.15, 0.15),
    box(1.39, 2.12, 0.15, 0.15),
    cyl(1.56, 3.7, 0.25),
    cyl(3.94, -4.66, 0.08),
    cyl(3.87, -3.21, 0.08),
    cyl(3.88, -1.63, 0.25),
    tri(3.99, 0.15, 0.35, 90),
    tri(4.03, 2.2, 0.35, 0),
    tri(6.29, -2.22, 0.35, 60),
    box(6.3, -0.51, 0.5, 0.5),
    cyl(6.15, 1.03, 0.25),
    cyl(6.14, 2.76, 0.25),
    tri(6.3, 4.15, 0.12, 90),
    tri(8.47, -3.9, 0.35, 90),
    tri(8.37, -2.09, 0.12, 60),
    cyl(8.38, -0.57, 0.25),
    cyl(8.5, 0.92, 0.08),
    box(8.43, 2.27, 0.15, 0.15),
    box(10.82, -0.0, 0.15, 0.15),
    tri(10.78, 1.46, 0.35, 90),
]


LAYOUTS = {
    "mixed_long": MIXED_LONG,
    "mixed_long_mirror": mirror(MIXED_LONG),
    "long_mirror": LONG_MIRROR,
    "weave_1": WEAVE_1,
    "weave_2": WEAVE_2,
    "weave_3": WEAVE_3,
    "weave_1_mirror": mirror(WEAVE_1),
    "weave_2_mirror": mirror(WEAVE_2),
    "weave_3_mirror": mirror(WEAVE_3),
}
