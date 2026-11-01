import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import go2_shapes as shp
import go2_mixed_layouts as lay

# Japanese font (avoid tofu): the first one found on Windows is used
matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def f(v):
    return float(v) if v not in ("", None) else None


def load_items_csv(path):
    items = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "box":
                items.append(shp.box(f(r["x"]), f(r["y"]), f(r["sx"]), f(r["sy"]), f(r["h"])))
            elif r["kind"] == "cyl":
                items.append(shp.cyl(f(r["x"]), f(r["y"]), f(r["r"]), f(r["h"])))
            else:
                items.append(shp.tri(f(r["x"]), f(r["y"]), f(r["R"]), f(r["yaw"]), f(r["h"])))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="trajectory CSV (step,x,y,...)")
    ap.add_argument("--layout", help="name in go2_mixed_layouts.LAYOUTS (mixed_long, long_mirror, ...)")
    ap.add_argument("--layout-csv", help="go2_mixed_layout_*.csv written by go2_mixed_course.py")
    ap.add_argument("--goal-x", type=float, default=None)
    ap.add_argument("--out", default="traj.png")
    ap.add_argument("--title", default="走行軌跡")
    a = ap.parse_args()

    if a.layout_csv:
        items = load_items_csv(a.layout_csv)
    else:
        items = lay.LAYOUTS[a.layout]
    goal = (a.goal_x, 0.0) if a.goal_x else lay.GOALS.get(a.layout, lay.GOAL)

    xs, ys = [], []
    with open(a.log, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            xs.append(float(r["x"]))
            ys.append(float(r["y"]))

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    for it in items:
        if it["kind"] == "box":
            ax.add_patch(Rectangle((it["x"] - it["sx"] / 2, it["y"] - it["sy"] / 2), it["sx"], it["sy"],
                                   fc="#c0392b", ec="k", lw=0.6, zorder=3))
        elif it["kind"] == "cyl":
            ax.add_patch(Circle((it["x"], it["y"]), it["r"], fc="#2e86c1", ec="k", lw=0.6, zorder=3))
        else:
            ax.add_patch(Polygon(shp.tri_vertices(it), closed=True, fc="#27ae60", ec="k", lw=0.6, zorder=3))
    ax.plot(xs, ys, "-", color="#222222", lw=1.4, zorder=4, label="走行軌跡")
    ax.plot([0], [0], "o", color="#1f77b4", ms=8, zorder=5)
    ax.plot([goal[0]], [goal[1]], "*", color="#2ca02c", ms=14, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, goal[0] + 0.6)
    lim = max([abs(v) for v in ys] + [abs(it["y"]) + 0.5 for it in items] + [0.9])
    ax.set_ylim(-lim - 0.2, lim + 0.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(a.title)
    ax.grid(alpha=0.25)
    # legend with shapes
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="#222222", lw=1.4, label="走行軌跡"),
                       Line2D([0], [0], marker="s", color="w", mfc="#c0392b", mec="k", ms=8, label="箱"),
                       Line2D([0], [0], marker="o", color="w", mfc="#2e86c1", mec="k", ms=8, label="円柱"),
                       Line2D([0], [0], marker="^", color="w", mfc="#27ae60", mec="k", ms=8, label="三角柱"),
                       Line2D([0], [0], marker="o", color="w", mfc="#1f77b4", ms=8, label="スタート"),
                       Line2D([0], [0], marker="*", color="w", mfc="#2ca02c", ms=12, label="ゴール")],
              loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
