# -*- coding: utf-8 -*-
"""Creates go2_env_mixed.py from go2_env_goal.py.
Obstacles are now created by go2_shapes.add_obstacle (box / cylinder / triangular prism).
The original go2_env_goal.py is NOT modified."""
from pathlib import Path

src = Path("go2_env_goal.py").read_text(encoding="utf-8-sig")
lines = src.splitlines()

# 1) import line after "import genesis as gs"
imp = [i for i, l in enumerate(lines) if l.strip() == "import genesis as gs"]
if len(imp) != 1:
    raise SystemExit("'import genesis as gs' not found exactly once")

# 2) obstacle loop (4 lines)
loop = [i for i, l in enumerate(lines) if l.strip() == 'for ox, oy in obstacle_cfg["positions"]:']
if len(loop) != 1:
    raise SystemExit("obstacle loop not found exactly once")
i = loop[0]
if lines[i + 1].strip() != "self.obstacles.append(" or lines[i + 3].strip() != ")":
    raise SystemExit("obstacle loop has an unexpected shape; send go2_env_goal.py lines %d-%d" % (i, i + 4))
indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
new_loop = [
    indent + 'for idx, item in enumerate(obstacle_cfg["items"]):',
    indent + "    self.obstacles.append(add_obstacle(self.scene, item, idx))",
]
out = lines[: imp[0] + 1] + ["from go2_shapes import add_obstacle"] + lines[imp[0] + 1 : i] + new_loop + lines[i + 4 :]
Path("go2_env_mixed.py").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("go2_env_mixed.py written")
