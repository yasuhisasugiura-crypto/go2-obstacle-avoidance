# -*- coding: utf-8 -*-
"""Creates go2_env_small.py from go2_env_goal.py (box size becomes obstacle_cfg['size']).
The original go2_env_goal.py is NOT modified."""
from pathlib import Path

src = Path("go2_env_goal.py").read_text(encoding="utf-8-sig")
old = "size=(0.4, 0.4, 0.5)"
new = 'size=(obstacle_cfg.get("size", 0.4), obstacle_cfg.get("size", 0.4), 0.5)'
if src.count(old) != 1:
    raise SystemExit("patch target not found exactly once: " + old)
Path("go2_env_small.py").write_text(src.replace(old, new), encoding="utf-8", newline="\n")
print("go2_env_small.py written")
