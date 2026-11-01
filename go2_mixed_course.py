import argparse
import csv
import math
import os
import pickle
import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env_mixed import Go2Env
from go2_simple_core import SimpleAvoider
import go2_shapes as shp
import go2_mixed_layouts as lay

GOAL_RADIUS = 0.3
WARMUP = 50


def contact_indices(robot, entities):
    """Indices of the obstacles that touch the robot at this step."""
    hit = []
    for i, e in enumerate(entities):
        c = robot.get_contacts(with_entity=e)
        vm = c.get("valid_mask")
        n = int(torch.as_tensor(vm).sum().item()) if vm is not None else int(torch.as_tensor(c["geom_a"]).numel())
        if n > 0:
            hit.append(i)
    return hit


def count_swings(traj, thr=0.12):
    n, last, ref = 0, 0, None
    for _, y in traj[::10]:
        if ref is None:
            ref = y
            continue
        if abs(y - ref) > thr:
            d = 1 if y > ref else -1
            if last and d != last:
                n += 1
            last = d
            ref = y
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--exp-name", type=str, default="go2-avoid-walk-v1")
    ap.add_argument("--ckpt", type=int, default=500)
    ap.add_argument("--layout", default="mixed_long", help="mixed_long / mixed_long_mirror / random")
    ap.add_argument("--seed", type=int, default=1, help="used only with --layout random")
    ap.add_argument("--n", type=int, default=16, help="number of obstacles (random layout)")
    ap.add_argument("--sep", type=float, default=0.9, help="min surface gap between obstacles (random layout)")
    ap.add_argument("--no-tri", action="store_true", help="replace triangular prisms by cylinders")
    ap.add_argument("--preview", action="store_true", help="build the scene and stand still for 200 steps (no walking)")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--follow", action="store_true", help="camera follows the robot")
    ap.add_argument("--r-robot", type=float, default=0.22)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--lookahead", type=float, default=0.9)
    ap.add_argument("--tan-half", type=float, default=128.0 / 96.0)
    args = ap.parse_args()

    if args.layout == "random":
        items = lay.random_layout(args.seed, n=args.n, sep=args.sep)
        name = "random_s%d" % args.seed
    else:
        items = lay.LAYOUTS[args.layout]
        name = args.layout
    if args.no_tri:
        items = shp.triangles_to_cylinders(items)
    goal = lay.GOALS.get(name, lay.GOAL)

    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print("layout:", name, "n:", len(items), "kinds:", kinds, "goal:", goal, flush=True)
    for it in items:
        print("  ", shp.label(it), "at", (it["x"], it["y"]), flush=True)

    with open("go2_mixed_layout_%s.csv" % name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "x", "y", "sx", "sy", "r", "R", "yaw", "h"])
        for it in items:
            w.writerow([it["kind"], it["x"], it["y"], it.get("sx", ""), it.get("sy", ""), it.get("r", ""),
                        it.get("R", ""), it.get("yaw", ""), it["h"]])

    gs.init(backend=gs.cpu, logging_level="warning")
    log_dir = "logs/%s" % args.exp_name
    with open("%s/cfgs.pkl" % log_dir, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 200.0

    env = Go2Env(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=args.viewer, obstacle_cfg={"items": items},
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, "model_%d.pt" % args.ckpt))
    policy = runner.get_inference_policy(device=gs.device)
    env.reset()
    gx = goal[0]
    if args.viewer:
        try:
            env.scene.viewer.set_camera_pose(pos=(gx / 2.0, -gx * 0.9, gx * 0.6), lookat=(gx / 2.0, 0.0, 0.0))
            env.scene.draw_debug_sphere(pos=(goal[0], goal[1], 0.1), radius=0.15, color=(0.0, 1.0, 0.0, 0.8))
        except Exception as exc:
            print("viewer decoration skipped:", exc, flush=True)

    if args.preview:
        with torch.no_grad():
            for t in range(200):
                env.commands[:] = torch.tensor([0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
                env._update_observation()
                env.step(policy(env.get_observations()))
        print("PREVIEW done (no walking). Check the viewer: shapes, sizes, positions.", flush=True)
        return

    av = SimpleAvoider(goal, radius=args.r_robot, margin=args.margin, lookahead=args.lookahead,
                       tan_half=args.tan_half)
    px = py = 0.0
    reached = fell = False
    steps_to_goal = -1
    path_len = 0.0
    min_clear = 99.0
    contact_ok = True
    contact_steps = 0
    first_contact = -1
    touched = set()
    stall_events = 0
    traj = []
    log_rows = []

    with torch.no_grad():
        for t in range(WARMUP + args.max_steps):
            vx = wz = 0.0
            info = {}
            if t >= WARMUP:
                fine, yaw = env.get_depth_sectors(n_sectors=128)
                fine = [float(v) for v in fine]
                av.observe(px, py, yaw, fine)
                vx, wz, _, info = av.decide(px, py, yaw)
                if info.get("reason") == "escape_start":
                    stall_events += 1
                if args.viewer and args.follow and (t - WARMUP) % 10 == 0:
                    try:
                        env.scene.viewer.set_camera_pose(pos=(px + 0.5, py - 3.5, 2.8), lookat=(px + 1.0, py, 0.2))
                    except Exception:
                        pass
                if (t - WARMUP) % 100 == 0:
                    print("STEP", t - WARMUP, "pos", round(px, 2), round(py, 2), info, flush=True)
            env.commands[:] = torch.tensor([vx, 0.0, wz], dtype=gs.tc_float, device=gs.device)
            env._update_observation()
            actions = policy(env.get_observations())
            _, _, dones, _ = env.step(actions)
            if bool(dones.any().item()):
                fell = True
                break
            pos = env.base_pos.reshape(-1, 3)[0]
            nx, ny = float(pos[0].item()), float(pos[1].item())
            if t >= WARMUP:
                path_len += math.hypot(nx - px, ny - py)
                traj.append((nx, ny))
                for it in items:
                    min_clear = min(min_clear, shp.distance(it, nx, ny))
                hit = []
                if contact_ok:
                    try:
                        hit = contact_indices(env.robot, env.obstacles)
                    except Exception as exc:
                        contact_ok = False
                        print("contact check failed:", exc, flush=True)
                if hit:
                    contact_steps += 1
                    if first_contact < 0:
                        first_contact = t - WARMUP
                    for i in hit:
                        touched.add(shp.label(items[i]) + " @(%.2f,%.2f)" % (items[i]["x"], items[i]["y"]))
                log_rows.append([t - WARMUP, round(nx, 3), round(ny, 3), round(vx, 3), round(wz, 3), len(hit),
                                 info.get("reason", "")])
                if math.hypot(goal[0] - nx, goal[1] - ny) < GOAL_RADIUS:
                    reached = True
                    steps_to_goal = t - WARMUP
                    px, py = nx, ny
                    break
            px, py = nx, ny

    ys = [y for _, y in traj] or [0.0]
    row = {
        "layout": name, "n_obstacles": len(items), "reached": reached, "fell": fell,
        "steps_to_goal": steps_to_goal, "path_len_m": round(path_len, 2),
        "min_clearance_m": round(min_clear, 3),
        "contact_steps": contact_steps if contact_ok else -1,
        "first_contact_step": first_contact if contact_ok else -1,
        "swings": count_swings(traj), "y_min": round(min(ys), 2), "y_max": round(max(ys), 2),
        "stall_events": stall_events, "no_tri": args.no_tri, "final_x": round(px, 3), "final_y": round(py, 3),
    }
    print("RESULT", row, flush=True)
    if touched:
        print("TOUCHED", sorted(touched), flush=True)
    out_path = "go2_mixed_results.csv"
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    with open("go2_mixed_log_%s.csv" % name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "x", "y", "vx_cmd", "wz_cmd", "contacts", "reason"])
        w.writerows(log_rows)


if __name__ == "__main__":
    main()
