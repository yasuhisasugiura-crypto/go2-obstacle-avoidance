import argparse
import csv
import math
import os
import pickle
import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env_small import Go2Env
from go2_simple_core import SimpleAvoider

GOAL = (4.0, 0.0)
GOAL_RADIUS = 0.3
WARMUP = 50

# 5 small boxes placed alternately left/right so that a straight run is blocked
# and the robot has to swing left <-> right to reach the goal.
SLALOM = [(0.9, 0.0), (1.6, -0.35), (2.3, 0.35), (3.0, -0.35), (3.5, 0.0)]
LAYOUTS = {
    "slalom": SLALOM,
    "slalom_mirror": [(x, -y) for x, y in SLALOM],
}


def box_clearance(px, py, bx, by, half):
    dx = max(abs(px - bx) - half, 0.0)
    dy = max(abs(py - by) - half, 0.0)
    return math.hypot(dx, dy)


def count_contacts(robot, box_entities):
    total = 0
    for b in box_entities:
        c = robot.get_contacts(with_entity=b)
        vm = c.get("valid_mask")
        if vm is not None:
            total += int(torch.as_tensor(vm).sum().item())
        else:
            total += int(torch.as_tensor(c["geom_a"]).numel())
    return total


def count_swings(traj, thr=0.12):
    """Number of left<->right direction reversals (lateral change > thr metres)."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-avoid-walk-v1")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--layout", choices=list(LAYOUTS.keys()), default="slalom")
    parser.add_argument("--box-size", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--r-robot", type=float, default=0.22)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--lookahead", type=float, default=0.9)
    parser.add_argument("--tan-half", type=float, default=128.0 / 96.0)
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 90.0

    boxes = LAYOUTS[args.layout]
    half = args.box_size / 2.0
    print("layout:", args.layout, "boxes:", boxes, "size:", args.box_size, flush=True)

    env = Go2Env(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=args.viewer, obstacle_cfg={"positions": boxes, "size": args.box_size},
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    env.reset()
    if args.viewer:
        try:
            env.scene.viewer.set_camera_pose(pos=(2.0, -4.0, 3.5), lookat=(2.0, 0.0, 0.0))
            env.scene.draw_debug_sphere(pos=(GOAL[0], GOAL[1], 0.1), radius=0.15, color=(0.0, 1.0, 0.0, 0.8))
        except Exception as exc:
            print("viewer decoration skipped:", exc, flush=True)

    av = SimpleAvoider(GOAL, radius=args.r_robot, margin=args.margin, lookahead=args.lookahead,
                       tan_half=args.tan_half)
    px = py = 0.0
    reached = fell = False
    steps_to_goal = -1
    path_len = 0.0
    min_clear = 99.0
    contact_ok = True
    contact_steps = 0
    first_contact = -1
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
                if (t - WARMUP) % 50 == 0:
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
                for bx, by in boxes:
                    min_clear = min(min_clear, box_clearance(nx, ny, bx, by, half))
                n_c = 0
                if contact_ok:
                    try:
                        n_c = count_contacts(env.robot, env.obstacles)
                    except Exception as exc:
                        contact_ok = False
                        print("contact check failed:", exc, flush=True)
                if n_c > 0:
                    contact_steps += 1
                    if first_contact < 0:
                        first_contact = t - WARMUP
                log_rows.append([t - WARMUP, round(nx, 3), round(ny, 3), round(vx, 3), round(wz, 3), n_c,
                                 info.get("reason", "")])
                if math.hypot(GOAL[0] - nx, GOAL[1] - ny) < GOAL_RADIUS:
                    reached = True
                    steps_to_goal = t - WARMUP
                    px, py = nx, ny
                    break
            px, py = nx, ny

    ys = [y for _, y in traj] or [0.0]
    row = {
        "layout": args.layout, "n_boxes": len(boxes), "box_size": args.box_size, "reached": reached,
        "fell": fell, "steps_to_goal": steps_to_goal, "path_len_m": round(path_len, 2),
        "min_clearance_m": round(min_clear, 3),
        "contact_steps": contact_steps if contact_ok else -1,
        "first_contact_step": first_contact if contact_ok else -1,
        "swings": count_swings(traj), "y_min": round(min(ys), 2), "y_max": round(max(ys), 2),
        "stall_events": stall_events, "r_robot": args.r_robot, "margin": args.margin,
        "final_x": round(px, 3), "final_y": round(py, 3),
    }
    print("RESULT", row, flush=True)
    out_path = "go2_slalom_results.csv"
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    with open(f"go2_slalom_log_{args.layout}_b{args.box_size}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "x", "y", "vx_cmd", "wz_cmd", "contacts", "reason"])
        w.writerows(log_rows)


if __name__ == "__main__":
    main()
