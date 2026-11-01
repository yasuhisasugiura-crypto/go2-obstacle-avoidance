import argparse
import csv
import math
import os
import pickle
import random
import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env_small import Go2Env
from go2_simple_core import SimpleAvoider, wrap

GOAL = (4.0, 0.0)
GOAL_RADIUS = 0.3
WARMUP = 50


def make_random_layout(seed, n, gap, goal_clear, start_clear):
    rng = random.Random(seed)
    boxes, tries = [], 0
    while len(boxes) < n and tries < 1000:
        tries += 1
        x = rng.uniform(0.9, 3.4)
        y = rng.uniform(-0.8, 0.8)
        if math.hypot(x - GOAL[0], y - GOAL[1]) < goal_clear:
            continue
        if math.hypot(x, y) < start_clear:
            continue
        if all(math.hypot(x - bx, y - by) >= gap for bx, by in boxes):
            boxes.append((round(x, 2), round(y, 2)))
    return boxes


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


def calib(fine, px, py, yaw, boxes, half):
    """Compare two camera models against the TRUE box positions (no walking)."""
    n = len(fine)
    for name, th in (("tan_half=1.000 (horizontal 90deg)", 1.0), ("tan_half=1.333 (vertical 90deg, 128x96)", 128.0 / 96.0)):
        errs = []
        for i, d in enumerate(fine):
            if not math.isfinite(d) or d <= 0.05 or d >= 3.0:
                continue
            a = math.atan((1.0 - 2.0 * (i + 0.5) / n) * th)
            fwd, lat = d, d * math.tan(a)
            cx = px + 0.3 * math.cos(yaw)
            cy = py + 0.3 * math.sin(yaw)
            wx = cx + fwd * math.cos(yaw) - lat * math.sin(yaw)
            wy = cy + fwd * math.sin(yaw) + lat * math.cos(yaw)
            errs.append(min(box_clearance(wx, wy, bx, by, half) for bx, by in boxes))
        if errs:
            print("CALIB", name, "points", len(errs), "mean_err_m", round(sum(errs) / len(errs), 3),
                  "max_err_m", round(max(errs), 3), flush=True)
        else:
            print("CALIB", name, "no points (no box in view)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-avoid-walk-v1")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-boxes", type=int, default=3)
    parser.add_argument("--box-size", type=float, default=0.15)
    parser.add_argument("--box-gap", type=float, default=1.2)
    parser.add_argument("--goal-clear", type=float, default=1.3)
    parser.add_argument("--start-clear", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--r-robot", type=float, default=0.22)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--lookahead", type=float, default=0.9)
    parser.add_argument("--tan-half", type=float, default=128.0 / 96.0)
    parser.add_argument("--calib", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = 60.0

    boxes = make_random_layout(args.seed, args.n_boxes, args.box_gap, args.goal_clear, args.start_clear)
    half = args.box_size / 2.0
    print("boxes:", boxes, "size:", args.box_size, flush=True)

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
    log_rows = []

    with torch.no_grad():
        for t in range(WARMUP + args.max_steps):
            vx = wz = 0.0
            info = {}
            if t >= WARMUP:
                fine, yaw = env.get_depth_sectors(n_sectors=128)
                fine = [float(v) for v in fine]
                if args.calib:
                    calib(fine, px, py, yaw, boxes, half)
                    return
                av.observe(px, py, yaw, fine)
                vx, wz, _, info = av.decide(px, py, yaw)
                if info.get("reason") == "escape_start":
                    stall_events += 1
                if (t - WARMUP) % 25 == 0:
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

    row = {
        "seed": args.seed, "n_boxes": args.n_boxes, "box_size": args.box_size, "reached": reached,
        "fell": fell, "steps_to_goal": steps_to_goal, "path_len_m": round(path_len, 2),
        "min_clearance_m": round(min_clear, 3),
        "contact_steps": contact_steps if contact_ok else -1,
        "first_contact_step": first_contact if contact_ok else -1,
        "stall_events": stall_events, "r_robot": args.r_robot, "margin": args.margin,
        "tan_half": round(args.tan_half, 3), "final_x": round(px, 3), "final_y": round(py, 3),
    }
    print("RESULT", row, flush=True)
    out_path = "go2_simple_results.csv"
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    with open(f"go2_simple_log_s{args.seed}_n{args.n_boxes}_b{args.box_size}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "x", "y", "vx_cmd", "wz_cmd", "contacts", "reason"])
        w.writerows(log_rows)


if __name__ == "__main__":
    main()
