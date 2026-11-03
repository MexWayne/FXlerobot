#!/usr/bin/env python
"""Figure for cup_compare.py: lidar vs head stereo on two paper cups. Needs matplotlib (run on the PC).

    python plot_cup_compare.py <snapshot dir with cups.json and left.jpg> out.png [--truth-cm 12]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

for f in ("/mnt/c/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#52524e", "#8a8a85", "#e6e6e3"
VISION, LIDAR = "#2a78d6", "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--truth-cm", type=float, default=12.0)
    args = ap.parse_args()
    snap = Path(args.snapshot)
    d = json.loads((snap / "cups.json").read_text())
    cups = d["cups"]
    o = np.array(d["lidar_origin_base_m"])
    sep = {k: v * 100 for k, v in d["separation_m"].items()}

    fig = plt.figure(figsize=(17, 10.4), facecolor=SURFACE)
    fig.text(0.012, 0.965, "纸杯定位：头部双目 vs 激光雷达", fontsize=22, color=INK)
    fig.text(0.012, 0.932, f"两个倒扣纸杯放在 40 mm ChArUco 板上，中心相距 3 格 = {args.truth_cm:g} cm；"
             f"脖子 yaw {d['neck_deg']['neck_yaw']:.1f}°，pitch {d['neck_deg']['neck_pitch']:.1f}°。"
             "双目：两眼杯底“+”号三角化；雷达：8 圈回波的杯壁弧线拟合圆心。",
             fontsize=13, color=INK2)

    # Left: left-eye image with the refined crosses.
    ax = fig.add_axes([0.012, 0.30, 0.47, 0.60])
    img = plt.imread(snap / "left.jpg")
    ax.imshow(img[60:400, 150:640], extent=(150, 640, 400, 60))
    worst = max(max(c["reproj_px"]) for c in cups)
    ax.set_title(f"头部左眼（右眼同样找到两个“+”号，三角化重投影误差 ≤ {worst:.1f} px）",
                 loc="left", fontsize=13, color=INK2, pad=8)
    for i, c in enumerate(cups):
        u, v = c["uv_left"]
        ax.plot(u, v, "o", ms=16, mfc="none", mec=VISION, mew=2.2)
        ax.annotate(f"杯{i + 1}", (u, v), xytext=(u - 35 + 70 * i, v - 55), fontsize=14, color=INK,
                    ha="center", arrowprops=dict(arrowstyle="-", color=VISION, lw=1.6),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=VISION, lw=1.2))
    ax.axis("off")

    # Right: top view in the base frame.
    ax = fig.add_axes([0.53, 0.30, 0.45, 0.585], facecolor=SURFACE)
    env = np.array(d["lidar_env_points"])
    ax.scatter(env[:, 1], env[:, 0], s=4, c="#b9b9b4", lw=0, label="其他雷达回波（左臂、车身等）")
    for i, c in enumerate(cups):
        p = np.array(c["lidar_points"])
        lc, s = np.array(c["lidar_centre_m"]), np.array(c["stereo_base_m"][:2])
        ax.scatter(p[:, 1], p[:, 0], s=7, c=LIDAR, lw=0, label="杯壁雷达回波（8 圈）" if i == 0 else None)
        ax.add_patch(Circle((lc[1], lc[0]), c["lidar_radius_m"], fill=False, ec=LIDAR, lw=1.6, ls="--",
                            label="雷达拟合圆" if i == 0 else None))
        ax.plot(s[1], s[0], "D", ms=8, color=VISION, mec="white", mew=1.2,
                label="双目：杯中心（底盘坐标）" if i == 0 else None)
        # Hollow and on top: the two centres nearly coincide.
        ax.plot(lc[1], lc[0], "o", ms=15, mfc="none", mec=LIDAR, mew=2.2,
                label="雷达：杯中心" if i == 0 else None)
        for q, col in ((lc, LIDAR), (s, VISION)):
            ax.plot([o[1], q[1]], [o[0], q[0]], color=col, lw=1, alpha=0.55)
        r = c["range_from_lidar_m"]
        ax.annotate(f"杯{i + 1}\n雷达 {r['lidar'] * 100:.1f} cm\n双目 {r['stereo'] * 100:.1f} cm",
                    (lc[1], lc[0]), xytext=(lc[1] + (0.07 if i == 0 else -0.07), lc[0] - 0.075),
                    fontsize=12, color=INK, ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    l0, l1 = (np.array(c["lidar_centre_m"]) for c in cups)
    mid = (l0 + l1) / 2
    ax.annotate("", (l1[1], l1[0]), (l0[1], l0[0]), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.plot(o[1], o[0], "+", ms=18, mew=2, color=INK)
    ax.text(o[1] - 0.012, o[0] - 0.012, f"雷达原点\n（底盘中心前 {o[0] * 100:.0f} cm、{'右' if o[1] < 0 else '左'} {abs(o[1]) * 100:.1f} cm）", fontsize=11, color=INK2, va="top")
    ax.text(mid[1], mid[0] + 0.045, f"杯距  雷达 {sep['lidar']:.1f} cm  ·  双目 {sep['stereo_3d']:.1f} cm",
            fontsize=12, color=INK, ha="center", va="bottom")
    ax.set_xlim(0.22, -0.22)
    ax.set_ylim(0.05, 0.46)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.set_xlabel("y（m，向左为正）", color=INK2, fontsize=12)
    ax.set_ylabel("x（m，向前为正）", color=INK2, fontsize=12)
    ax.set_title("俯视图（底盘坐标）", loc="left", fontsize=13, color=INK2, pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.22), fontsize=10.5, frameon=True, facecolor="white", edgecolor=GRID)

    # Bottom: the numbers.
    diff = [(np.array(c["lidar_centre_m"]) - c["stereo_base_m"][:2]) * 1000 for c in cups]
    r = [c["range_from_lidar_m"] for c in cups]
    rows = [
        ("杯距（真值 {:g} cm）".format(args.truth_cm),
         f"雷达 {sep['lidar']:.1f} cm（{sep['lidar'] - args.truth_cm:+.1f}）    "
         f"双目 {sep['stereo_3d']:.1f} cm（{sep['stereo_3d'] - args.truth_cm:+.1f}）"),
        ("杯到雷达的水平距离",
         f"杯1  雷达 {r[0]['lidar'] * 100:.1f} cm / 双目 {r[0]['stereo'] * 100:.1f} cm（差 {(r[0]['stereo'] - r[0]['lidar']) * 1000:+.0f} mm）    "
         f"杯2  雷达 {r[1]['lidar'] * 100:.1f} cm / 双目 {r[1]['stereo'] * 100:.1f} cm（差 {(r[1]['stereo'] - r[1]['lidar']) * 1000:+.0f} mm）"),
        ("杯中心位置差（雷达−双目）",
         f"杯1  x {diff[0][0]:+.0f} mm, y {diff[0][1]:+.0f} mm    杯2  x {diff[1][0]:+.0f} mm, y {diff[1][1]:+.0f} mm"
         f"（雷达按中线右侧 {abs(o[1]) * 1000:.0f} mm 安装换算；不加这个偏移时 y 差 22–26 mm）"),
        ("雷达拟合",
         f"杯壁半径 {cups[0]['lidar_radius_m'] * 1000:.1f} / {cups[1]['lidar_radius_m'] * 1000:.1f} mm，"
         f"残差 {cups[0]['lidar_fit_rms_m'] * 1000:.1f} / {cups[1]['lidar_fit_rms_m'] * 1000:.1f} mm，"
         f"每杯 {len(cups[0]['lidar_points'])} / {len(cups[1]['lidar_points'])} 个点（只扫到朝向雷达的约 1/4 圈）"),
    ]
    y = 0.215
    for k, v in rows:
        fig.text(0.02, y, k, fontsize=14, color=INK2)
        fig.text(0.215, y, v, fontsize=14, color=INK)
        y -= 0.047
    fig.savefig(args.out, dpi=110, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
