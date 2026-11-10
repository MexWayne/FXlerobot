#!/usr/bin/env python
"""Pick a small thing (a scrap of paper, a plastic chip) off the floor with the left arm. The base never moves.

    python pick_small.py calib          # once: a flat card strip in front; where the wrist camera
                                        # must see a thing from TIP_ALIGN_M up (wrist_target.json)
    python pick_small.py jawzone        # where the jaws are in the wrist camera, from the wrist photos
                                        # under runs/ (wrist_jaw_zone.png); no arm, no lock
    python pick_small.py                # look, then pick it up: all in one go
    python pick_small.py look           # neck to the grasp view, photos, find it, plan; the arm parks only
    python pick_small.py check          # jaws open to TIP_PRE_M above it and back: photos to see the aim
    python pick_small.py grab           # pick up what the last look found (same neck pose)
    python pick_small.py grab --dx 5    # aim this much further (mm, base frame) than the last look found
    python pick_small.py place          # put what it holds back where the last pick took it from, park
    python pick_small.py place --dx 20 --dy -30 --turn 45   # ... 2 cm further, 3 cm right, turned 45 deg
    python pick_small.py drop           # open the jaws where the arm is
    python pick_small.py park           # the rest pose
    python pick_small.py all            # every small thing in reach, nearest the arm first: pick it up,
                                        # hold it up, put it back where it lay, on to the next

Two things matter, aim and grip; the thing's height does not (the user, 2026-11-04). The grasp is the
user's: find the thing's straight edge, bring the FIXED jaw's tip as near the thing as it goes with
its face along that edge, and close, so that the moving jaw pushes the thing onto the fixed one.
The tool comes down (upright, or leaning the least of TILT_STEPS_DEG that reaches) with the jaws open
OPEN_GAP_M to TIP_ALIGN_M above the floor, above what lies about. The head camera's aim (a few mm to
a centimetre off) is finished by the left wrist camera, which sees the open jaws and the floor
between them. How a floor point's pixel moves per mm of the arm `calib` measured once in the
gripper's frame; turned with the jaws (wrist_J) it brings the outline's straight edges into the base
frame, and the wrist turns so the fixed jaw lies along the longest one it reaches, on its side
(choose_edge). Then the middle of the thing's side facing the
fixed jaw is brought EDGE_GAP_MM off the fixed jaw's face, which `calib` measured (the fixed jaw
does not move: its face is where the closed tips meet, seen from TIP_ALIGN_M up), moving by what is
left up to ALIGN_TRIES times; steps longer than DAMP_FROM_MM go ALIGN_DAMP of the way (the px/mm
grows nearer the jaws: a 35 mm step overshot by 12). Of several things in view the one followed is
the one nearest where it should be. Things are found by detection (the VLM's boxes, cut out by
GrabCut inside them), in the head camera and the wrist camera alike, never by colour thresholds.
History: aimed at the middle between the jaws, by a target taken from long strips, an 18 mm block
lay beside the jaws when they closed; aimed 6 mm up, the probing moves pushed a thick block about; a second, 1 mm up step to
bring the fixed jaw up to the edge measured 0.3-2 px/mm where ~5 are (the fixed jaw hides the thing's
near side there) and pushed the jaw 4-6 mm off instead (2026-11-04).
Then the fixed-jaw tip goes down to TIP_PRESS_M and the jaws close all the way (close()): LeftArm.grip
stops where the jaw first meets resistance, and a scrap half in the carpet's pile resists being pushed,
so the jaw stopped on it 15 mm short of the fixed jaw and nothing was held (2026-11-04, second try).
They close in CLOSE_STEP_DEG steps: watched step by step, a 46 mm card strip was caught by one end,
swung round and straightened between the tips, and came up; closed in one sweep it slipped out.
The tips go below the floor
the kinematics put there: told 1 mm up, the closed jaws passed over a 3 mm thick scrap (third try),
so the real tips stand a few mm higher than the kinematics and arm_mount.yaml say. The moving jaw
swings about a pivot 81 mm above the tips: half closed its tip is ~6 mm lower than the fixed one (the
URDF pivot, the tips meeting when closed), so it may dig into the carpet and stop half way. When it
stops more than JAW_STUCK_DEG short of closed, the tool rises RISE_M and closes on, at most
RISE_TRIES times. The jaws open only OPEN_GAP_M for this (the user, 2026-11-04): the carpet gives,
and the wider they open, the longer the moving jaw's sweep and the deeper it digs in on the way
(full open is ~9 cm between the tips). The jaws close almost fully on paper, so whether it
came up is told by the camera: after the arm parks, the thing is not found round where it lay.

Into runs/<time>/: photos, look_left_detect.jpg (what was found), *_marked.jpg (the planned fixed-jaw
tip in red, the grasp point in yellow and the closing line drawn into the picture), run.json.
"""

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LLM_VLA = HERE.parent
sys.path[:0] = [str(HERE), str(LLM_VLA / "pick_bottle"), str(LLM_VLA), str(LLM_VLA.parent / "project_tool"),
                str(LLM_VLA.parent / "vln")]
import detect_small  # noqa: E402
import ik  # noqa: E402
import mount  # noqa: E402
import pick  # noqa: E402
import so101_kin  # noqa: E402
from arm import LeftArm, joint_limits  # noqa: E402
from arm_snapshot import read_neck_deg  # noqa: E402
from neck_look import move_neck  # noqa: E402
from touch_calib import grab  # noqa: E402

RUNS = HERE / "runs"
GRASP_NECK = (-15.0, 35.0)  # pick.ROBOT_MASK is drawn for this head pose, the arm in PARK
NECK_TOL_DEG = 1.5
WRIST_CAM = "/dev/cam_wrist_left"
# The wrist camera stays open through a grasp (wrist_release() after): opened for every photo it took
# ~1.2 s a photo (15 frames for the exposure to settle), 7-8 photos a grasp; kept open ~0.2 s, the
# frames the driver queued while the arm moved (4 buffers) dropped first (2026-11-06).
WRIST_SETTLE_FRAMES = 15
WRIST_STALE_FRAMES = 5
# The thing goes between the jaws this far from the fixed jaw along the closing line: the moving jaw
# sweeps it to the fixed one, and an aim a centimetre off either way still has it between them.
FROM_FIXED_M = 0.018
HOVER_M = 0.06       # fixed-jaw tip this high over the thing first
TIP_PRE_M = 0.010    # `place` lets go this high
# The wrist camera aims the jaws from here: above what lies about, so the probing moves touch nothing
# (at 6 mm the jaws pushed a green connector block about while probing, 2026-11-04).
TIP_ALIGN_M = 0.015
TIP_LOW_M = 0.001    # `calib`: a flat strip is looked at from here and from TIP_ALIGN_M
TIP_PRESS_M = -0.003  # the fixed-jaw tip closes this high by the kinematics (really a few mm higher)
TIP_FLOOR_M = -0.006  # no move is planned with the tip lower than this
LIFT_M = 0.10
PRESS_DEG_S = 6.0
# Left wrist camera, tool upright: it looks along the jaws from the side and above. Floor along the
# jaws' faces runs left-right in the picture, the closing line up-down; the white plates on the right
# are the jaws going up from their tips. Where a thing must show from TIP_ALIGN_M up is measured by
# `calib` from where the closed tips show (TIP_PX leftmost pixels of each jaw). A target taken from a
# 46 mm strip aimed "by eye" at 6 mm (8 picks of 8) was ~80 px beside the tips along the jaws: long
# strips came up anyway, an 18 mm plastic block lay beside the jaws when they closed (2026-11-04).
TARGET_FILE = HERE / "wrist_target.json"
LOCK_FILE = HERE / ".pick_small.lock"
WRIST_TOL_PX = 8.0
WRIST_GIVE_UP_PX = 25.0  # still this far off after ALIGN_TRIES: no grasp
# After an aiming move the thing is cut out by GrabCut round where its last outline should have moved
# (TRACK_PAD_PX more each way), not boxed by the VLM again (~3 s a step, 3-4 steps a grasp). Taken only
# when its size stays within TRACK_AREA of the last and its middle within TRACK_NEAR_PX of where it
# should be; else the VLM boxes it as before (2026-11-06).
TRACK_PAD_PX = 12
TRACK_AREA = (0.8, 1.25)
TRACK_NEAR_PX = 40.0
STUCK_EXPECT_PX = 20.0  # a move that should shift the thing this far in the picture ...
STUCK_SHARE = 0.25      # ... and shifted it less than this share of it: it is on the jaws
TIP_PX = 40
# Only a long thing keeps its measured direction while planning; a short one's jaws' direction is
# tried round AXIS_TRIES_DEG until the arm reaches, and in any case the wrist camera then turns the
# jaws onto a straight edge of it (choose_edge). Nor a wide one (from_fixed): a milk carton's middle out
# of upright reach was planned with the jaws turned 60 deg, closing on its corners (2026-11-07); it
# is gripped off its middle instead (GRIP_ALONG_MIN_M).
LONG_MIN_M = 0.025
LONG_RATIO = 2.0
AXIS_TRIES_DEG = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0)
# The user's way (2026-11-04): find the thing's straight edge, bring the FIXED jaw's tip as near the
# thing as it goes, its face along that edge, and close: the moving jaw then pushes the thing onto the
# fixed one. So the point followed in the wrist camera is the middle of the thing's side nearest the
# fixed jaw, and its goal is EDGE_GAP_MM off the fixed jaw's face towards the moving jaw. The fixed jaw
# does not move: its face is where the closed tips meet (`calib`).
EDGE_GAP_MM = 2.0
EDGE_SIDE_SHARE = 0.1  # the thing's pixels this far out towards the fixed jaw are its side facing it
# The wrist turned past this range pulled the wrist camera's cable loose (2026-11-04: picks between
# -40 and -150 deg were fine; one leaning to +92 and the camera was gone): solutions outside it are
# taken the other way round, or not at all.
WRIST_ROLL_RANGE = (-180.0, 0.0)
# Out of upright reach the tool may lean, the least that reaches: the jaws still close level, so both
# tips stay at one height. Things 25-26 cm from the arm base needed 8-9 deg (2026-11-04); pick_bottle
# takes caps up to 12.
TILT_STEPS_DEG = (0.0, 4.0, 8.0, 12.0)
# Beyond this from the arm base (horizontal) nothing is tried: a scrap 54 cm out took a minute of IK
# to be refused, the body standing still (2026-11-07).
REACH_MAX_M = 0.32
# A thing at least this long (a milk carton, any box-like thing) can be gripped anywhere along it (user,
# 2026-11-07): when its middle is out of reach the grasp point moves along it towards the arm (the
# body's side), GRIP_STEP_M at a time, up to GRIP_END_M from its end (the jaws' faces still on it).
# REACH_MAX_M counts from the nearest of these points. The wrist camera aims for the same point.
GRIP_ALONG_MIN_M = 0.05
GRIP_STEP_M = 0.01
GRIP_END_M = 0.02
SEEDS = 6
AIM_TILT_MAX_DEG = 8.0
# In a wrist photo the VLM's box nearest where the thing should be, within this, is the thing.
VLM_NEAR_PX = 150.0
SHIFT_SEARCH_PX = 120  # a probing move shifts the thing at most this far in the wrist picture
SHIFT_MIN_SCORE = 0.9
# Jaws close best on a straight edge of the thing (the user, 2026-11-04): from the first wrist photo the
# longest straight stretch of its outline (a polygon fit within EDGE_FIT_PX, at least EDGE_MIN_PX long,
# not along the jaws or the picture's border) is turned into the base frame by the measured px/mm, and
# when the jaws' faces lie more than EDGE_TURN_DEG off it the wrist turns so they lie along it.
EDGE_FIT_PX = 2.5
EDGE_MIN_PX = 25.0
EDGE_BORDER_PX = 6
EDGE_TURN_DEG = 15.0
EDGE_KEEP_SHARE = 0.5  # an edge the jaws already suit is kept when at least this share of the longest
JAW_L_MIN = 170.0
JAW_CHROMA_MAX = 10.0
JAW_MIN_PX = 3000
# Where the jaws lie in the wrist picture (the camera rides on the fixed jaw): pixels jaw-coloured in at
# least JAW_ZONE_SHARE of the saved wrist photos, grown by JAW_ZONE_PAD_PX (`jawzone`). Only there is a
# light colourless thing the jaws: a white paper ball touching the fixed jaw went in with it, its outline
# shrank to a few shadowed scraps, the aim jumped 43-91 px for 4 tries and the ball was not gripped
# though it lay between the jaws from the first photo (2026-11-06).
JAW_ZONE_FILE = HERE / "wrist_jaw_zone.png"
JAW_ZONE_SHARE = 0.3
JAW_ZONE_PAD_PX = 9
PROBE_MM = 8.0
ALIGN_TRIES = 4
ALIGN_DAMP = 0.7
DAMP_FROM_MM = 10.0
ALIGN_MAX_MM = 40.0  # the wrist camera's correction is not trusted beyond this
JAW_STUCK_DEG = 12.0
CLOSE_DEG_S = 20.0   # the moving jaw sweeps the thing across the carpet to the fixed jaw this fast
CLOSE_STEP_DEG = 8.0
OPEN_GAP_M = 0.04    # between the jaw tips, open, for a small thing
OPEN_MARGIN_M = 0.02  # a wider thing gets the jaws its width and this much apart, ...
OPEN_MAX_M = 0.09     # ... up to this, full open (was 8 cm: a milk carton measured 7.4 cm wide was skipped, 2026-11-07)
# A wider thing (wider than OPEN_GAP_M less OPEN_MARGIN_M) goes down with its middle between the open
# jaws, one jaw on each side of it, not FROM_FIXED_M off the fixed jaw: that put the fixed jaw on top of a
# milk carton, at the 15 mm aiming height and again pressing down, and dented it (user, 2026-11-07).
# Measured by its box only (no outline) its width is not trusted, the jaws open OPEN_MAX_M: the same
# carton was 4.4 cm by its box, 6.8 by its outline.
# A thing at least this wide (a milk carton) is gripped where the head camera aimed when the wrist
# camera does not find it: close up it fills the wrist picture, the VLM boxed a scrap beside it and the
# grasp was given up with the open jaws right round the carton, which then closed and lifted it (user,
# 2026-11-07).
WIDE_M = 0.05
# So is a thing that fills the wrist picture (its box at least WRIST_FILL_SHARE of it and on two of its
# borders), whatever its width from the head camera: no edge of it is seen to aim at. A milk carton
# 15 mm under the tips spilled out at the top and right, its clipped box hardly moved when the arm
# moved 21 mm and the grasp was given up as "stuck on the jaws" (2026-11-07; box 0.64 of it, a
# cardboard scrap with a long tail 0.41 and its edges in view).
WRIST_FILL_SHARE = 0.5
# One cut off by the picture's border on the side facing the fixed jaw over at least WRIST_CUT_SPAN_PX
# (its outline along that border, else its box's) does not show the side aimed at: the arm goes that
# way CUT_STEP_MM at a time (CUT_TRIES more aiming steps, CUT_MAX_MM in all) until it does, then aims
# as usual; still cut off, it is gripped where the head camera aimed. A clear snack bag the head
# camera measured by its 6.7 cm label ran out of the picture there over ~270 px and the aim asked
# for 54 mm, refused; gripped where the head camera aimed, the fixed jaw came down on the bag and the
# jaws closed over it (2026-11-07). A cardboard scrap whose ~20 px tail ran out stays aimed.
WRIST_CUT_SPAN_PX = 60
CUT_STEP_MM = 20.0
CUT_TRIES = 2
CUT_MAX_MM = 60.0
# A thing wider than OPEN_GAP_M (a box, a paper ball) is lifted this far out from the body too (base
# +x) as it goes up: a milk carton swung against the bumper on the way up and fell (user, 2026-11-07).
# Planned after the grasp, from its lift: as one of the grasp's steps it turned the jaws' direction and
# the wrist (-3 -> -153 deg) to reach out, and the grasp itself went wrong (user, 2026-11-07).
LIFT_OUT_M = 0.06
# Held, the jaws may turn a little on the way out: one at wrist_roll -3 could not keep its direction
# (the wrist would have gone past 0 and round), so these turns are tried, the least first.
LIFT_OUT_TURNS_DEG = (0.0, 10.0, -10.0, 20.0, -20.0)
EXPECT_NEAR_M = 0.15  # look(expect=...): a thing farther than this from where a caller expects it is another
OPEN_FIT_M = 0.01     # a thing wider than OPEN_MAX_M less this does not fit between the open jaws: skipped
                      # (an 11.8 cm plastic wrapper was tried twice, 3 min, 2026-11-05)
# The moving jaw turns about this point of gripper_link (URDF "gripper" joint, x and z) and its tip
# meets the fixed-jaw tip when closed, so the tips are 2 R sin(angle from closed / 2) apart.
JAW_PIVOT_XZ = (0.0202, -0.0234)
RISE_M = 0.0015
RISE_TRIES = 3
FIND_PAD_PX = 60     # after the pick the thing is looked for this far round where it lay
DONE_NEAR_M = 0.03   # `all`: a thing found this near one done (or given up) is that one
ALL_TRIES = 2
MAX_POS_ERR_M = 0.002
MAX_JAW_ERR_DEG = 3.0
GONE_SHARE = 0.3     # after the pick, less than this share of its pixels left where it lay: picked up
MODEL = "qwen3.8-max"


class TooBig(SystemExit):
    """The things found are all too wide for the jaws (wider than OPEN_MAX_M less OPEN_FIT_M)."""


class NotInWrist(SystemExit):
    """The wrist camera's look found nothing to aim at."""


class FillsWrist(NotInWrist):
    """The thing fills the wrist picture: none of its edges to aim at."""


class Rig:
    """The devices a pick uses: the left arm, the neck (look(yaw, pitch) -> its pose as read_neck_deg
    gives it), the neck's reading and the head's photos ({"left", "right"}). Run alone, this script
    opens them itself; a caller that holds them already (demo/bottle_to_bin.py: its base owns the
    neck's port, its head the cameras, one arm) hands them in, one owner per port."""

    def __init__(self, arm=None, look=None, neck_deg=None, frames=None):
        self._own = arm is None
        self.arm = arm or LeftArm()
        self.look = look or move_neck
        self.neck_deg = neck_deg or read_neck_deg
        self.frames = frames or (lambda: grab()[1])

    def close(self):
        if self._own:
            self.arm.close()


def offset(from_fixed=FROM_FIXED_M):
    """The grasp point in gripper_link: at the fixed-jaw tip's height, from_fixed towards the moving jaw."""
    return so101_kin.FIXED_JAW_TIP + np.array([from_fixed, 0.0, 0.0])


def jaw_gap(run):
    """How far apart the jaw tips open for the thing."""
    if run["width_m"] > OPEN_GAP_M - OPEN_MARGIN_M and run.get("kind") in ("box", "bag"):
        return OPEN_MAX_M
    return float(np.clip(run["width_m"] + OPEN_MARGIN_M, OPEN_GAP_M, OPEN_MAX_M))


def from_fixed(run):
    """Where the thing's middle goes between the open jaws, from the fixed jaw: FROM_FIXED_M for a
    small thing, half way for a wider one."""
    return FROM_FIXED_M if run["width_m"] <= OPEN_GAP_M - OPEN_MARGIN_M else jaw_gap(run) / 2


def save(d, run):
    (d / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n")


def last_run():
    runs = sorted(p for p in RUNS.glob("*/run.json"))
    if not runs:
        raise SystemExit("还没有 look 过：python pick_small.py look")
    return runs[-1].parent, json.loads(runs[-1].read_text())


def photos(d, tag, rig):
    frames = rig.frames()
    for eye, img in frames.items():
        cv2.imwrite(str(d / f"{tag}_{eye}.jpg"), img)
    return frames


_wrist_cap = None


def wrist_photo(d, tag):
    """One frame of the left wrist camera, or None when it cannot be opened (another program may hold
    it). The camera stays open until wrist_release(); a read that fails opens it afresh once."""
    global _wrist_cap
    img = None
    if _wrist_cap is not None:
        for _ in range(WRIST_STALE_FRAMES):
            _wrist_cap.grab()
        ok, f = _wrist_cap.read()
        img = f if ok else None
    if img is None:
        wrist_release()
        _wrist_cap = cv2.VideoCapture(WRIST_CAM, cv2.CAP_V4L2)
        _wrist_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        _wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(WRIST_SETTLE_FRAMES):  # let the exposure settle
            ok, f = _wrist_cap.read()
            if ok:
                img = f
    if img is not None:
        cv2.imwrite(str(d / f"{tag}_wrist.jpg"), img)
    return img


def wrist_release():
    global _wrist_cap
    if _wrist_cap is not None:
        _wrist_cap.release()
        _wrist_cap = None


def to_px(cam, p_base):
    T = np.linalg.inv(cam["T_base_cam"])
    pc = T[:3, :3] @ p_base + T[:3, 3]
    px, _ = cv2.projectPoints(pc.reshape(1, 3), np.zeros(3), np.zeros(3), cam["K"], cam["dist"])
    return tuple(int(round(v)) for v in px.reshape(2))


def mark(img, cam, q, path, run=None):
    """The picture with where joints q put the fixed-jaw tip (red), the grasp point (yellow) and the
    closing line, by the kinematics and arm_mount.yaml; with `run`, the open moving-jaw tip (blue)."""
    T = mount.load()
    R, t = T[:3, :3], T[:3, 3]
    ff = FROM_FIXED_M if run is None else from_fixed(run)
    p, _, close = ik.pose(q, offset(ff))
    tip, gp, end = R @ so101_kin.tip_position(q) + t, R @ p + t, R @ (p + close * 0.03) + t
    out = img.copy()
    cv2.circle(out, to_px(cam, tip), 5, (0, 0, 255), 2)
    cv2.circle(out, to_px(cam, gp), 5, (0, 220, 255), 2)
    cv2.line(out, to_px(cam, tip), to_px(cam, end), (0, 220, 255), 1)
    if run is not None:
        cv2.circle(out, to_px(cam, R @ (p + close * (jaw_gap(run) - ff)) + t), 5, (255, 120, 0), 2)
    cv2.imwrite(str(path), out)


def is_long(run):
    return run["length_m"] >= LONG_MIN_M and run["length_m"] >= LONG_RATIO * run["width_m"]


def plan(run, arm_now):
    """IK with the jaws closing across the thing for press, align, pre, hover and lift, each step
    from the one before so the wrist keeps its orientation all the way down and up. The tool upright
    if that reaches, else leaning the least of TILT_STEPS_DEG; a short thing's jaws' direction is
    tried round AXIS_TRIES_DEG. run["close_dir"], when set, is the side the fixed jaw must be on.
    Sets run["axis_base"] and run["tilt_deg"] to what was taken."""
    T = mount.load()
    R = T[:3, :3]
    mid = np.array(run["mid_base_m"]) + np.array(run["nudge_mm"]) / 1000.0
    # Seeds: where the arm is, and earlier presses of this script and grasps of pick_bottle (each one
    # tilt x direction tried costs a solve from every seed, so few and different ones).
    seeds, seen = [arm_now], set()
    for r in sorted(RUNS.glob("*/run.json"), reverse=True) + sorted(
            (LLM_VLA / "pick_bottle" / "runs").glob("*/run.json"), reverse=True):
        old = json.loads(r.read_text()).get("plan") or {}
        q = (old.get("press") or old.get("grasp") or {}).get("joints")
        key = None if q is None else tuple(round(q[j] / 10) for j in ik.JOINTS)
        if key is not None and key not in seen:
            seen.add(key)
            seeds.append(q)
        if len(seeds) > SEEDS:
            break
    seeds += [{**s_, "wrist_roll": s_["wrist_roll"] + d} for s_ in seeds[:2] for d in (-90, 90)]
    axis0 = np.array(run.get("axis_measured", run["axis_base"]))
    run["axis_measured"] = axis0.tolist()
    fixed = "tilt_deg" in run  # planned before: the same lean and direction
    turns = (0.0,) if is_long(run) or fixed or from_fixed(run) > FROM_FIXED_M else AXIS_TRIES_DEG
    close_dir = run.get("close_dir")
    steps = (("press", TIP_PRESS_M), ("align", TIP_ALIGN_M), ("pre", TIP_PRE_M), ("hover", HOVER_M),
             ("lift", LIFT_M))
    tilts = (run["tilt_deg"],) if fixed else TILT_STEPS_DEG
    shifts = [np.zeros(3)]
    if not fixed and run["length_m"] >= GRIP_ALONG_MIN_M:
        to_arm = 1.0 if float(axis0[:2] @ (T[:2, 3] - mid[:2])) >= 0 else -1.0
        far = run["length_m"] / 2 - GRIP_END_M
        shifts += [to_arm * s_ * axis0 for s_ in list(np.arange(GRIP_STEP_M, far, GRIP_STEP_M)) + [far]]
    dist = [float(np.linalg.norm((mid + s_ - T[:3, 3])[:2])) for s_ in shifts]
    if min(dist) > REACH_MAX_M:
        raise SystemExit(f"够不着：离左臂底座水平 {min(dist) * 100:.0f} cm"
                         f"{'（夹最靠近的地方）' if len(shifts) > 1 else ''}，超过 {REACH_MAX_M * 100:.0f} cm")
    err = None
    # The least lean that reaches: the middle, any of the directions; else off the middle (upright
    # before leaning), the jaws across it (turned 30 deg they would have closed on a carton's corners).
    tries = [(shift, turn) for shift, far in zip(shifts, dist) if far <= REACH_MAX_M
             for turn in (turns if not shift.any() else (0.0,))]
    for tilt in tilts:
        for shift, turn in tries:
            a = np.radians(turn)
            axis = (np.array(run["axis_base"]) if fixed else
                    np.array([np.cos(a) * axis0[0] - np.sin(a) * axis0[1], np.sin(a) * axis0[0] + np.cos(a) * axis0[1], 0.0]))
            out, prev = {}, None
            try:
                for name, z in steps:
                    q = solve_at(mid + shift, R.T @ axis, z, [prev] if prev else seeds, name, tilt, close_dir,
                                 from_fixed(run))
                    if prev and abs(q["wrist_roll"] - prev["wrist_roll"]) > 30:
                        raise SystemExit(f"{name} 和下面那一步手腕差半圈，不动")
                    out[name] = {"joints": q, "tip_z_m": z}
                    prev = q
            except SystemExit as e:
                err = e
                continue
            run["axis_base"], run["tilt_deg"] = axis.tolist(), tilt
            if tilt:
                print(f"  竖直够不着，工具倾斜 ≤{tilt:.0f}° 去夹", flush=True)
            if shift.any():
                run["nudge_mm"] = (np.array(run["nudge_mm"]) + shift * 1000.0).tolist()
                run["grip_shift_mm"] = (shift * 1000.0).tolist()
                print(f"  中间够不着，夹靠手臂的一头：沿它挪 {np.linalg.norm(shift) * 100:.1f} cm", flush=True)
            if run["width_m"] > OPEN_GAP_M:
                for t in np.radians(LIFT_OUT_TURNS_DEG):
                    turned = np.array([np.cos(t) * axis[0] - np.sin(t) * axis[1], np.sin(t) * axis[0] + np.cos(t) * axis[1], 0.0])
                    try:
                        q = solve_at(mid + shift + [LIFT_OUT_M, 0.0, 0.0], R.T @ turned, LIFT_M, [prev], "lift_out", tilt,
                                     close_dir, from_fixed(run))
                    except SystemExit:
                        continue
                    if abs(q["wrist_roll"] - prev["wrist_roll"]) <= 30:
                        out["lift_out"] = {"joints": q, "tip_z_m": LIFT_M}
                        break
                else:
                    print(f"  抬起时往外 {LIFT_OUT_M * 100:.0f} cm 够不着，直着抬", flush=True)
            return out
    raise err


def solve_at(mid, axis_arm, z, seeds, name, max_tilt=0.0, close_dir=None, from_fixed=FROM_FIXED_M):
    """Joints with the jaws closing level and across axis_arm, the grasp point (offset(from_fixed)) over base-frame `mid`,
    the fixed-jaw tip z above the floor, the tool upright or leaning up to max_tilt, the wrist in
    WRIST_ROLL_RANGE and, with close_dir (base frame), the moving jaw on that side of the fixed one.
    A solution the wrong way round is solved again from the wrist turned half round."""
    T = mount.load()
    R, t = T[:3, :3], T[:3, 3]
    goal = R.T @ (np.array([mid[0], mid[1], z]) - t)
    limits = joint_limits()

    def check(q):
        p, tool, close = ik.pose(q, offset(from_fixed))
        pos = float(np.linalg.norm(p - goal))
        jaw = float(max(np.degrees(np.arcsin(min(1.0, abs(close @ axis_arm)))),
                        np.degrees(np.arcsin(min(1.0, abs(close[2]))))))
        tilt = float(np.degrees(np.arccos(np.clip(-tool[2], -1, 1))))
        reach = pos <= MAX_POS_ERR_M and jaw <= MAX_JAW_ERR_DEG and tilt <= max(2.0, max_tilt + 1.0)
        way = (WRIST_ROLL_RANGE[0] <= q["wrist_roll"] <= WRIST_ROLL_RANGE[1]
               and (close_dir is None or float((R @ close) @ np.asarray(close_dir)) > 0))
        return reach, way, (pos, jaw, tilt)

    q, _ = ik.solve(goal, axis_arm, offset(from_fixed), limits, seeds, max_tilt_deg=max_tilt)
    reach, way, nums = check(q)
    if reach and not way:
        turned = [{**q, "wrist_roll": q["wrist_roll"] + d} for d in (180.0, -180.0)]
        q2, _ = ik.solve(goal, axis_arm, offset(from_fixed), limits, turned, max_tilt_deg=max_tilt)
        reach, way, nums = check(q2)
        q = q2
    if not reach:
        far = np.linalg.norm((np.asarray(mid) - t)[:2]) * 100
        raise SystemExit(f"{name} 够不着：离左臂底座水平 {far:.0f} cm（位置差 {nums[0] * 1000:.1f} mm，"
                         f"夹爪方向差 {nums[1]:.1f}°，工具倾斜 {nums[2]:.1f}°）")
    if not way:
        raise SystemExit(f"{name}：手腕要转到 {q['wrist_roll']:.0f}°（限 {WRIST_ROLL_RANGE[0]:.0f}~"
                         f"{WRIST_ROLL_RANGE[1]:.0f}°）才能让固定爪在要的一侧")
    return q


def open_deg(arm, gap=OPEN_GAP_M):
    """Gripper angle with the jaw tips `gap` apart."""
    tip = so101_kin.FIXED_JAW_TIP
    r = float(np.hypot(tip[0] - JAW_PIVOT_XZ[0], tip[2] - JAW_PIVOT_XZ[1]))
    return arm.limits["gripper"][0] + float(np.degrees(2 * np.arcsin(min(1.0, gap / (2 * r)))))


def close(arm):
    """Close the jaws all the way in CLOSE_STEP_DEG steps, pushing what is between them to the fixed
    jaw; the angle it got to."""
    closed = arm.limits["gripper"][0]
    g = arm.deg()["gripper"]
    for target in np.arange(g - CLOSE_STEP_DEG, closed, -CLOSE_STEP_DEG).tolist() + [closed]:
        g = arm.move({"gripper": target}, CLOSE_DEG_S)["gripper"]
    return g


def wrist_calib():
    """`calib`'s numbers: (fixed-jaw tip, moving-jaw tip, rise shift) in wrist pixels, and the unit
    picture direction from the moving jaw towards the fixed one."""
    if not TARGET_FILE.is_file() or "fixed_tip_px" not in json.loads(TARGET_FILE.read_text()):
        raise SystemExit("还没标定手腕相机：先放一根平的纸条在车前，python pick_small.py calib")
    c = json.loads(TARGET_FILE.read_text())
    fixed, moving, shift = (np.array(c[k], float) for k in ("fixed_tip_px", "moving_tip_px", "rise_shift_px"))
    u = (fixed - moving) / np.linalg.norm(fixed - moving)
    return fixed, moving, shift, u


def tool_yaw(q):
    """Direction (rad, base frame) the jaws close along (fixed towards moving) for joints q."""
    c = mount.load()[:3, :3] @ ik.pose(q, offset())[2]
    return float(np.arctan2(c[1], c[0]))


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def wrist_J(q):
    """px per mm of the arm along base x and y (columns) in the wrist camera from TIP_ALIGN_M up, for
    joints q: `calib` measured it once in the gripper's own frame (the camera turns with the jaws), so
    it only turns with the jaws' direction. Measured afresh by two moves at every grasp it came out
    different each time, up to 3x off (the thing pushed, half hidden, a stale photo; 2026-11-04)."""
    c = json.loads(TARGET_FILE.read_text())
    if "px_per_mm_tool" not in c:
        raise SystemExit("标定里没有手腕相机的 px/mm：重新 python pick_small.py calib")
    return np.array(c["px_per_mm_tool"]) @ rot(-tool_yaw(q))


def jaws_in_wrist(img, zone=True):
    """The jaws' pixels in a wrist photo: big, light, colourless, and inside JAW_ZONE_FILE once `jawzone`
    has made it (zone False: anywhere, as `jawzone` itself counts them)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[..., 1:] - 128.0, axis=-1)
    m = ((lab[..., 0] > JAW_L_MIN) & (chroma < JAW_CHROMA_MAX)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    if zone and JAW_ZONE_FILE.is_file():
        m &= (cv2.imread(str(JAW_ZONE_FILE), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= JAW_MIN_PX]
    return np.isin(lab_img, keep) if keep else np.zeros(img.shape[:2], bool)


def straight_edges(mask, jaws):
    """[(direction (unit, pixels), length, middle)] of the straight stretches of a thing's outline in a
    wrist photo, longest first: a polygon fit within EDGE_FIT_PX, at least EDGE_MIN_PX long; stretches
    along the jaws or the picture's border (where the jaws hide it or it leaves the picture) left out."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    poly = cv2.approxPolyDP(max(contours, key=cv2.contourArea), EDGE_FIT_PX, True).reshape(-1, 2).astype(float)
    near_jaws = cv2.dilate(jaws.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    h, w = mask.shape
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        length = float(np.linalg.norm(b - a))
        mid = (a + b) / 2
        if (length < EDGE_MIN_PX or mid[0] < EDGE_BORDER_PX or mid[1] < EDGE_BORDER_PX
                or mid[0] > w - EDGE_BORDER_PX or mid[1] > h - EDGE_BORDER_PX or near_jaws[int(mid[1]), int(mid[0])]):
            continue
        out.append(((b - a) / length, length, mid))
    return sorted(out, key=lambda e: -e[1])


def find_by_vlm(img, near, model):
    """(middle, box) in pixels of the VLM's box nearest `near` in a wrist photo, within VLM_NEAR_PX, or None."""
    near = np.asarray(near, float)
    best = None
    for _, b in detect_small.ask(img, model, 20.0, detect_small.WRIST_PROMPT):
        m = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        dist = float(np.linalg.norm(m - near))
        if dist <= VLM_NEAR_PX and (best is None or dist < best[0]):
            best = (dist, m, b)
    return None if best is None else best[1:]


def detect_in_wrist(img, near, model, jaws):
    """(middle, box, pixels) of the thing in a wrist photo: the VLM's box nearest `near`, cut out of the
    carpet by GrabCut inside it (the jaws left out); pixels None when GrabCut finds nothing. None when
    the VLM boxes nothing near."""
    vlm = find_by_vlm(img, near, model)
    if vlm is None:
        return None
    mid, box = vlm
    mask = detect_small.segment_grabcut(img, box, None)
    if mask is not None:
        mask &= ~jaws
        if not mask.any():
            mask = None
    return mid, box, mask


def track_in_wrist(img, last, shift, jaws):
    """(middle, box, pixels) of the thing in a wrist photo cut out by GrabCut round its last pixels
    `last` moved by `shift` (pixels); None when that cut does not look like the same thing."""
    ys, xs = np.nonzero(last)
    h, w = last.shape
    x1, y1 = xs.min() + shift[0] - TRACK_PAD_PX, ys.min() + shift[1] - TRACK_PAD_PX
    x2, y2 = xs.max() + shift[0] + TRACK_PAD_PX, ys.max() + shift[1] + TRACK_PAD_PX
    box = [max(0.0, x1), max(0.0, y1), min(w - 1.0, x2), min(h - 1.0, y2)]
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    mask = detect_small.segment_grabcut(img, box, None)
    if mask is None:
        return None
    mask &= ~jaws
    want = np.array([xs.mean(), ys.mean()]) + shift
    mid = np.argwhere(mask)[:, ::-1].mean(axis=0) if mask.any() else None
    if (mid is None or not TRACK_AREA[0] <= mask.sum() / last.sum() <= TRACK_AREA[1]
            or np.linalg.norm(mid - want) > TRACK_NEAR_PX):
        return None
    return mid, box, mask


def wrist_px(d, tag, near, run, model, u, goal=None, last=None, shift=None):
    """In a new wrist photo `tag`: (the middle of the thing's side facing the fixed jaw (u: picture
    direction towards it), its middle, its pixels (or its box as pixels), the jaws' pixels, the photo,
    whether that side is cut off by the picture's border).
    The thing is found by detection (detect_in_wrist), never by its colour; with its pixels in the last
    photo (`last`) and how far the arm moved it in the picture (`shift`), it is followed there first
    (track_in_wrist). None when it is not there."""
    img = wrist_photo(d, tag)
    if img is None:
        raise SystemExit(f"{WRIST_CAM} 没有画面（别的程序占着它的话先停掉）")
    jaws = jaws_in_wrist(img)
    got = None if last is None else track_in_wrist(img, last, np.asarray(shift, float), jaws)
    if got is None:
        if last is not None:
            print("  跟不上它挪过去的样子，再让 VLM 框一次", flush=True)
        got = detect_in_wrist(img, near, model, jaws)
    if got is None:
        return None
    mid, box, mask = got
    if mask is not None:
        pts = np.argwhere(mask)[:, ::-1].astype(float)
        t = pts @ u
        side = pts[t >= np.quantile(t, 1.0 - EDGE_SIDE_SHARE)].mean(axis=0)
    else:  # the VLM's box only: its side facing the fixed jaw
        half = abs(u[0]) * (box[2] - box[0]) / 2 + abs(u[1]) * (box[3] - box[1]) / 2
        side = mid + u * half
    vis = img.copy()
    cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 200, 255), 1)
    if mask is not None:
        vis[mask] = (0.5 * vis[mask] + [0, 0, 120]).astype(np.uint8)
    cv2.drawMarker(vis, tuple(int(v) for v in side), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    if goal is not None:
        cv2.drawMarker(vis, tuple(int(v) for v in goal), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
    cv2.imwrite(str(d / f"{tag}_wrist_found.jpg"), vis)
    h, w = img.shape[:2]
    borders = sum((box[0] < EDGE_BORDER_PX, box[1] < EDGE_BORDER_PX,
                   box[2] > w - EDGE_BORDER_PX, box[3] > h - EDGE_BORDER_PX))
    if (box[2] - box[0]) * (box[3] - box[1]) >= WRIST_FILL_SHARE * w * h and borders >= 2:
        raise FillsWrist(f"它占满了手腕画面，看不到它的边：看 {d}/{tag}_wrist_found.jpg")
    return side, mid, mask, jaws, img, cut_span(mask, box, u, w, h) >= WRIST_CUT_SPAN_PX


def cut_span(mask, box, u, w, h):
    """How far (pixels) the thing runs along the picture's border on the side facing the fixed jaw
    (u: picture direction towards it): by its outline, else its box."""
    b = EDGE_BORDER_PX
    if mask is not None:
        span = {"l": mask[:, :b].any(axis=1).sum(), "r": mask[:, -b:].any(axis=1).sum(),
                "t": mask[:b].any(axis=0).sum(), "b": mask[-b:].any(axis=0).sum()}
    else:
        span = {"l": (box[3] - box[1]) * (box[0] < b), "r": (box[3] - box[1]) * (box[2] > w - b),
                "t": (box[2] - box[0]) * (box[1] < b), "b": (box[2] - box[0]) * (box[3] > h - b)}
    sides = ("r" if u[0] > 0.5 else "l" if u[0] < -0.5 else None, "b" if u[1] > 0.5 else "t" if u[1] < -0.5 else None)
    return float(max([span[k] for k in sides if k] or [0]))


def align(arm, d, run, model, z=TIP_ALIGN_M):
    """The user's grasp: the jaws turned onto the thing's straight edge (choose_edge), then the middle
    of its side facing the fixed jaw brought EDGE_GAP_MM off the fixed jaw's face in the wrist camera,
    the tip z up; run["nudge_mm"] and run["plan"] follow. Returns the pixel error left."""
    fixed_px, moving_px, shift, u = wrist_calib()
    face = (fixed_px + moving_px) / 2 + shift  # the fixed jaw's face (where the closed tips meet), from z up
    T = mount.load()
    mid0 = np.array(run["mid_base_m"])
    tilt = run.get("tilt_deg", 0.0)

    def go(nudge, label):
        # From where the arm is, and from the plan (after choose_edge, the wrist turned round).
        axis_arm = T[:3, :3].T @ np.array(run["axis_base"])
        q = solve_at(mid0 + np.r_[nudge, 0.0] / 1000.0, axis_arm, z, [arm.deg(), run["plan"]["align"]["joints"]],
                     label, tilt, run.get("close_dir"), from_fixed(run))
        move(arm, q, pick.NEAR_DEG_S, label)

    def find(tag, near, goal=None, last=None, shift=None):
        got = wrist_px(d, tag, near, run, model, u, goal, last, shift)
        if got is None:
            raise NotInWrist(f"手腕相机里没看到它：看 {d}/{tag}_wrist.jpg")
        return got

    nudge = np.array(run["nudge_mm"][:2], float)
    nudge0 = nudge.copy()
    c0, m0, mask0, jaws0, _, cut = find("align0", face)
    J = wrist_J(arm.deg())
    # A cut-off outline's edges are partly the picture's border: no turning on them.
    if not cut and choose_edge(run, J, mask0, jaws0, m0):  # turned onto a straight edge: look again
        go(nudge, "转手腕：固定爪顺着平直的边")
        c0, m0, mask0, _, _, cut = find("turned0", face)
        J = wrist_J(arm.deg())
    # The goal: EDGE_GAP_MM from the face towards the moving jaw, in pixels by the px/mm across the jaws.
    across = np.cross([0.0, 0.0, 1.0], run["axis_base"])[:2]
    goal = face - u * EDGE_GAP_MM * float(np.linalg.norm(J @ across))
    # Gripped off its middle (GRIP_ALONG_MIN_M): the thing sits that much further along the jaws.
    along = np.array(run["axis_base"])[:2]
    goal = goal + J @ (along * float(along @ np.array(run.get("grip_shift_mm", [0.0, 0.0, 0.0]))[:2]))
    c, m, mask = c0, m0, mask0
    left = float(np.linalg.norm(goal - c))
    for i in range(ALIGN_TRIES + CUT_TRIES):
        delta = np.linalg.solve(J, goal - c)
        if cut:  # towards the side out of the picture, a step at a time
            if np.linalg.norm(nudge - nudge0) >= CUT_MAX_MM - 1e-6:
                break
            delta = delta * min(1.0, CUT_STEP_MM / float(np.linalg.norm(delta)))
            nudge = nudge + delta
            go(nudge, f"它朝固定爪那边伸出了画面，往那边挪 ({delta[0]:+.1f}, {delta[1]:+.1f}) mm 再看")
            c, m, mask, _, _, cut = find(f"align{i + 1}", m + J @ delta, goal, mask, J @ delta)
            left = float(np.linalg.norm(goal - c))
            continue
        if np.linalg.norm(delta) > ALIGN_MAX_MM:
            raise SystemExit(f"手腕相机要挪 {np.linalg.norm(delta):.0f} mm 才对得上，不信：看 {d}/align*_wrist_found.jpg")
        if np.linalg.norm(delta) > DAMP_FROM_MM:
            delta = delta * ALIGN_DAMP
        nudge = nudge + delta
        go(nudge, f"对准 {i + 1}：再挪 ({delta[0]:+.1f}, {delta[1]:+.1f}) mm")
        c_was = c
        c, m, mask, _, _, cut = find(f"align{i + 1}", m + J @ delta, goal, mask, J @ delta)
        # A thing on the floor moves in the wrist picture when the arm moves; one stuck to the jaws
        # does not (a pumpkin seed caught in the fixed jaw was chased for 4 steps, 2026-11-04).
        if (not cut and np.linalg.norm(J @ delta) >= STUCK_EXPECT_PX
                and np.linalg.norm(c - c_was) < STUCK_SHARE * np.linalg.norm(J @ delta)):
            raise SystemExit(f"臂挪了 {np.linalg.norm(delta):.0f} mm，它在手腕画面里几乎没动：像是卡在夹爪上了，"
                             f"不夹（看 {d}/align{i + 1}_wrist_found.jpg）")
        left = float(np.linalg.norm(goal - c))
        print(f"  还差 {left:.0f} px", flush=True)
        if left <= WRIST_TOL_PX:
            break
    if cut:
        raise FillsWrist(f"挪了 {np.linalg.norm(nudge - nudge0):.0f} mm 它朝固定爪那边还在画面外：看 {d}/align*_wrist_found.jpg")
    if left > WRIST_GIVE_UP_PX:  # a leaning tool ran off by 198 px and closed anyway (2026-11-04)
        raise SystemExit(f"对了 {ALIGN_TRIES} 次还差 {left:.0f} px，不夹：看 {d}/align*_wrist_found.jpg")
    run["nudge_mm"] = [float(nudge[0]), float(nudge[1]), 0.0]
    # The aim may have moved the grasp out of upright reach (a washer 25 cm off the arm base, aimed
    # within 4 px, 2026-11-04): then the same aim and jaws, leaning a little more, up to AIM_TILT_MAX_DEG
    # (the wrist camera's px/mm was taken upright; a few degrees hardly change it).
    for tilt in [run["tilt_deg"]] + [t for t in TILT_STEPS_DEG if run["tilt_deg"] < t <= AIM_TILT_MAX_DEG]:
        run["tilt_deg"] = tilt
        try:
            run["plan"] = plan(run, arm.deg())
            break
        except SystemExit as err:
            last = err
    else:
        raise last
    return left


def choose_edge(run, J, mask, jaws, centre):
    """Turn the jaws onto a straight edge of the thing, the fixed jaw on that edge's side: the longest
    edge the arm reaches that way (WRIST_ROLL_RANGE included) sets run["axis_base"] (along it) and
    run["close_dir"] (from it into the thing), and True when the wrist must turn for it. A picture
    direction w is the floor direction -J^-1 w."""
    # Only a real outline has edges: a box standing in for one (GrabCut found nothing, a small screw)
    # gave the box's own sides, and the wrist turned for them (2026-11-04).
    edges = [] if mask is None else straight_edges(mask, jaws)
    if not edges:
        print("  手腕画面里没分出轮廓或没有平直的边，夹爪方向不变", flush=True)
        return False
    T = mount.load()
    R = T[:3, :3]
    q_now = run["plan"]["align"]["joints"]
    close_now = R @ ik.pose(q_now, offset())[2]
    old = {k: run.get(k) for k in ("axis_base", "close_dir", "plan")}
    found = []
    for e_px, length, e_mid in edges:
        along = np.linalg.solve(J, e_px)
        along = np.r_[along / np.linalg.norm(along), 0.0]
        out = -np.linalg.solve(J, e_mid - centre)  # from the thing's middle out to this edge, base frame
        out = np.r_[out, 0.0] - along * float(np.r_[out, 0.0] @ along)
        if np.linalg.norm(out) < 1e-6:
            continue
        into = -out / np.linalg.norm(out)
        off = float(np.degrees(np.arccos(min(1.0, abs(float(along @ np.array(run["axis_base"])))))))
        found.append((length, along, into, off, float(close_now @ into) > 0))
    # Jaws already along a long enough edge stay as they are, whichever side the fixed jaw is on: the
    # side of a strip facing the fixed jaw is half hidden by it, so only its far side showed as an
    # edge, and turning the wrist half round for that lost the strip twice (2026-11-04); its two
    # sides are parallel anyway. Only jaws across the edges turn, and then the fixed jaw goes on the
    # edge's side when the arm reaches that way.
    longest = max((f[0] for f in found), default=0.0)
    for length, along, into, off, same_side in found:
        if off <= EDGE_TURN_DEG and length >= EDGE_KEEP_SHARE * longest:
            print(f"  平直的边 {length:.0f} px：夹爪面已经顺着它（差 {off:.0f}°），不转", flush=True)
            if same_side:
                run["close_dir"] = into.tolist()
            return False
    for length, along, into, off, _ in found:
        run["axis_base"], run["close_dir"] = along.tolist(), into.tolist()
        try:
            run["plan"] = plan(run, q_now)
        except SystemExit as err:
            print(f"  平直的边 {length:.0f} px：固定爪靠过去够不着（{err}）", flush=True)
            for k, v in old.items():
                if v is None:
                    run.pop(k, None)
                else:
                    run[k] = v
            continue
        print(f"  平直的边 {length:.0f} px：转手腕，让固定爪顺着它、在它外侧", flush=True)
        return True
    print("  哪条平直的边都靠不过去，夹爪方向不变", flush=True)
    return False


def shift_of(img0, mask0, img1):
    """How far (pixels) the thing moved from img0 to img1: its pixels in img0 (mask0, a little grown)
    matched against img1 within SHIFT_SEARCH_PX. The whole thing, not one point of it: the middle of
    its side slides along a strip as the jaws hide more or less of it, and its free end jumps when it
    leaves the picture (px/mm off by 3x, 2026-11-04). None when the best match is poor."""
    m = cv2.dilate(mask0.astype(np.uint8), np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    tpl = img0[y1:y2, x1:x2]
    h, w = img1.shape[:2]
    sx1, sy1 = max(0, x1 - SHIFT_SEARCH_PX), max(0, y1 - SHIFT_SEARCH_PX)
    sx2, sy2 = min(w, x2 + SHIFT_SEARCH_PX), min(h, y2 + SHIFT_SEARCH_PX)
    win = img1[sy1:sy2, sx1:sx2]
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return None
    weight = cv2.merge([m[y1:y2, x1:x2] * 255] * 3)
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCORR_NORMED, mask=weight)
    _, best, _, loc = cv2.minMaxLoc(np.nan_to_num(res, nan=0.0))
    if best < SHIFT_MIN_SCORE:
        return None
    return np.array([sx1 + loc[0] - x1, sy1 + loc[1] - y1], float)


def probe(go, photo, img0, mask0):
    """`calib`: px per mm of the arm along base x and y (columns), two PROBE_MM moves from where the
    arm is, the thing's shift in the picture by shift_of."""
    J = np.zeros((2, 2))
    for k, name in enumerate(("x", "y")):
        step = np.zeros(2)
        step[k] = PROBE_MM
        go(step, f"试探：沿底盘 {name} 挪 {PROBE_MM:.0f} mm")
        moved = shift_of(img0, mask0, photo(f"calib_probe_{name}"))
        if moved is None:
            raise SystemExit(f"试探挪动后在手腕画面里对不上它（看 calib_probe_{name}_wrist.jpg）")
        J[:, k] = moved / PROBE_MM
    print(f"  手腕相机：每 mm 像素 x→{np.round(J[:, 0], 1).tolist()} y→{np.round(J[:, 1], 1).tolist()}", flush=True)
    if abs(np.linalg.det(J)) < 1.0:
        raise SystemExit("试探挪动后手腕画面里几乎没动，量不出换算")
    return J


def move(arm, q, speed, label, floor=pick.MIN_TIP_Z_M):
    """pick.move, but the fixed-jaw tip may go down to `floor` (below the floor for the press)."""
    now = arm.deg()
    full = {**now, **q}
    z = [pick.tip_base({k: now[k] + (full[k] - now[k]) * s for k in now})[2] for s in np.linspace(0, 1, 21)]
    if min(z) < min(z[0], z[-1]) - pick.PATH_CLEAR_M or min(z) < min(floor, z[0] - 0.001):
        raise SystemExit(f"{label}：路上爪尖会低到 {min(z) * 1000:.0f} mm，不动")
    t0 = time.monotonic()
    got = arm.move(q, speed)
    print(f"{label} {time.monotonic() - t0:.1f}s：{pick.fmt(got)}", flush=True)
    return got


def look(model, rig, skip=(), longest=False, expect=None, plan_it=True):
    """Neck to the grasp view, arm parked, photos; the things found, those not near `skip` (done),
    nearest the arm first (longest: ones with an outline only, longest first, for `calib`; expect,
    base-frame x, y: only those within EXPECT_NEAR_M of it, nearest first), the first the arm reaches
    is planned (plan_it False: the first is only measured, for a caller that moves the body to it
    first). Returns (dir, run)."""
    for _ in range(3):  # the neck has backlash: try again when it stops short
        neck = rig.look(*GRASP_NECK)
        if max(abs(neck["neck_yaw"] - GRASP_NECK[0]), abs(neck["neck_pitch"] - GRASP_NECK[1])) <= NECK_TOL_DEG:
            break
    else:
        raise SystemExit(f"脖子没到抓取视角：{neck}")
    pick.park(rig.arm)  # out of the view, where ROBOT_MASK expects it
    arm_now = rig.arm.deg()
    d = RUNS / time.strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True)
    frames = photos(d, "look", rig)
    run = {"neck_deg": rig.neck_deg(), "nudge_mm": [0.0, 0.0, 0.0]}
    cam = pick.head_cam(run["neck_deg"])
    found, missed, vis = detect_small.locate_all(frames["left"], cam, model, 20.0, pick.ROBOT_MASK["left"])
    cv2.imwrite(str(d / "look_left_detect.jpg"), vis)
    for label, box in missed:
        print(f"  {label} 框在 {box}，量不出位置，不夹", flush=True)
    t = mount.load()[:3, 3]
    found = [g for g in found if all(np.hypot(*(g["mid"][:2] - np.asarray(s_[:2]))) > DONE_NEAR_M for s_ in skip)]
    found.sort(key=lambda g: float(np.hypot(*(g["mid"][:2] - t[:2]))))  # nearest the arm first
    if longest:  # the longest one with an outline the arm reaches
        found = sorted((g for g in found if g["kind"] != "box"), key=lambda g: -g["length"])
    if expect is not None:  # the one a caller came for
        e = np.asarray(expect[:2], float)
        found = sorted((g for g in found if np.hypot(*(g["mid"][:2] - e)) <= EXPECT_NEAR_M),
                       key=lambda g: float(np.hypot(*(g["mid"][:2] - e))))
    run["skipped"] = []
    wide = []
    for got in found:
        if got["width"] > OPEN_MAX_M - OPEN_FIT_M and got["kind"] != "bag":  # a bag is soft: the jaws take a fold
            print(f"  {got['label']} 宽 {got['width'] * 100:.1f} cm，夹爪张不开那么大，不夹", flush=True)
            wide.append(got["mid"].tolist())
            run["skipped"].append(got["mid"].tolist())
            continue
        for k in ("axis_measured", "tilt_deg", "close_dir"):
            run.pop(k, None)
        run.update({"label": got["label"], "box_px": got["box"], "pixels": got["px"], "mid_px": got["mid_px"],
                    "mid_base_m": got["mid"].tolist(), "axis_base": got["axis"].tolist(),
                    "length_m": got["length"], "width_m": got["width"], "kind": got["kind"]})
        how = {"grabcut": "", "box": "（框里分不出轮廓，只按检测框）", "bag": "（袋子：按整个袋子的框，透明部分也算）"}[got["kind"]]
        print(f"{got['label']}{how}：中心 (mm) {np.round(got['mid'] * 1000).astype(int).tolist()}，"
              f"长 {got['length'] * 100:.1f} cm，宽 {got['width'] * 100:.1f} cm，"
              f"长边朝 {np.degrees(np.arctan2(got['axis'][1], got['axis'][0])):.0f}°（{d.name}）", flush=True)
        if not plan_it:
            save(d, run)
            return d, run
        try:
            run["plan"] = plan(run, arm_now)
            break
        except SystemExit as err:
            print(f"  {err}", flush=True)
            run["skipped"].append(got["mid"].tolist())
    else:
        save(d, run)
        if wide and len(wide) == len(found):
            raise TooBig(f"抓取视角里的东西都太大，夹爪张不开：看 {d}/look_left_detect.jpg")
        what = "够得着的小东西" if plan_it else "要找的小东西"
        raise SystemExit(f"抓取视角里没有{what}：看 {d}/look_left_detect.jpg")
    mark(frames["left"], cam, run["plan"]["press"]["joints"], d / "look_left_marked.jpg", run)
    save(d, run)
    return d, run


def check_neck(run, rig):
    now = rig.neck_deg()
    if max(abs(now[k] - run["neck_deg"][k]) for k in now) > 0.3:
        raise SystemExit("脖子比 look 时动了：重新 look")


def still_there(d, run, tag, model=MODEL):
    """1 when the VLM, in the photos `tag` (arm parked), boxes something within FIND_PAD_PX / 2 of
    where the thing lay (the jaws may have pushed it a little), else 0."""
    x1, y1, x2, y2 = run["box_px"]
    img = cv2.imread(str(d / f"{tag}_left.jpg"))
    mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    near = [b for _, b in detect_small.ask(img, model, 20.0)
            if np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - mid) <= FIND_PAD_PX / 2]
    return 1.0 if near else 0.0


def cmd_look(args, rig):
    d, run = look(args.model, rig)
    for name, s in run["plan"].items():
        print(f"{name:5s}: {pick.fmt(s['joints'])}")
    print(f"计划画在 {d}/look_left_marked.jpg；python pick_small.py check 先试着对准，grab 夹")


def cmd_check(args, rig):
    d, run = last_run()
    check_neck(run, rig)
    p = run["plan"]
    cam = pick.head_cam(run["neck_deg"])
    arm = rig.arm
    move(arm, {**p["hover"]["joints"], "gripper": open_deg(arm)}, pick.FREE_DEG_S,
         f"到上方，夹爪张开 {OPEN_GAP_M * 100:.0f} cm")
    move(arm, p["pre"]["joints"], pick.NEAR_DEG_S, f"下到爪尖离地 {TIP_PRE_M * 100:.0f} cm")
    time.sleep(0.5)
    frames = photos(d, "check", rig)
    wrist_photo(d, "check")
    q = arm.deg()
    move(arm, p["hover"]["joints"], pick.NEAR_DEG_S, "回到上方")
    pick.park(arm)
    mark(frames["left"], cam, q, d / "check_left_marked.jpg")
    print(f"{d}/check_left_marked.jpg：红圈是按运动学的固定爪尖，黄圈是夹取点；和真夹爪比，差多少用 grab --dx/--dy 修")


def cmd_grab(args, rig, d=None, run=None):
    """Aim (wrist camera), close, lift, park holding it; True when it is gone from where it lay. A
    SystemExit leaves the arm where it stopped: the caller parks it."""
    arm = rig.arm
    if d is None:
        d, run = last_run()
        check_neck(run, rig)
        if args.dx or args.dy:
            run["nudge_mm"] = (np.array(run["nudge_mm"]) + [args.dx, args.dy, 0.0]).tolist()
            run["plan"] = plan(run, arm.deg())
            print(f"瞄准再偏 (mm) {run['nudge_mm'][:2]}")
    try:
        return aim_and_grip(args, rig, arm, d, run)
    finally:
        wrist_release()


def aim_and_grip(args, rig, arm, d, run):
    p = run["plan"]
    T = mount.load()
    # A wider thing (a paper ball, a carton) gets the jaws open wider, to OPEN_MAX_M at most.
    gap = jaw_gap(run)
    move(arm, {**p["hover"]["joints"], "gripper": open_deg(arm, gap)}, pick.FREE_DEG_S,
         f"到上方，夹爪张开 {gap * 100:.0f} cm")
    move(arm, p["align"]["joints"], pick.NEAR_DEG_S, f"下到爪尖离地 {TIP_ALIGN_M * 1000:.0f} mm")
    if run.get("kind") == "bag":
        # Close up the wrist camera's cut takes the print for the thing too: the head camera's whole bag.
        print("  袋子大半透明，手腕相机分不清袋子的边：按头部相机量的整个袋子夹", flush=True)
        left = None
    else:
        try:
            left = align(arm, d, run, args.model)
        except NotInWrist as err:
            if run["width_m"] < WIDE_M and not isinstance(err, FillsWrist):
                raise
            print(f"  {err}；它宽 {run['width_m'] * 100:.0f} cm，按头部相机瞄的位置夹", flush=True)
            # Aiming steps may have moved the jaws off it: back over the head camera's aim before going down.
            move(arm, run["plan"]["align"]["joints"], pick.NEAR_DEG_S, "回到头部相机瞄的位置上方")
            left = None
    save(d, run)
    p = run["plan"]
    got = move(arm, p["press"]["joints"], PRESS_DEG_S, f"慢慢下到爪尖离地 {TIP_PRESS_M * 1000:.0f} mm",
               floor=TIP_FLOOR_M)
    tip_z = float((T[:3, :3] @ so101_kin.tip_position(got) + T[:3, 3])[2])
    print(f"  爪尖在离地 {tip_z * 1000:.0f} mm（按运动学），"
          + (f"手腕相机对准还差 {left:.0f} px" if left is not None else "没用手腕相机对准"), flush=True)
    photos(d, "press", rig)
    wrist_photo(d, "press")
    closed = arm.limits["gripper"][0]
    g = close(arm)
    print(f"  夹爪合到 {g:.1f}°（全合 {closed:.1f}°）", flush=True)
    axis_arm = T[:3, :3].T @ np.array(run["axis_base"])
    mid = np.array(run["mid_base_m"]) + np.array(run["nudge_mm"]) / 1000.0
    # A wide thing stops the jaws itself: a milk carton held at -50 deg (closed -62) was taken for the
    # moving jaw caught in the carpet and the rise refused, the arm left on the floor (2026-11-07).
    for k in range(0 if from_fixed(run) > FROM_FIXED_M else RISE_TRIES):
        if g <= closed + JAW_STUCK_DEG:
            break
        # The moving jaw is in the carpet half way: up a little, close on.
        z = TIP_PRESS_M + RISE_M * (k + 1)
        move(arm, solve_at(mid, axis_arm, z, [arm.deg()], "rise", run.get("tilt_deg", 0.0), run.get("close_dir"),
                           from_fixed(run)),
             PRESS_DEG_S, f"抬到爪尖离地 {z * 1000:.0f} mm")
        g = close(arm)
        print(f"  夹爪合到 {g:.1f}°", flush=True)
    wrist_photo(d, "closed")
    if "lift_out" in p:
        move(arm, p["lift_out"]["joints"], pick.NEAR_DEG_S,
             f"抬起 {LIFT_M * 100:.0f} cm，同时往外 {LIFT_OUT_M * 100:.0f} cm（别碰保险杠）")
    else:
        move(arm, p["lift"]["joints"], pick.NEAR_DEG_S, f"抬起 {LIFT_M * 100:.0f} cm")
    wrist_photo(d, "lift")
    # A light thing in the jaws does not swing: through the air as fast as to the hover (PARK_DEG_S
    # took ~9 s, 2026-11-06).
    pick.park(arm, keep_grip=True, speed=pick.FREE_DEG_S)
    arm.ease_grip(closed)
    photos(d, "after", rig)
    left = still_there(d, run, "after", args.model)
    run["result"] = {"tip_z_m": tip_z, "grip_deg": g, "left_share": left, "picked": left < GONE_SHARE}
    save(d, run)
    if left < GONE_SHARE:
        print("夹起来了：原处检测不到它了。夹着它停在收臂位（python pick_small.py drop 松开）")
    else:
        print(f"没夹起来：原处还检测得到（看 {d}/press_left.jpg、after_left.jpg）")
    return left < GONE_SHARE


def cmd_all(args, rig):
    """Every small thing in reach, nearest the arm first: pick it up, hold it up (the pick parks with
    it), put it back where it lay and leave it be; one not picked in ALL_TRIES tries is given up."""
    done, tries, results = [], {}, []
    while True:
        try:
            d, run = look(args.model, rig, done)
        except SystemExit as err:
            print(err)
            break
        done += run.get("skipped", [])
        mid = run["mid_base_m"]
        key = next((k for k in tries if np.hypot(k[0] - mid[0], k[1] - mid[1]) <= DONE_NEAR_M), tuple(mid))
        tries[key] = tries.get(key, 0) + 1
        try:
            picked = cmd_grab(args, rig, d, run)
        except SystemExit as err:
            print(f"  {err}", flush=True)
            pick.park(rig.arm)
            picked = False
        if picked:
            cmd_place(argparse.Namespace(dx=0.0, dy=0.0, turn=0.0), rig)
        if picked or tries[key] >= ALL_TRIES:
            done.append(mid)
            results.append((run["label"], [round(v * 1000) for v in mid[:2]], picked, tries[key]))
    print("结果：")
    for label, mid, picked, n in results:
        print(f"  {label} {mid} mm：{'夹起来了' if picked else '没夹起来'}（试了 {n} 次）")


def jaw_tips(img):
    """(fixed-jaw tip, moving-jaw tip) in a wrist photo of the closed jaws: the mean of each jaw's
    TIP_PX leftmost pixels; the fixed jaw is the lower one in the picture."""
    jaws = jaws_in_wrist(img)
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(jaws.astype(np.uint8), connectivity=8)
    tips = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= JAW_MIN_PX:
            ys, xs = np.nonzero(lab_img == k)
            i = np.argsort(xs)[:TIP_PX]
            tips.append(np.array([xs[i].mean(), ys[i].mean()]))
    if len(tips) != 2:
        raise SystemExit(f"合上的夹爪在手腕画面里认出 {len(tips)} 块，要 2 块")
    return max(tips, key=lambda p: p[1]), min(tips, key=lambda p: p[1])


def cmd_calib(args, rig):
    """Where a thing lying right for the grasp shows in the wrist camera from TIP_ALIGN_M up. The camera
    sits on the gripper, so the closed tips always show at one pixel, and with the tips on the floor
    the floor between them shows there. From TIP_ALIGN_M up that floor shows shifted: the shift is
    measured on a flat strip (detected, cut out by GrabCut) seen from TIP_ALIGN_M and from TIP_LOW_M at
    one place, the whole strip matched between the two (shift_of). With it, the px/mm (two probing
    moves) in the gripper's frame. The closed tips (the fixed jaw's face is where they meet), the shift
    and the px/mm go into TARGET_FILE."""
    d, run = look(args.model, rig, longest=True)
    if not is_long(run):
        raise SystemExit(f"标定要一根平的长条（≥{LONG_MIN_M * 100:.1f} cm），这个是 {run['length_m'] * 100:.1f} cm")
    p = run["plan"]
    T = mount.load()
    axis_arm = T[:3, :3].T @ np.array(run["axis_base"])
    mid = np.array(run["mid_base_m"])
    arm = rig.arm
    try:
        move(arm, {**p["hover"]["joints"], "gripper": open_deg(arm)}, pick.FREE_DEG_S, "到上方")
        move(arm, p["align"]["joints"], pick.NEAR_DEG_S, f"下到爪尖离地 {TIP_ALIGN_M * 1000:.0f} mm")
        img = wrist_photo(d, "calib_high")
        high = None if img is None else detect_in_wrist(img, (320.0, 240.0), args.model, jaws_in_wrist(img))
        if high is None or high[2] is None:
            raise SystemExit(f"手腕相机里没检测出纸条的轮廓：看 {d}/calib_high_wrist.jpg")
        img_high = img
        q_high = arm.deg()

        def go(step, label):
            move(arm, solve_at(mid + np.r_[step, 0.0] / 1000.0, axis_arm, TIP_ALIGN_M, [arm.deg()], label),
                 pick.NEAR_DEG_S, label)

        J = probe(go, lambda tag: wrist_photo(d, tag), img_high, high[2])
        J_tool = J @ rot(tool_yaw(q_high))
        go(np.zeros(2), "回到纸条上方")
        low_q = solve_at(mid, axis_arm, TIP_LOW_M, [arm.deg()], "low")
        move(arm, low_q, PRESS_DEG_S, f"竖直下到爪尖离地 {TIP_LOW_M * 1000:.0f} mm", floor=TIP_FLOOR_M)
        img = wrist_photo(d, "calib_low")
        down = None if img is None else shift_of(img_high, high[2], img)
        if down is None:
            raise SystemExit(f"下到 {TIP_LOW_M * 1000:.0f} mm 后在手腕画面里对不上纸条：看 {d}/calib_low_wrist.jpg")
        move(arm, p["align"]["joints"], PRESS_DEG_S, f"竖直回到爪尖离地 {TIP_ALIGN_M * 1000:.0f} mm")
        move(arm, {"gripper": arm.limits["gripper"][0]}, pick.FREE_DEG_S, "在空中合上夹爪")
        img = wrist_photo(d, "calib_tips")
        fixed_tip, moving_tip = jaw_tips(img)
        move(arm, {"gripper": open_deg(arm)}, pick.FREE_DEG_S, "张开")
    finally:
        pick.park(arm)  # also when it stopped half way, the jaws a few mm off the floor
    shift = -down  # floor seen from TIP_ALIGN_M up rather than TIP_LOW_M: the whole strip matched
    face = (fixed_tip + moving_tip) / 2
    vis = img.copy()
    for pt, colour in ((fixed_tip, (0, 0, 255)), (moving_tip, (255, 0, 0)), (face + shift, (0, 255, 0))):
        cv2.drawMarker(vis, tuple(int(v) for v in pt), colour, cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite(str(d / "calib_tips_marked.jpg"), vis)
    TARGET_FILE.write_text(json.dumps({"fixed_tip_px": [round(float(v), 1) for v in fixed_tip],
                                       "moving_tip_px": [round(float(v), 1) for v in moving_tip],
                                       "rise_shift_px": [round(float(v), 1) for v in shift],
                                       "px_per_mm_tool": J_tool.round(3).tolist(),
                                       "tip_m": TIP_ALIGN_M, "run": d.name,
                                       "time": time.strftime("%Y-%m-%d %H:%M")}, indent=1) + "\n")
    print(f"合上的爪尖：固定爪 {np.round(fixed_tip, 1).tolist()} px，活动爪 {np.round(moving_tip, 1).tolist()} px；"
          f"从离地 {TIP_LOW_M * 1000:.0f} mm 抬到 {TIP_ALIGN_M * 1000:.0f} mm，地面上的点移动 "
          f"{np.round(shift, 1).tolist()} px；夹爪坐标系里每 mm 像素 {J_tool.round(2).tolist()}。存到 {TARGET_FILE.name}（{d}/calib_tips_marked.jpg）")


def cmd_auto(args, rig):
    d, run = look(args.model, rig)
    cmd_grab(args, rig, d, run)


def cmd_place(args, rig):
    """Put it down where the last pick took it from, or args.dx, args.dy (mm, base frame) from there
    and turned args.turn degrees: it lies across the jaws, so turning them turns it."""
    d, run = last_run()
    p = run["plan"]
    arm = rig.arm
    if args.dx or args.dy or args.turn:
        T = mount.load()
        a = np.radians(args.turn)
        ax = np.array(run["axis_base"])
        ax = np.array([np.cos(a) * ax[0] - np.sin(a) * ax[1], np.sin(a) * ax[0] + np.cos(a) * ax[1], 0.0])
        mid = np.array(run["mid_base_m"]) + np.array(run["nudge_mm"]) / 1000 + np.array([args.dx, args.dy, 0.0]) / 1000
        axis_arm = T[:3, :3].T @ ax
        seeds = [p["hover"]["joints"]] + [{**p["hover"]["joints"], "wrist_roll": p["hover"]["joints"]["wrist_roll"] + t}
                                          for t in (args.turn, args.turn - 180, args.turn + 180)]
        tilt = run.get("tilt_deg", 0.0)
        hover = solve_at(mid, axis_arm, HOVER_M, seeds, "放下处上方", tilt)
        pre = solve_at(mid, axis_arm, TIP_PRE_M, [hover], "放下处", tilt)
        if abs(pre["wrist_roll"] - hover["wrist_roll"]) > 30:
            raise SystemExit("放下处上下两个解手腕差半圈，不动")
        print(f"放到 (mm) {np.round(mid[:2] * 1000).astype(int).tolist()}，转 {args.turn:+.0f}°", flush=True)
        p = {"hover": {"joints": hover}, "pre": {"joints": pre}}
    move(arm, p["hover"]["joints"], pick.NEAR_DEG_S, "到放下处上方")
    move(arm, p["pre"]["joints"], pick.NEAR_DEG_S, f"下到爪尖离地 {TIP_PRE_M * 100:.0f} cm")
    move(arm, {"gripper": open_deg(arm)}, pick.FREE_DEG_S, "松开")
    move(arm, p["hover"]["joints"], pick.NEAR_DEG_S, "抬起")
    pick.park(arm)


def cmd_drop(args, rig):
    move(rig.arm, {"gripper": pick.OPEN_DEG}, pick.FREE_DEG_S, "松开")


def cmd_park(args, rig):
    pick.park(rig.arm)


def cmd_jawzone(args):
    """JAW_ZONE_FILE from the wrist photos the aims took (runs/*/align*_wrist.jpg): the jaws stay put in
    the picture while the floor and the things under it change."""
    photos = sorted(RUNS.glob("*/align*_wrist.jpg"))
    if len(photos) < 20:
        raise SystemExit(f"runs/ 里只有 {len(photos)} 张手腕照片，太少，认不出夹爪在哪")
    share = sum(jaws_in_wrist(cv2.imread(str(f)), zone=False).astype(np.float32) for f in photos) / len(photos)
    zone = cv2.dilate((share >= JAW_ZONE_SHARE).astype(np.uint8), np.ones((2 * JAW_ZONE_PAD_PX + 1,) * 2, np.uint8))
    cv2.imwrite(str(JAW_ZONE_FILE), zone * 255)
    print(f"{len(photos)} 张手腕照片，夹爪区域 {int(zone.sum())} 像素：{JAW_ZONE_FILE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("look", "check", "drop", "park", "all", "calib", "jawzone"):
        sub.add_parser(name)
    pl = sub.add_parser("place")
    for k in ("--dx", "--dy"):
        pl.add_argument(k, type=float, default=0.0, help="放到上次夹取处再偏这么多 (mm，底盘坐标)")
    pl.add_argument("--turn", type=float, default=0.0, help="放下时转这么多度")
    g = sub.add_parser("grab")
    for k in ("--dx", "--dy"):
        g.add_argument(k, type=float, default=0.0, help="瞄准再偏这么多 (mm，底盘坐标)")
    args = ap.parse_args()
    if args.cmd == "jawzone":  # photos only: no arm, no lock
        return cmd_jawzone(args)
    # One at a time: two of these on the arm's port at once broke a `calib` and left an `all` running
    # unseen (2026-11-04).
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("另一个 pick_small.py 正在用左臂，先等它结束或停掉它")
    if args.cmd is None:
        args.dx = args.dy = 0.0
    if args.cmd == "all":
        args.dx = args.dy = 0.0
    rig = Rig()
    try:
        {"look": cmd_look, "check": cmd_check, "grab": cmd_grab, "place": cmd_place, "drop": cmd_drop,
         "park": cmd_park, "all": cmd_all, "calib": cmd_calib, None: cmd_auto}[args.cmd](args, rig)
    finally:
        wrist_release()
        rig.close()


if __name__ == "__main__":
    main()
