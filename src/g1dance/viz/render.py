"""Offscreen rendering of a motion or a policy rollout to mp4.

A dance is judged by eye, so this is not a nicety. Two things it must make easy
to see: whether the retargeted reference is right at all (`render_reference`,
pure kinematics, no physics), and how far the trained policy drifts from that
reference (`render_comparison`, side by side). Separating those two is what
stops a bad reference being mistaken for a failed policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from ..motion.ir import ReferenceMotion
from ..sim.model import G1Model

DEFAULT_W, DEFAULT_H = 640, 480


@dataclass
class Camera:
    distance: float = 3.0
    azimuth: float = 135.0
    elevation: float = -15.0
    lookat_z: float = 0.9

    def apply(self, cam: mujoco.MjvCamera, centre: np.ndarray) -> None:
        cam.distance = self.distance
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation
        cam.lookat[:] = [centre[0], centre[1], self.lookat_z]


def facing_camera(motion: ReferenceMotion, base: Camera | None = None) -> Camera:
    """A camera in front of the dancer, which is where the source camera was.

    A fixed azimuth shows each path from an arbitrary side, because the paths
    put the video camera in different places in the world: MediaPipe's hip
    centred frame faces the dancer along +x, GVHMR's gravity-view frame along
    +y. Both agree on one thing, that a dancer mostly faces the camera, so the
    median reference heading recovers where it stood without either
    convention being written down here.

    MuJoCo looks along (cos az, sin az), so the camera in front of a body
    facing heading h sits at az = h + 180 degrees. The circular mean, not the
    plain median, because headings wrap.
    """
    q = np.asarray(motion.root_quat, np.float64)
    h = np.arctan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                   1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    v = np.asarray(motion.valid, bool)
    h = h[v] if v.any() else h
    mean = np.degrees(np.arctan2(np.sin(h).mean(), np.cos(h).mean()))
    base = base or Camera()
    return Camera(distance=base.distance, azimuth=float(mean + 180.0),
                  elevation=base.elevation, lookat_z=base.lookat_z)


class Renderer:
    """Thin wrapper over mujoco.Renderer that tracks the pelvis."""

    def __init__(self, model: mujoco.MjModel, width: int = DEFAULT_W,
                 height: int = DEFAULT_H, camera: Camera | None = None) -> None:
        self._r = mujoco.Renderer(model, height=height, width=width)
        self._cam = camera or Camera()
        self._model = model

    def frame(self, data: mujoco.MjData, centre: np.ndarray,
              camera: Camera | None = None) -> np.ndarray:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        (camera or self._cam).apply(cam, centre)
        self._r.update_scene(data, camera=cam)
        return self._r.render()

    def close(self) -> None:
        self._r.close()


def _writer(path: Path, fps: float):
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, macro_block_size=1,
                              codec="libx264", quality=7)


def _set_pose(g1: G1Model, d: mujoco.MjData, motion: ReferenceMotion, i: int) -> None:
    d.qpos[:3] = motion.root_pos[i]
    d.qpos[3:7] = motion.root_quat[i]
    d.qpos[g1.body_qpos] = motion.qpos_ref[i]
    d.qpos[g1.hand_qpos] = motion.finger_ref[i]


def render_reference(motion: ReferenceMotion, out: Path, g1: G1Model | None = None,
                     stride: int = 1, width: int = DEFAULT_W,
                     height: int = DEFAULT_H,
                     camera: Camera | None = None) -> Path:
    """Kinematic playback: mj_forward only, no physics, no policy.

    If this looks wrong, the problem is upstream of RL and no amount of
    training will fix it.
    """
    g1 = g1 or G1Model()
    m, d = g1.model, g1.data()
    r = Renderer(m, width, height, camera or facing_camera(motion))
    try:
        with _writer(Path(out), motion.fps / stride) as w:
            for i in range(0, motion.n_frames, stride):
                if not motion.valid[i]:
                    continue
                _set_pose(g1, d, motion, i)
                mujoco.mj_forward(m, d)
                w.append_data(r.frame(d, d.xpos[g1.pelvis_id]))
    finally:
        r.close()
    return Path(out)


def render_rollout(states: np.ndarray, out: Path, fps: float = 50.0,
                   g1: G1Model | None = None, width: int = DEFAULT_W,
                   height: int = DEFAULT_H) -> Path:
    """Render a recorded qpos trajectory (T, nq) from a policy rollout."""
    g1 = g1 or G1Model()
    m, d = g1.model, g1.data()
    r = Renderer(m, width, height)
    try:
        with _writer(Path(out), fps) as w:
            for q in states:
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                w.append_data(r.frame(d, d.xpos[g1.pelvis_id]))
    finally:
        r.close()
    return Path(out)


def render_comparison(motion: ReferenceMotion, states: np.ndarray, out: Path,
                      fps: float = 50.0, g1: G1Model | None = None,
                      width: int = DEFAULT_W, height: int = DEFAULT_H,
                      start_frame: int = 0,
                      camera: Camera | None = None) -> Path:
    """Reference on the left, policy on the right, same camera, same clock.

    The same camera, not one per panel: the tracker steers toward the
    reference's own heading, so a policy that fails to turn should be seen
    facing away rather than have the camera turn with it.
    """
    g1 = g1 or G1Model()
    m = g1.model
    dref, dpol = g1.data(), g1.data()
    r = Renderer(m, width, height, camera or facing_camera(motion))
    n = min(len(states), motion.n_frames - start_frame)
    try:
        with _writer(Path(out), fps) as w:
            for k in range(n):
                _set_pose(g1, dref, motion, start_frame + k)
                mujoco.mj_forward(m, dref)
                left = r.frame(dref, dref.xpos[g1.pelvis_id])
                dpol.qpos[:] = states[k]
                mujoco.mj_forward(m, dpol)
                right = r.frame(dpol, dpol.xpos[g1.pelvis_id])
                w.append_data(np.concatenate([left, right], axis=1))
    finally:
        r.close()
    return Path(out)


def rollout_states(motion_path, config_path, seed: int = 0,
                   max_steps: int = 500, controller=None,
                   start_frame: int | None = None,
                   outcome: dict | None = None) -> tuple[np.ndarray, int]:
    """Run one episode and record `qpos` per control step, for rendering.

    `control.evaluate` deliberately records only the scalars its metrics need, so this
    lives here: rendering is the one consumer that wants the full state, and
    keeping it out of the metrics path stops eval paying for it.

    Returns the (T, nq) trajectory and the reference frame the episode started
    on, which `render_comparison` needs to line the two halves up.

    `start_frame` plays from a chosen frame rather than a random one, and
    `max_steps` then also lifts the environment's own episode cap, so a whole
    clip can be played through as one episode. It still ends at a fall or at
    the end of the segment, which is the honest place for it to end.

    `outcome`, if given, is filled with how the episode ended: `terminated`
    is a fall or a tracking failure, as opposed to running out of reference.
    """
    from ..control.controllers import OpenLoopController
    from ..sim.env import DanceTrackEnv

    env = DanceTrackEnv(motion_path, config_path)
    env.seed(seed)
    if start_frame is not None:
        env.cfg["max_episode_steps"] = max(int(env.cfg["max_episode_steps"]),
                                           int(max_steps))
    # No controller replays the reference open loop: PD straight to it.
    if controller is None:
        controller = OpenLoopController()

    obs, priv = env.reset(start_frame=start_frame)
    # `env.frame` indexes the env's own resampled track. It lines up with the
    # ReferenceMotion's frames only because both run at the control rate; a
    # motion at another fps would need mapping through env.track.
    start = int(getattr(env, "frame", 0))
    states = [env.data.qpos.copy()]
    controller.reset(env)
    absolute = bool(getattr(controller, "absolute", False))
    for _ in range(max_steps):
        a = controller.act(env, obs, priv)
        obs, priv, _, term, trunc, info = env.step(a, absolute=absolute)
        states.append(env.data.qpos.copy())
        if term or trunc:
            if outcome is not None:
                outcome.update(terminated=bool(term),
                               reason=str((info or {}).get("reason", "")))
            break
    env.close()
    return np.asarray(states), start


#: Body edges between MediaPipe pose slots that every frontend fills, the
#: COCO-17 subset GVHMR's ViTPose supplies included. Left limbs first.
_SKELETON_LEFT = (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                  ("left_shoulder", "left_hip"), ("left_hip", "left_knee"),
                  ("left_knee", "left_ankle"))
_SKELETON_RIGHT = tuple((a.replace("left", "right"), b.replace("left", "right"))
                        for a, b in _SKELETON_LEFT)
_SKELETON_MID = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))


def _draw_skeleton(img: np.ndarray, px: np.ndarray, to_panel) -> None:
    """Draw the 2D pose on a BGR panel; `to_panel` maps frame pixels into it.

    A slot left at zero is one the frontend had no point for, so edges touching
    it are skipped rather than drawn to the corner of the frame.
    """
    import cv2

    from ..motion.ir import MP_POSE

    def pt(name):
        xy = px[MP_POSE[name]]
        if not np.all(np.isfinite(xy)) or not np.any(xy):
            return None
        return to_panel(xy)

    # BGR: left limbs orange, right limbs cyan, so a mirrored pose is visible
    for edges, colour in ((_SKELETON_LEFT, (0, 165, 255)),
                          (_SKELETON_RIGHT, (255, 200, 0)),
                          (_SKELETON_MID, (230, 230, 230))):
        for a, b in edges:
            pa, pb = pt(a), pt(b)
            if pa is not None and pb is not None:
                cv2.line(img, pa, pb, colour, 3, cv2.LINE_AA)
    names = {n for e in _SKELETON_LEFT + _SKELETON_RIGHT for n in e}
    for n in names | {"nose"}:
        p = pt(n)
        if p is not None:
            cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)


def steady_boxes(bbox: np.ndarray, fps: float, window_s: float = 0.5,
                 size_pct: float = 95.0) -> np.ndarray:
    """Per-frame crop boxes for a steady camera: one size, a smoothed centre.

    Cropping each frame to that frame's detection box zooms the panel whenever
    the dancer raises an arm or crouches. On koidance the box runs from 450 to
    657 px and changes by up to 14 percent between frames, which reads as the
    camera pumping. Fixing the size at a high percentile keeps the dancer in
    frame at their widest, and a moving average over `window_s` turns centre
    jitter into a pan. Frames with no box borrow their neighbours' centre.
    """
    b = np.asarray(bbox, np.float64)
    ok = np.all(np.isfinite(b), axis=1) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    if not ok.any():
        return b.copy()
    side = np.maximum(b[ok, 2] - b[ok, 0], b[ok, 3] - b[ok, 1])
    half = 0.5 * float(np.percentile(side, size_pct))
    t = np.arange(len(b))
    cx = np.interp(t, t[ok], 0.5 * (b[ok, 0] + b[ok, 2]))
    cy = np.interp(t, t[ok], 0.5 * (b[ok, 1] + b[ok, 3]))
    k = max(int(round(window_s * fps)) | 1, 1)          # odd, centred
    if k > 1 and len(b) > k:
        pad = k // 2
        kern = np.ones(k) / k
        cx = np.convolve(np.pad(cx, pad, mode="edge"), kern, mode="valid")
        cy = np.convolve(np.pad(cy, pad, mode="edge"), kern, mode="valid")
    return np.stack([cx - half, cy - half, cx + half, cy + half], axis=1)


def _source_panel(cap, src_frame: int, bbox, size: tuple[int, int],
                  pad: float = 0.35,
                  skeleton: np.ndarray | None = None) -> np.ndarray:
    """One frame of the source video, cropped to the dancer and letterboxed.

    Cropping matters for legibility: the subject is a small part of a wide
    shot, so an uncropped panel next to two close-framed robots reads as three
    unrelated pictures rather than a comparison.

    `skeleton`, the frame's (33, 2) pose in full frame pixels, is drawn through
    the same crop and scale as the image, so it lands on the dancer.
    """
    import cv2

    w, h = size
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_frame))
    ok, frame = cap.read()
    if not ok:
        return np.zeros((h, w, 3), np.uint8)
    fh, fw = frame.shape[:2]
    cx0 = cy0 = 0
    if bbox is not None and np.all(np.isfinite(bbox)) and bbox[2] > bbox[0]:
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * (0.5 + pad)
        ar = w / h
        hx, hy = min(half * ar, fw / 2), min(half, fh / 2)
        # slid back inside the frame rather than clipped, so the crop keeps its
        # size at the edges and the panel does not zoom there
        cx = min(max(cx, hx), fw - hx)
        cy = min(max(cy, hy), fh - hy)
        x0, x1 = int(round(cx - hx)), int(round(cx + hx))
        y0, y1 = int(round(cy - hy)), int(round(cy + hy))
        if x1 > x0 + 4 and y1 > y0 + 4:
            frame = frame[y0:y1, x0:x1]
            cx0, cy0 = x0, y0
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    out = np.zeros((h, w, 3), np.uint8)
    rs = cv2.resize(frame, (max(int(fw * s), 1), max(int(fh * s), 1)))
    oy, ox = (h - rs.shape[0]) // 2, (w - rs.shape[1]) // 2
    out[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs
    if skeleton is not None:
        _draw_skeleton(out, np.asarray(skeleton, np.float64), lambda xy: (
            int(round((xy[0] - cx0) * s + ox)), int(round((xy[1] - cy0) * s + oy))))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def render_triptych(video: Path, motion: ReferenceMotion, states: np.ndarray,
                    out: Path, tracks: Path | None = None,
                    g1: G1Model | None = None, width: int = DEFAULT_W,
                    height: int = DEFAULT_H, start_frame: int = 0,
                    label: bool = True, camera: Camera | None = None,
                    policy_yaw_offset: float = 0.0,
                    pose=None) -> Path:
    """Source video, retargeted reference and policy, side by side.

    Given `pose`, a `PoseSequence` carrying `landmarks_px`, a fourth panel shows
    the source again with the 2D pose the reference was estimated from drawn on
    it, and the four are laid out two by two: the footage on top, the robots
    below. That panel is where a wrong limb is caught at its origin, before
    retargeting has had a chance to make it look plausible.

    The kinematic reference only shows whether the policy matches what it was
    asked to do. Putting the footage beside it shows whether what it was asked
    to do resembles the dance, which is the question retargeting can silently
    get wrong.

    The three clocks are tied to the motion's frame index: panel k shows
    reference frame `start_frame + k`, and the source frame at the same wall
    time, which is `(start_frame + k) / motion.fps * source_fps`.

    `policy_yaw_offset`, in radians, is the heading the robot's world is turned
    by relative to the reference's. It is zero for a tracker steering toward the
    reference's own heading, and the anchoring rotation for an anchored one,
    whose reference was turned so its start frame faces the robot. Without it
    an anchored robot is filmed from the side it was never asked to face.
    """
    import cv2

    g1 = g1 or G1Model()
    m = g1.model
    dref, dpol = g1.data(), g1.data()
    cam = camera or facing_camera(motion)
    pol_cam = replace(cam, azimuth=cam.azimuth + np.degrees(policy_yaw_offset))
    r = Renderer(m, width, height, cam)
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or motion.fps

    px = None
    if pose is not None and getattr(pose, "landmarks_px", None) is not None:
        px = np.asarray(pose.landmarks_px, np.float64)
        kind = (pose.meta or {}).get("landmarks_px", "") or \
            (pose.meta or {}).get("frontend", "")
        # Hershey has no parentheses worth reading, hence the colon.
        skel_label = f"2D pose: {kind.split('_')[0]}" if kind else "2D pose"

    bbox = None
    if tracks is not None and Path(tracks).exists():
        from ..track import TrackResult
        bbox = steady_boxes(np.asarray(TrackResult.load(tracks).bbox),
                            src_fps)

    n = min(len(states), motion.n_frames - start_frame)
    try:
        with _writer(Path(out), motion.fps) as w:
            for k in range(n):
                f = start_frame + k
                sf = int(round(f / motion.fps * src_fps))
                bb = bbox[min(sf, len(bbox) - 1)] if bbox is not None else None
                src = _source_panel(cap, sf, bb, (width, height))

                _set_pose(g1, dref, motion, f)
                mujoco.mj_forward(m, dref)
                ref = r.frame(dref, dref.xpos[g1.pelvis_id])

                dpol.qpos[:] = states[k]
                mujoco.mj_forward(m, dpol)
                pol = r.frame(dpol, dpol.xpos[g1.pelvis_id], pol_cam)

                if px is None:
                    tiles = [[src, ref, pol]]
                    names = [["source", "reference", "policy"]]
                else:
                    skel = _source_panel(cap, sf, bb, (width, height),
                                         skeleton=px[min(sf, len(px) - 1)])
                    tiles = [[src, skel], [ref, pol]]
                    names = [["source", skel_label], ["reference", "policy"]]
                panel = np.ascontiguousarray(np.concatenate(
                    [np.concatenate(row, axis=1) for row in tiles], axis=0))
                if label:
                    for r_i, row in enumerate(names):
                        for c_i, t in enumerate(row):
                            cv2.putText(panel, t, (12 + c_i * width,
                                                   26 + r_i * height),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 255, 255), 2, cv2.LINE_AA)
                w.append_data(panel)
    finally:
        cap.release()
        r.close()
    return Path(out)


def _heading(q: np.ndarray) -> float:
    q = np.asarray(q, np.float64)
    return float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


def render_showcase(motion_path, config_path, out, *, controller=None,
                    video=None, tracks=None, pose_path=None,
                    seed: int = 0, start_frame: int | None = None,
                    full: bool = False) -> tuple[Path, dict]:
    """Roll out once and render it the way a result is shown to people.

    With `video` this is the triptych, or the two by two layout when a pose
    carrying pixel landmarks is available; without it, reference beside policy.
    `full` plays the whole clip from frame 0 as one episode. Returns the path
    and what was rendered, so a caller can report a fall rather than imply a
    clip played through.

    The robot camera turns by the rotation between the robot's world and the
    reference's, read from the rollout itself: the robot's heading at the
    start against the reference's at the start frame. That is zero for a
    tracker steering toward the reference's own heading, and the anchoring
    rotation for an anchored one.
    """
    from ..motion.ir import PoseSequence

    motion = ReferenceMotion.load(motion_path)
    if full:
        start_frame, max_steps = 0, motion.n_frames
    else:
        max_steps = 500
    outcome: dict = {"terminated": False, "reason": "step limit"}
    states, start = rollout_states(motion_path, config_path,
                                   seed=seed, controller=controller,
                                   start_frame=start_frame, max_steps=max_steps,
                                   outcome=outcome)
    anchored = bool(getattr(controller, "_anchor", False))
    offset = (_heading(states[0][3:7]) - _heading(motion.root_quat[start])
              if anchored else 0.0)
    out = Path(out)
    if video is not None:
        pose = None
        if pose_path is not None and Path(pose_path).exists():
            pose = PoseSequence.load(pose_path)
        render_triptych(Path(video), motion, states, out, tracks=tracks,
                        start_frame=start, policy_yaw_offset=offset, pose=pose)
    else:
        render_comparison(motion, states, out, fps=motion.fps,
                          start_frame=start)
    info = {"start_frame": int(start), "steps": int(len(states) - 1),
            "frames_available": int(motion.n_frames - start),
            "terminated": outcome["terminated"], "reason": outcome["reason"],
            "anchored": anchored, "yaw_offset_deg": float(np.degrees(offset))}
    return out, info
