"""MediaPipe hands, as a hand estimator rather than as part of the body frontend.

This is the code that used to live inside `mediapipe_fe.estimate`, moved out
unchanged. Three of its decisions are load bearing and are restated here because
they are the reason it cannot simply be replaced by pointing a hand model at the
frame:

* the region of interest comes from the pose wrist and elbow, pushed past the
  wrist along the forearm. That is not an optimisation. It is why hands resolve
  at all when the dancer is a few hundred pixels tall in a wide shot.
* one landmarker per side. They run in VIDEO mode, which carries tracking state
  between calls, so interleaving two crops through one instance would have each
  hand poison the other's prior.
* the side comes from which wrist the box was built on, not from the handedness
  label, which was unreliable on a tight crop: both hands of a reference frame
  came back "Right". Only the score is kept.

The timestamp sequence is the delicate part of having separated this. VIDEO mode
is stateful, so the pass must present the same stamps in the same order as the
fused loop did or the tracking state diverges and the output changes with no
error anywhere. The stamp advances on every frame the subject window marks
valid, whatever happens afterwards, and inference runs only on frames the body
estimator accepted, which is exactly what `pose.valid` records.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..hands import HandTracks
from .mediapipe_fe import (
    _base_options, _read_frames, _wrap_image, ensure_model, letterbox,
    to_ir_frame, wrist_roi)


def estimate_hands(source, subject, pose, *, hand_size: int = 224,
                   min_landmark_visibility: float = 0.5,
                   model_dir: Path | str | None = None,
                   download: bool = True,
                   min_detection_confidence: float = 0.3,
                   min_presence_confidence: float = 0.3) -> HandTracks:
    """Landmark both hands over the clip, using the body pose to find them."""
    from mediapipe.tasks.python import vision as V

    n = len(pose.valid)
    tracks = HandTracks.empty(n, estimator="mediapipe",
                              roi="pose_wrist_elbow", hand_size=hand_size)
    if pose.landmarks_px is None:
        raise ValueError(
            "the mediapipe hand estimator needs pose.landmarks_px, which this "
            "pose sequence does not carry. It is produced by a frontend that "
            "read a video; a sequence loaded from parameters has no pixels.")
    if subject is None:
        raise ValueError("the mediapipe hand estimator needs a subject window")

    import cv2

    hand_file = ensure_model("hand", model_dir, download)
    fps = float(subject.fps)
    subject_valid = np.asarray(subject.valid, bool)
    pose_valid = np.asarray(pose.valid, bool)
    px = np.asarray(pose.landmarks_px, np.float32)
    vis = np.asarray(pose.visibility, np.float32)

    lms = {side: V.HandLandmarker.create_from_options(
        V.HandLandmarkerOptions(
            base_options=_base_options(hand_file),
            running_mode=V.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence))
        for side in ("left", "right")}

    stamp = -1
    seen = 0
    try:
        for t, frame, frame_w, frame_h in _read_frames(source, n):
            if not subject_valid[t]:
                continue
            # Advances on every subject-valid frame, matching the fused loop,
            # which stamped before it knew whether the pose would be usable.
            stamp = max(stamp + 1, int(round(t * 1000.0 / fps)))
            if not pose_valid[t] or frame is None:
                continue
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            for col, side in enumerate(("left", "right")):
                roi = wrist_roi(px[t], vis[t], side, frame_w, frame_h,
                                min_visibility=min_landmark_visibility)
                if roi is None:
                    continue
                crop = rgba[roi[1]:roi[3], roi[0]:roi[2]]
                canvas, _, _, _ = letterbox(crop, hand_size)
                res = lms[side].detect_for_video(_wrap_image(canvas), stamp)
                if not res.hand_world_landmarks:
                    continue
                hw = np.array([[lm.x, lm.y, lm.z]
                               for lm in res.hand_world_landmarks[0]], np.float32)
                score = float(res.handedness[0][0].score) if res.handedness else 0.0
                target = tracks.left if side == "left" else tracks.right
                target[t] = to_ir_frame(hw)
                tracks.conf[t, col] = score
                seen += 1
    finally:
        for lm in lms.values():
            lm.close()

    tracks.meta["frames_with_a_hand"] = int(seen)
    return tracks


__all__ = ["estimate_hands"]
