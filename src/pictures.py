"""The picture check: do the dimensions found in the descriptions show up in the frames?"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from .media import find_cuts, shot_intervals


def frame_embeddings(model, video_path, frame_fps):
    """Image embedding and timestamp of every sampled frame of one clip."""
    # The frames are written to a temporary folder that is removed again afterwards.
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path),
                        "-vf", f"fps={frame_fps}", "-q:v", "2", f"{tmp}/%05d.jpg"], check=True)
        files   = sorted(Path(tmp).glob("*.jpg"))
        vectors = model.encode([Image.open(f).convert("RGB") for f in files], normalize_embeddings=True)
    return vectors, np.arange(len(files)) / frame_fps


def shot_picture_vectors(shots, videos, model, video_dir, frame_fps,
                         scene_threshold, min_shot_seconds, manual_cuts):
    """One picture vector per described shot: the average of the frames that fall inside the shot.

    Returns {row of `shots`: vector} and the clips that had to be left out because the cut detection
    and the description disagree on the number of shots, so shots cannot be matched to frames.
    """
    visual, skipped = {}, []
    for v in videos.itertuples():
        rows      = shots[shots["video_id"] == v.video_id].sort_values("shot")
        cuts, _   = find_cuts(v.video_id, v.duration, video_dir, scene_threshold, min_shot_seconds, manual_cuts)
        intervals = shot_intervals(cuts, v.duration)
        if len(intervals) != len(rows):
            skipped.append(v.video_id)
            continue
        vectors, times = frame_embeddings(model, video_dir / f"{v.video_id}.mp4", frame_fps)
        for row, (start, end) in zip(rows.index, intervals):
            inside = (times >= start) & (times < end)
            if inside.any():
                mean = vectors[inside].mean(axis=0)
                visual[row] = mean / np.linalg.norm(mean)
    return visual, skipped


def predicted_from_pictures(target, picture, source):
    """How well a score is predicted from pictures of broadcasts the model has not seen (correlation)."""
    # One broadcast at a time is left out: the model learns from the other broadcasts and predicts
    # the left-out one. A dimension that only reflects a broadcaster's visual style cannot be
    # predicted this way, so a high value means the content itself is visible in the pictures.
    guess = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 2, 12)), picture, target,
                              groups=source, cv=LeaveOneGroupOut())
    return np.corrcoef(target, guess)[0, 1]
