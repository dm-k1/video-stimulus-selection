"""Working with the video files: reading their properties, normalizing them, finding the cuts,
and joining clips into a compilation.

Everything here calls ffmpeg or ffprobe, which are installed with the conda environment.
"""
import json
import re
import subprocess

import numpy as np


def run(cmd):
    """Run a command; stop with its error message if it fails. Returns (stdout, stderr)."""
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{res.stderr[-2000:]}")
    return res.stdout, res.stderr


# ---------- properties of a file ----------
def read_specs(path):
    """Technical properties stored in one video file (read with ffprobe, nothing is decoded)."""
    out, _ = run(["ffprobe", "-v", "error", "-show_entries",
                  "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels,duration",
                  "-of", "json", path])
    j = json.loads(out)
    video = next(s for s in j["streams"] if s["codec_type"] == "video")
    audio = next((s for s in j["streams"] if s["codec_type"] == "audio"), {})     # {} if the file has no sound
    # "duration" is the length of the file; "video_duration" that of the picture alone, which can be a
    # few frames shorter when the sound runs on.
    return {"duration": float(j["format"]["duration"]),
            "video_duration": float(video.get("duration", j["format"]["duration"])),
            "width": video["width"], "height": video["height"], "fps": video["r_frame_rate"],
            "video_codec": video["codec_name"], "pix_fmt": video["pix_fmt"],
            "audio_codec": audio.get("codec_name"), "sample_rate": audio.get("sample_rate"),
            "channels": audio.get("channels")}


def measure_loudness(path):
    """Integrated loudness of the audio track in LUFS (EBU R128)."""
    # ffmpeg's ebur128 filter prints a running measurement to stderr; the last "I:" value is the
    # integrated loudness of the whole clip.
    _, log = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn",
                  "-af", "ebur128", "-f", "null", "-"])
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", log)[-1])


# ---------- normalizing ----------
def reformat_clip(path, has_audio, target_specs):
    """Re-encode one clip in place to the target size, frame rate and formats."""
    tmp = path.with_suffix(".tmp.mp4")          # ffmpeg cannot overwrite the file it is reading
    w, h, fps = target_specs["width"], target_specs["height"], target_specs["fps"]
    # Scale to fit inside the target frame without distortion, then pad with black bars to fill it.
    video_filter = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}")

    if has_audio:
        silent_input, audio_map, shortest = [], "0:a:0", []
    else:   # add a silent track so every file has identical streams
        silent_input = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_map, shortest = "1:a:0", ["-shortest"]

    run(["ffmpeg", "-y", "-v", "error", "-i", path, *silent_input,
         "-map", "0:v:0", "-map", audio_map,
         "-vf", video_filter, "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",          # crf 18: visually lossless
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         *shortest, "-movflags", "+faststart", tmp])
    tmp.replace(path)


def level_loudness(path, measured, target_lufs):
    """Shift the volume of one clip in place so it lands on the target. The picture is copied untouched."""
    # A plain volume shift (not a compressor): the dynamics within the clip stay as they were.
    tmp = path.with_suffix(".tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-c:v", "copy",
         "-af", f"volume={target_lufs - measured:.2f}dB",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", tmp])
    tmp.replace(path)


# ---------- shots ----------
def detect_cuts(path, scene_threshold):
    """Times (in seconds) at which the picture changes abruptly."""
    # ffmpeg scores every frame on how much it differs from the previous one ("scene", 0 to 1)
    # and prints the timestamp of each frame that scores above the threshold.
    out, _ = run(["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-an",
                  "-vf", f"select='gt(scene,{scene_threshold})',metadata=print:file=-",
                  "-f", "null", "-"])
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", out)]


def clean_cuts(cuts, duration, min_shot_seconds):
    """Split detected cuts into trusted ones and doubtful ones (in a burst, or too close to an edge)."""
    # Several detections close together usually mean a flash, a fade or fast motion, not real
    # cuts; a detection right at the start or end is a leftover frame from trimming.
    kept, dropped = [], []
    for i, t in enumerate(cuts):
        in_burst  = any(abs(t - other) < min_shot_seconds for j, other in enumerate(cuts) if j != i)
        near_edge = t < min_shot_seconds or duration - t < min_shot_seconds
        (dropped if in_burst or near_edge else kept).append(t)
    return kept, dropped


def find_cuts(video_id, duration, video_dir, scene_threshold, min_shot_seconds, manual_cuts):
    """The cuts of one clip and the detections that were dropped. Cut times entered by hand win over detection."""
    if video_id in manual_cuts:
        return manual_cuts[video_id], []
    return clean_cuts(detect_cuts(video_dir / f"{video_id}.mp4", scene_threshold), duration, min_shot_seconds)


def shot_intervals(cuts, duration):
    """Start and end time of every shot, given the cuts: start of clip, each cut, end of clip."""
    edges = [0.0, *cuts, duration]
    return list(zip(edges[:-1], edges[1:]))


def shot_seconds(shots, videos, video_dir, scene_threshold, min_shot_seconds, manual_cuts):
    """Screen time of every described shot, in seconds, in the order of the rows of `shots`.

    Taken from the cut detection for clips where it finds as many shots as Gemini described. Where the
    two disagree, the shots cannot be matched to times, so the clip's length is split evenly over its
    described shots. Returns the seconds and the clips that were split evenly.
    """
    seconds, split_evenly = np.zeros(len(shots)), []
    clip_of_row, shot_number = shots["video_id"].to_numpy(), shots["shot"].to_numpy()
    for v in videos.itertuples():
        rows = np.where(clip_of_row == v.video_id)[0]
        if len(rows) == 0:
            continue
        rows      = rows[np.argsort(shot_number[rows])]                 # shots in order of appearance
        cuts, _   = find_cuts(v.video_id, v.duration, video_dir, scene_threshold, min_shot_seconds, manual_cuts)
        intervals = shot_intervals(cuts, v.duration)
        if len(intervals) == len(rows):
            seconds[rows] = [end - start for start, end in intervals]
        else:
            seconds[rows] = v.duration / len(rows)
            split_evenly.append(v.video_id)
    return seconds, split_evenly


# ---------- the compilation ----------
def join_clips(video_ids, video_dir, out_path, sound):
    """Join whole clips, in the given order, into one video, with or without their sound.

    Re-encoded (H.264 CRF 18; AAC 192 kbps if with sound) through ffmpeg's concat filter, so the
    joins are frame-exact and the result is one continuous stream. All clips already share format
    and loudness (step 1), so nothing else is adjusted. Without sound the video has no audio track.
    """
    inputs, pads = [], ""
    for n, vid in enumerate(video_ids):
        inputs += ["-i", video_dir / f"{vid}.mp4"]
        pads   += f"[{n}:v][{n}:a]" if sound else f"[{n}:v]"
    if sound:
        concat, maps = f"{pads}concat=n={len(video_ids)}:v=1:a=1[v][a]", ["-map", "[v]", "-map", "[a]"]
        audio = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        concat, maps = f"{pads}concat=n={len(video_ids)}:v=1:a=0[v]", ["-map", "[v]"]
        audio = ["-an"]
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", concat, *maps,
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "25",
         *audio, "-movflags", "+faststart", out_path])
