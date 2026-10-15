"""Content types: from continuous dimensions to a yes-or-no table of what occurs in each clip."""
import numpy as np

from .content import refits


def standardize(columns):
    """Put every column on a standard scale: 0 = the average shot, 1 = one standard deviation."""
    return (columns - columns.mean(axis=0)) / columns.std(axis=0)


def clip_extremes(scaled, video_of_shot, clip_names):
    """Per clip and dimension: the value of the clip's most extreme shot, sign kept (0 if the clip has no shots here)."""
    # A clip "contains" something if any one of its shots shows it, so the most extreme shot
    # counts. Averaging would cancel a clip that shows a boat in one shot and a road in the next.
    out = np.zeros((len(clip_names), scaled.shape[1]))
    for c, vid in enumerate(clip_names):
        block = scaled[video_of_shot == vid]
        if len(block):
            out[c] = block[np.abs(block).argmax(axis=0), np.arange(block.shape[1])]
    return out


def clip_sides(scaled, video_of_shot, clip_names):
    """Clips x sides: how far each clip's furthest shot reaches on the high side and on the low side
    of each dimension (0 if it does not reach that side). Columns: all high sides, then all low sides.

    The two sides are measured separately, so a clip with one shot far out on the high side and
    another far out on the low side of the same dimension is credited with both."""
    k   = scaled.shape[1]
    out = np.zeros((len(clip_names), 2 * k))
    for c, vid in enumerate(clip_names):
        block = scaled[video_of_shot == vid]
        if len(block):
            out[c, :k] = np.clip(block.max(axis=0), 0, None)
            out[c, k:] = np.clip(-block.min(axis=0), 0, None)
    return out


def tag_shares(embeddings, rotated_all, keep, rotation, video_of_shot, clip_names, strong,
               rng, n_resamples, subsample):
    """Clips x sides: in what share of the refits does the clip get the tag for that side?

    The whole route to a tag is repeated on random subsets of the shots: PCA, rotation, the clip's
    furthest shot on each side, and the comparison with the line `strong`. A tag that only just passes the
    line in the full analysis fails in many refits; a clear one survives nearly all of them.
    """
    tagged    = np.zeros((len(clip_names), 2 * len(keep)))   # refits in which the clip had the tag
    took_part = np.zeros(len(clip_names))                    # refits in which the clip had at least one shot
    for idx, aligned, _ in refits(embeddings, rotated_all, rotation, rng, n_resamples, subsample):
        scaled     = standardize(aligned[:, keep])
        tagged    += clip_sides(scaled, video_of_shot[idx], clip_names) >= strong
        took_part += np.isin(clip_names, video_of_shot[idx])
    return tagged / np.maximum(took_part, 1)[:, None]
