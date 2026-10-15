"""The content space: from shot descriptions to stable, interpretable dimensions."""
import re

import numpy as np
import pandas as pd
from factor_analyzer.rotator import Rotator
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

# Words about framing, removed before embedding so that shots compare on content.
CAMERA_WORDS = (r"\b(medium|wide|close-up|closeup|high-angle|low-angle|eye-level|aerial|elevated|"
                r"tracking|static|handheld|panning|overhead|shot|shots|camera|angle|view)\b")


# ---------- descriptions to vectors ----------
def load_shots(desc_path):
    """One row per described shot, with camera vocabulary removed so shots compare on content, not framing."""
    rows = []
    for r in pd.read_csv(desc_path, encoding="utf-8-sig").itertuples():
        for line in r.shots.split("\n"):                       # "1. text", "2. text", ...
            number, text = line.split(". ", 1)
            rows.append({"video_id": r.video_id, "source_id": r.video_id.rsplit("_", 1)[0],
                         "shot": int(number), "description": text})
    shots = pd.DataFrame(rows)
    # "description" keeps the original text for printing; "content" is what gets embedded.
    shots["content"] = shots["description"].str.replace(CAMERA_WORDS, " ", regex=True, flags=re.IGNORECASE)
    return shots


def embed(shots, model_name):
    """Turn every shot description into a vector; similar content gives similar vectors."""
    from sentence_transformers import SentenceTransformer       # imported here: loading it is slow
    # Vectors are scaled to length 1, so distances between them depend on direction only.
    return SentenceTransformer(model_name).encode(shots["content"].tolist(), normalize_embeddings=True)


def count_real_dimensions(embeddings, rng, n_scrambles, noise_quantile):
    """Parallel analysis: how many leading dimensions are stronger than in scrambled data."""
    centered = embeddings - embeddings.mean(axis=0)
    strength = np.linalg.svd(centered, compute_uv=False) ** 2          # variation held by each dimension
    # Scrambling shuffles every column separately: each keeps its values, but all relations
    # between columns are destroyed. What remains is the strength dimensions get by chance.
    noise    = np.array([np.linalg.svd(rng.permuted(centered, axis=0), compute_uv=False) ** 2
                         for _ in range(n_scrambles)])
    limit    = np.percentile(noise, noise_quantile, axis=0)
    real     = strength > limit
    # Count from the first dimension up to (not including) the first one that fails.
    return (len(real) if real.all() else int(np.argmin(real))), strength, limit


# ---------- rotation and stability ----------
def rotate(points, method):
    """Rotate the dimensions so that each one separates a few shots clearly from the rest.
    "varimax" keeps the dimensions uncorrelated; "oblimin" allows them to correlate."""
    return Rotator(method=method, normalize=False).fit_transform(points)


def oblimin_correlations(scores):
    """Correlations between the dimensions when they are allowed to correlate (oblimin rotation).
    Measured directly on the rotated shot scores, not taken from the rotator's factor model (phi_),
    which treats its input as loadings and so does not give exactly these correlations."""
    return np.corrcoef(rotate(scores, "oblimin").T)


def refits(embeddings, rotated_all, rotation, rng, n_resamples, subsample):
    """Redo PCA and rotation on random subsets of the shots, and line each result up with the full analysis.

    Yields, per refit: the shots that took part, the refitted dimensions put in the original order and
    with the original sign, and how strongly each one correlates with its original (0 to 1).
    """
    n_dims = rotated_all.shape[1]
    for _ in range(n_resamples):
        idx   = rng.choice(len(embeddings), int(subsample * len(embeddings)), replace=False)
        refit = rotate(PCA(n_components=n_dims, svd_solver="full").fit_transform(embeddings[idx]), rotation)
        corr  = np.corrcoef(rotated_all[idx].T, refit.T)[:n_dims, n_dims:]      # original x refitted dimensions
        # A refit returns its dimensions in arbitrary order and with arbitrary sign, so they are
        # paired one-to-one with the originals and flipped where they came out mirrored.
        _, partner = linear_sum_assignment(-np.abs(corr))
        matched    = corr[np.arange(n_dims), partner]
        yield idx, refit[:, partner] * np.sign(matched), np.abs(matched)


def dimension_stability(embeddings, rotated_all, rotation, rng, n_resamples, subsample):
    """Per rotated dimension: how well it comes back, on average, when the analysis is redone on subsets."""
    total = np.zeros(rotated_all.shape[1])
    for _, _, match in refits(embeddings, rotated_all, rotation, rng, n_resamples, subsample):
        total += match
    return total / n_resamples


# ---------- interpretation ----------
def typical_words(shots, column, n_words=6):
    """The words most typical of the high side and of the low side of one dimension."""
    # Only words that occur in at least three descriptions are considered.
    counter  = CountVectorizer(binary=True, stop_words="english", min_df=3)
    has_word = counter.fit_transform(shots["content"]).toarray()         # shots x words, 1 if the word occurs
    words    = np.array(counter.get_feature_names_out())
    # For every word: the average score on this dimension of the shots that contain it.
    # The words with the highest averages describe the high side, the lowest the low side.
    order    = ((has_word * column[:, None]).sum(axis=0) / has_word.sum(axis=0)).argsort()
    return ", ".join(words[order[::-1][:n_words]]), ", ".join(words[order[:n_words]])
