"""Selecting a compilation: count every possible combination of clips, list the ones that qualify,
rank them, and check how robust and representative the best ones are.

A compilation is a set of clips; the order of the clips plays no role here.
"""
import numpy as np
import pandas as pd


# ---------- continuous measures ----------
def shot_weights(seconds, how):
    """How much each shot counts in the gap and the mix, from its screen time.

    "seconds": every second of footage counts equally (a 14 s shot weighs 10 times a 1.4 s shot)
    "sqrt":    longer shots count more, with diminishing returns (14 s vs 1.4 s: about 3 times)
    "shot":    every shot counts once, however long it is
    """
    if how == "seconds":
        return np.asarray(seconds, dtype=float)
    if how == "sqrt":
        return np.sqrt(seconds)
    if how == "shot":
        return np.ones(len(seconds))
    raise ValueError(f"Unknown shot weighting: {how!r} (use 'seconds', 'sqrt' or 'shot')")


def coverage_gaps(clips, shot_clip, distances):
    """For every shot in the set: distance to the nearest shot that is in the compilation."""
    # A shot that is in the compilation has distance 0 to itself. A small average means that
    # every shot in the set, including the one-off ones, has something similar in the compilation.
    return distances[:, np.isin(shot_clip, clips)].min(axis=1)


def content_mix(clips, shot_clip, distances, weight):
    """How far the compilation's mix of content is from the mix of the whole set, every shot counted
    by its weight (see shot_weights). Returns (mix, spread).

    mix:    energy distance between the compilation's footage and all footage (pairs of a shot with
            itself left out). Random compilations land near one value: above it the content is too
            narrow (a few kinds dominate), below it too broad (rare content over-represented).
    spread: average distance between two moments of the compilation, relative to the whole set
            (above 1 = more varied than the set).
    """
    w_in = np.where(np.isin(shot_clip, list(clips)), weight, 0.0)

    def between(wa, wb):
        # Average distance between a moment of footage a and a moment of footage b, from different shots.
        return (wa @ distances @ wb) / (wa.sum() * wb.sum() - (wa * wb).sum())

    whole, within = between(weight, weight), between(w_in, w_in)
    return 2 * between(w_in, weight) - within - whole, within / whole


def side_coverage(clips, sides, ceiling):
    """Average share of every side of every dimension that the compilation shows, each up to the ceiling."""
    reach    = np.minimum(sides[list(clips)].max(axis=0), ceiling)     # how far the compilation reaches per side
    possible = np.minimum(sides.max(axis=0), ceiling)                  # how far any clip in the set reaches
    return (reach / np.maximum(possible, 1e-9)).mean()


def type_strength(sides, has_type, type_sides, strong, strength_ceiling):
    """Clips x content types: how clearly each clip shows each type (0 = absent, 1 = a perfectly clear example).

    `type_sides` gives, per content type, the column of `sides` it comes from, or None for a type
    that was added by hand and has no dimension behind it.
    """
    strength = np.zeros(has_type.shape)
    for t, s in enumerate(type_sides):
        reach = sides[:, s] if s is not None else np.zeros(len(sides))
        reach = np.where(has_type[:, t], np.maximum(reach, strong), reach)           # a tagged clip counts as clear
        strength[:, t] = np.minimum(reach, strength_ceiling) / strength_ceiling
    return strength


# ---------- sets of content types as whole numbers ----------
def type_bits(has_type):
    """Each clip's set of content types packed into one whole number (one bit per type),
    so that combining clips becomes a single "or" operation."""
    return (has_type.astype(np.int64) << np.arange(has_type.shape[1])).sum(axis=1)


def bit_counts(n_types):
    """For every possible set of content types (as a number): how many types it holds."""
    return np.array([bin(v).count("1") for v in range(2 ** n_types)], dtype=np.int8)


# ---------- part 1: count all combinations ----------
def count_combinations(durations, has_type, step, up_to_seconds):
    """Exact number of combinations of clips, by number of content types shown and by total length.

    The combinations are not listed (with 46 clips there are 2^46). They are counted: for the
    question "how many types, how long" it only matters which set of types a combination has and
    how long it is. ways[set, length] holds the number of combinations that end up there. The clips
    are added one at a time: every combination so far either leaves the clip out (stays where it is)
    or takes it (moves to the set with the clip's types added, and to a greater length).

    Returns an array (number of types) x (length in units of `step` seconds). Clip lengths are
    rounded to `step`, so counts right at the edge of a length range are approximate.
    """
    n_all = has_type.shape[1]
    assert n_all <= 18, "Too many content types for exact counting."        # needs 2^n_all rows
    ticks = np.round(durations / step).astype(int)            # clip lengths in units of `step`
    last  = int(up_to_seconds / step)
    ways  = np.zeros((2 ** n_all, last + 1))
    ways[0, 0] = 1                                            # the empty combination: no types, length 0
    for i in range(len(durations)):
        # The combinations that take clip i: all that still fit, viewed with one axis per content type
        # (index 0 = type absent, 1 = present) so that a type can be switched on for all of them at once.
        moved = ways[:, :last + 1 - ticks[i]].copy().reshape((2,) * n_all + (-1,))
        for j in np.where(has_type[i])[0]:                    # adding this clip adds its content types
            with_type = np.moveaxis(moved, n_all - 1 - j, 0)
            with_type[1] += with_type[0]
            with_type[0] = 0
        ways[:, ticks[i]:] += moved.reshape(2 ** n_all, -1)   # ... and they become longer by the clip's length
    type_count = bit_counts(n_all)
    return np.array([ways[type_count == k].sum(axis=0) for k in range(n_all + 1)])


# ---------- part 2: list the combinations that qualify ----------
def _half_subsets(index, durations, bits, max_seconds):
    """Every subset of these clips that is not already too long: its length, its set of content
    types, and which clips it holds (one bit per clip)."""
    sec, typ, mem = np.zeros(1), np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    for pos, i in enumerate(index):
        fits = sec + durations[i] <= max_seconds
        sec, typ, mem = (np.concatenate([sec, sec[fits] + durations[i]]),
                         np.concatenate([typ, typ[fits] | bits[i]]),
                         np.concatenate([mem, mem[fits] | (np.int64(1) << pos)]))
    return sec, typ, mem


def list_combinations(durations, has_type, min_seconds, max_seconds, min_types):
    """Every combination of clips with a total length inside the window that shows at least `min_types`
    content types. Exact lengths are used. Returns a list of tuples of clip positions.

    The clips are split into two halves. Every sub-selection of each half is listed (a few million
    at most), and the two lists are joined: one from half A with one from half B.
    """
    bits, type_count = type_bits(has_type), bit_counts(has_type.shape[1])
    index_a, index_b = np.arange(0, len(durations), 2), np.arange(1, len(durations), 2)     # even and odd clips
    sec_a, typ_a, mem_a = _half_subsets(index_a, durations, bits, max_seconds)
    sec_b, typ_b, mem_b = _half_subsets(index_b, durations, bits, max_seconds)
    # Half B is sorted by length, so for each entry of half A the partners that bring the total
    # inside the window form one continuous stretch (from `low` up to `high`).
    by_length = np.argsort(sec_b)
    sec_b, typ_b, mem_b = sec_b[by_length], typ_b[by_length], mem_b[by_length]
    low  = np.searchsorted(sec_b, min_seconds - sec_a, side="left")
    high = np.searchsorted(sec_b, max_seconds - sec_a, side="right")

    found = []
    for a in np.where(high > low)[0]:
        for b in np.where(type_count[typ_a[a] | typ_b[low[a]:high[a]]] >= min_types)[0] + low[a]:
            # Turn the two bit patterns back into the list of clips.
            found.append(tuple(sorted([int(index_a[p]) for p in range(len(index_a)) if mem_a[a] >> p & 1]
                                      + [int(index_b[p]) for p in range(len(index_b)) if mem_b[b] >> p & 1])))
    return found


# ---------- the pool and its ranking ----------
def build_pool(combinations, durations, sources, has_type, strength_of, sides, ceiling,
               shot_clip, distances, shot_weight, weights):
    """One row per combination with its properties and the three continuous measures, best first.

    strength: per content type the clearest clip in the compilation, averaged over the types
    coverage: how fully both sides of all stable dimensions are shown
    gap:      average distance from each shot in the set to the nearest selected shot, every shot counted
              by `shot_weight` (see shot_weights), so a clip need not weigh by how often it cuts (lower is better)
    """
    rows = []
    for clips in combinations:
        chosen = list(clips)
        gaps   = coverage_gaps(clips, shot_clip, distances)
        rows.append({"clips": clips, "seconds": durations[chosen].sum(),
                     "n_types": int(has_type[chosen].any(axis=0).sum()), "n_sources": len(set(sources[chosen])),
                     "shortest_clip": durations[chosen].min(),
                     "strength": strength_of[chosen].max(axis=0).mean(),
                     "coverage": side_coverage(clips, sides, ceiling),
                     "gap": np.average(gaps, weights=shot_weight), "worst_shot": int(gaps.argmax())})
    pool = pd.DataFrame(rows)
    # The three measures are put on one scale (mean 0, standard deviation 1 within the pool) and
    # added; the gap is subtracted because a smaller gap is better.
    standard = lambda column: (column - column.mean()) / column.std()
    pool["score"] = (weights["strength"] * standard(pool["strength"]) + weights["coverage"] * standard(pool["coverage"])
                     - weights["gap"] * standard(pool["gap"]))
    return pool.sort_values("score", ascending=False).reset_index(drop=True)      # row 0 = rank 1


def pick_distinct(pool, n_candidates, max_shared_clips):
    """Go down the ranking and keep a compilation only if it shares no more than `max_shared_clips`
    clips with each one already kept; otherwise the list would be near-copies of rank 1."""
    picked = []
    for r in pool.itertuples():
        if all(len(set(r.clips) & set(p.clips)) <= max_shared_clips for p in picked):
            picked.append(r)
        if len(picked) == n_candidates:
            break
    return picked


# ---------- checks on the ranking ----------
def random_compilations(durations, min_seconds, max_seconds, n, rng):
    """`n` compilations drawn without looking at content: clips in random order, as many as fit under
    `max_seconds`, kept if they reach `min_seconds`. The reference for what is typical at this length."""
    found = []
    while len(found) < n:
        order = rng.permutation(len(durations))
        fit   = np.searchsorted(np.cumsum(durations[order]), max_seconds, side="right")
        if durations[order[:fit]].sum() >= min_seconds:
            found.append(order[:fit])
    return found


def robust_ranking(compilations, rankings):
    """Rank `compilations` (the pool or a part of it) under several weightings at once.

    `rankings` maps each weighting to {compilation: rank in the full pool}. Ranks are counted again
    within `compilations`, so a part of the pool is ranked among itself. The robust rank is the worst
    of a compilation's ranks: one that is near the top here is near the top under every weighting.
    Ties are broken by the average rank. Returns a copy, best first, with a column rank_<weighting>
    per weighting and the columns worst_rank and mean_rank.
    """
    out, columns = compilations.copy(), []
    for how, full_ranking in rankings.items():
        order    = sorted(out["clips"], key=full_ranking.get)
        position = {c: r for r, c in enumerate(order, start=1)}
        out[f"rank_{how}"] = [position[c] for c in out["clips"]]
        columns.append(f"rank_{how}")
    out["worst_rank"] = out[columns].max(axis=1)
    out["mean_rank"]  = out[columns].mean(axis=1)
    return out.sort_values(["worst_rank", "mean_rank"], kind="stable").reset_index(drop=True)


def missing_types(clips, has_type, type_names):
    """The content types that none of these clips shows, as text ("none" if every type is shown)."""
    return "; ".join(type_names[~has_type[list(clips)].any(axis=0)]) or "none"
