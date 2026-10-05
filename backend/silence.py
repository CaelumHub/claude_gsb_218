"""
silence.py — Silence detection and trimming.

The module answers two questions:

1. **Where is the silence?**  :func:`detect_silence` streams the file once,
   measures the energy of short blocks, derives an *adaptive* threshold from the
   recording's own noise floor, and tracks silent runs with hysteresis so that
   brief dips, consonant gaps and low-level speech are not mistaken for silence.
2. **What should the result sound like?**  :func:`trim_silence` removes
   leading/trailing silence and either deletes or *compresses* internal gaps to
   a fixed retained length, joining the surviving segments with a short
   equal-gain crossfade so no edit point clicks.

Everything is pure standard-library Python and fully streaming: detection uses
O(blocks) memory with a fixed block size, and rendering reads the source once
while holding back only a few milliseconds of audio for each crossfade join.

Detection design (why quiet speech survives)
--------------------------------------------
* Energy is measured as RMS per ~20 ms block, converted to dB and smoothed with
  a median-3 pre-filter plus a mildly skewed follower (fast attack, fairly
  quick release so genuine short gaps stay visible).
* In adaptive mode an orthogonal **content gate** runs on every block: the
  coefficient of variation of Schmitt-trigger zero-crossing intervals detects
  periodic content (voiced speech, tones, hum down to ~65 Hz via a 60 ms
  rolling context), and a peak/RMS crest factor detects plosives and fricative
  bursts.  A block that is quiet *in energy* but carries structure is never
  silence — this is what keeps normal low-level content from being cut.
* The adaptive threshold is then estimated from the **featureless** blocks
  only (15th percentile = noise floor; 90th percentile = programme level) and
  placed ``margin_db`` above the floor, clamped to stay at least 3 dB below the
  active level.  A manual absolute threshold overrides this (energy gate only).
* Silence is entered when the smoothed level crosses below the threshold and is
  only left again after it rises ``hysteresis_db`` above it — chatter right at
  the threshold stays classified as sound.
* Silent runs shorter than ``min_silence`` are discarded (short speech pauses),
  and two silent runs separated by less than ``min_sound`` are bridged into one
  (a tiny click or mouth noise does not split a real gap).

Extreme inputs are handled explicitly: a file with essentially no silence
yields an empty region list and an unchanged trim result; a file that is almost
entirely silence is flagged ``all_silence`` and the trimmed render falls back to
a short fragment instead of an empty, unplayable WAV.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import audio_io

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

DEFAULT_MIN_SILENCE = 0.30    # s — gaps shorter than this are speech pauses
DEFAULT_MIN_SOUND = 0.06      # s — blips shorter than this bridge a gap
DEFAULT_MARGIN_DB = 10.0      # adaptive threshold sits this far above noise
DEFAULT_HYSTERESIS_DB = 4.0   # must rise this far above threshold to exit
DEFAULT_BLOCK_S = 0.02        # ~20 ms energy blocks
DEFAULT_EDGE_PADDING = 0.02   # s kept next to trimmed edges (avoids clipping)
DEFAULT_KEEP_SECONDS = 0.30   # s retained when compressing internal gaps
DEFAULT_CROSSFADE_MS = 5.0
ENVELOPE_POINTS = 1200

# Content-gate cues (only used in adaptive mode).  A block that is quiet in
# *energy* but carries structure is never silence:
PERIODIC_CV = 0.35            # zero-crossing-interval coefficient of variation
IMPULSE_CREST = 2.75          # block peak/RMS ratio (plosives / bursts)
FLOOR_SAMPLE_MIN_RATIO = 0.10 # need ≥10% "featureless" blocks to trust p15 floor

_PERCENTILE_STRIDE = 200_000  # cap sorting cost on very long files
_FLOOR_DB = -90.0


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        return _FLOOR_DB
    k = (len(sorted_vals) - 1) * pct
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def _median3(a: float, b: float, c: float) -> float:
    return sorted((a, b, c))[1]


def _content_cues(window: Sequence[float]) -> Tuple[float, float, bool]:
    """Return (rms, crest_factor, is_structure) for one analysis window.

    Periodic content (voiced speech, tones, hum) is recognised by the
    coefficient of variation of Schmitt-trigger zero-crossing intervals:
    periodic signals have nearly-regular intervals (CV ≈ 0), whereas noise-floor
    crossings jitter heavily.  Impulsive content (plosives, clicks, fricative
    bursts) is recognised by a high peak/RMS ratio.  Steady noise-floor and
    digital silence trip neither cue.
    """
    n = len(window)
    if n < 8:
        return 0.0, 0.0, False
    s = 0.0
    peak = 0.0
    for x in window:
        s += x * x
        ax = abs(x)
        if ax > peak:
            peak = ax
    rms = math.sqrt(s / n)
    if rms < 1e-9:
        return 0.0, 0.0, False
    crest = peak / rms

    dz = 0.25 * rms
    intervals: List[int] = []
    last = 0
    state = 1 if window[0] >= 0 else -1
    for i, v in enumerate(window):
        if state > 0 and v < -dz:
            if last:
                intervals.append(i - last)
            last = i
            state = -1
        elif state < 0 and v > dz:
            if last:
                intervals.append(i - last)
            last = i
            state = 1
    cv = 9.0
    if len(intervals) >= 4:
        m = sum(intervals) / len(intervals)
        if m > 0:
            var = sum((v - m) ** 2 for v in intervals) / len(intervals)
            cv = math.sqrt(var) / m
    periodic = cv < PERIODIC_CV
    impulsive = crest >= IMPULSE_CREST
    return rms, crest, bool(periodic or impulsive)


def _analyze_blocks(path: str, block_s: float) -> Tuple[List[float], List[bool], int, int, int]:
    """Stream the file once.

    Returns (rms_db per block, structure flag per block, sr, total_frames,
    block_size).  Structure is measured on a rolling ~60 ms context window so
    low-pitched voiced content (down to ~65 Hz) still gets recognised.
    """
    carry: List[float] = []
    context: List[float] = []
    levels: List[float] = []
    content: List[bool] = []
    sr = 44100
    total = 0
    with audio_io.WavReader(path) as r:
        sr = r.sr
        total = r.nframes
        block = max(64, int(round(sr * block_s)))
        ctx_len = max(block, int(round(sr * 0.06)))

        def emit(seg: Sequence[float]) -> None:
            s = 0.0
            for x in seg:
                s += x * x
            rms = math.sqrt(s / len(seg))
            levels.append(20.0 * math.log10(rms + 1e-10))
            context.extend(seg)
            if len(context) > ctx_len:
                del context[:len(context) - ctx_len]
            _, _, structured = _content_cues(context)
            content.append(structured)

        for chunk in r.iter_chunks():
            carry.extend(audio_io.to_mono(chunk))
            while len(carry) >= block:
                seg = carry[:block]
                del carry[:block]
                emit(seg)
        if len(carry) >= block // 2:
            emit(carry)
    return levels, content, sr, total, block


def _smooth_levels(levels: Sequence[float]) -> List[float]:
    """Median-3 de-click plus a slightly-skewed follower.

    Release is deliberately fairly fast: the *min_silence* duration filter is
    what separates a real gap from a momentary dip (not the envelope time
    constant), so a slow release would only blur genuine short silences.
    """
    n = len(levels)
    med: List[float] = []
    for i in range(n):
        if 0 < i < n - 1:
            med.append(_median3(levels[i - 1], levels[i], levels[i + 1]))
        else:
            med.append(levels[i])
    out: List[float] = [0.0] * n
    if n:
        e = med[0]
        for i, v in enumerate(med):
            alpha = 0.55 if v > e else 0.35
            e += alpha * (v - e)
            out[i] = e
    return out


def _adaptive_threshold(levels: Sequence[float], content: Sequence[bool],
                        margin_db: float) -> Tuple[float, float, float]:
    """Return (threshold_db, noise_floor_db, active_level_db).

    The noise floor is estimated only from blocks that carry no structure
    (the content gate): a recording with quiet-but-real passages and no true
    floor would otherwise estimate its quietest *programme* as the floor.
    """
    floor_vals = [v for v, c in zip(levels, content) if not c]
    if len(floor_vals) < max(20, int(len(levels) * FLOOR_SAMPLE_MIN_RATIO)):
        # Almost everything is structured content: trust the very quietest
        # levels instead (e.g. tiny gaps between words), never the body level.
        floor_vals = list(levels)
        floor_vals.sort()
        noise = _percentile(floor_vals, 0.02)
    else:
        sample = floor_vals
        if len(sample) > _PERCENTILE_STRIDE:
            step = len(sample) // _PERCENTILE_STRIDE
            sample = sample[::max(1, step)]
        sample = sorted(sample)
        noise = _percentile(sample, 0.15)
    all_sorted = sorted(levels)
    active = _percentile(all_sorted, 0.90)

    threshold = noise + margin_db
    # Guard: the threshold must remain well below the genuine programme level,
    # otherwise quiet content would be eaten on noisy, low-SNR recordings.
    if active - threshold < 6.0:
        threshold = noise + max(4.0, (active - noise) * 0.5)
    threshold = min(threshold, active - 3.0)
    threshold = max(threshold, _FLOOR_DB)
    return threshold, noise, active


def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """Run-length encode a bool sequence into [start, end_exclusive) pairs."""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def detect_silence(path: str, threshold_db: Optional[float] = None,
                   min_silence: float = DEFAULT_MIN_SILENCE,
                   min_sound: float = DEFAULT_MIN_SOUND,
                   margin_db: float = DEFAULT_MARGIN_DB,
                   hysteresis_db: float = DEFAULT_HYSTERESIS_DB,
                   block_s: float = DEFAULT_BLOCK_S,
                   envelope_points: int = ENVELOPE_POINTS) -> Dict:
    """Detect silent passages in an audio file.

    Two gates decide each block: an *energy* gate (smoothed RMS against the
    threshold, with hysteresis) and — in adaptive mode — a *content* gate
    (periodicity/impulsiveness cues).  Quiet structured content can never be
    silence even if it dips below the energy threshold.

    ``threshold_db`` selects manual energy-only mode; ``None`` (default)
    derives the threshold adaptively from the recording's noise floor.

    Returns a JSON-serialisable report — see the module docstring and the
    region/stat keys consumed by the front-end.
    """
    min_silence = max(0.01, float(min_silence))
    min_sound = max(0.0, float(min_sound))
    margin_db = float(margin_db)
    hysteresis_db = max(0.0, float(hysteresis_db))

    levels, content, sr, total_frames, block = _analyze_blocks(path, block_s)
    duration = total_frames / sr if sr else 0.0

    if not levels:
        return _empty_report(sr, duration, total_frames, block_s,
                             threshold_db, None, None, envelope_points)

    smoothed = _smooth_levels(levels)

    if threshold_db is None:
        adaptive = True
        threshold, noise, active = _adaptive_threshold(levels, content, margin_db)
    else:
        adaptive = False
        threshold = float(threshold_db)
        # Still report the recording's own floor/active level as reference info.
        _, noise, active = _adaptive_threshold(levels, content, margin_db)

    # State machine: enter silence below the energy threshold (content gate
    # vetoes); exit either on structured content or on the hysteresis rise.
    flags = [False] * len(smoothed)
    silent = smoothed[0] <= threshold and not (adaptive and content[0])
    for i, v in enumerate(smoothed):
        structured = adaptive and content[i]
        if silent:
            if structured or v > threshold + hysteresis_db:
                silent = False
        else:
            if v <= threshold and not structured:
                silent = True
        flags[i] = silent

    raw_runs = _runs(flags)

    # Bridge: a sound run shorter than min_sound between two silent runs is
    # reclaimed (isolated clicks must not split a genuine long silence).
    bridge_blocks = int(round(min_sound / block_s))
    bridged: List[Tuple[int, int]] = []
    for run in raw_runs:
        if bridged:
            ps, pe = bridged[-1]
            if run[0] - pe <= bridge_blocks:
                bridged[-1] = (ps, run[1])
                continue
        bridged.append(run)

    # Minimum-duration filter (drops short speech pauses).
    min_blocks = max(1, int(round(min_silence / block_s)))
    kept = [(a, b) for (a, b) in bridged if b - a >= min_blocks]

    tol = max(1, int(round(1.5 * block_s)))  # boundary snap tolerance
    regions: List[Dict] = []
    total_silence_frames = 0
    for idx, (a, b) in enumerate(kept):
        f0 = a * block
        f1 = min(total_frames, b * block)
        lead = f0 <= tol
        trail = f1 >= total_frames - tol
        if lead and trail:
            position = "all"
        elif lead:
            position = "leading"
        elif trail:
            position = "trailing"
        else:
            position = "internal"
        seg = smoothed[a:b]
        regions.append({
            "index": idx,
            "start": f0 / sr,
            "end": f1 / sr,
            "duration": (f1 - f0) / sr,
            "position": position,
            "mean_db": sum(seg) / len(seg),
            "min_db": min(seg),
            "max_db": max(seg),
        })
        total_silence_frames += f1 - f0

    ratio = total_silence_frames / total_frames if total_frames else 0.0
    # "All silence" covers the fully-silent case as well as a recording with a
    # tiny scrap of content in an ocean of silence.
    all_silence = (
        ratio > 0.95
        and (total_frames - total_silence_frames) < sr * 0.30
        and any(r["position"] in ("leading", "trailing", "all") for r in regions)
    )
    no_silence = not regions or ratio < 0.005

    envelope = _downsample_envelope(smoothed, block, envelope_points)

    return {
        "sr": sr,
        "frames": total_frames,
        "duration": duration,
        "block_s": block_s,
        "threshold": {
            "adaptive": adaptive,
            "threshold_db": round(threshold, 2),
            "noise_floor_db": round(noise, 2),
            "active_db": round(active, 2),
            "margin_db": margin_db,
            "hysteresis_db": hysteresis_db,
        },
        "params": {
            "threshold_db": None if adaptive else round(threshold, 2),
            "min_silence": min_silence,
            "min_sound": min_sound,
            "margin_db": margin_db,
            "hysteresis_db": hysteresis_db,
            "block_s": block_s,
        },
        "regions": regions,
        "stats": {
            "count": len(regions),
            "leading": sum(1 for r in regions if r["position"] == "leading"),
            "trailing": sum(1 for r in regions if r["position"] == "trailing"),
            "internal": sum(1 for r in regions if r["position"] == "internal"),
            "total_silence": total_silence_frames / sr if sr else 0.0,
            "ratio": ratio,
        },
        "envelope": envelope,
        "all_silence": all_silence,
        "no_silence": no_silence,
    }


def _downsample_envelope(levels: Sequence[float], block_s: float,
                         points: int) -> Dict:
    """Average the block levels into a compact envelope for UI rendering."""
    n = len(levels)
    points = max(64, int(points))
    bucket = max(1, (n + points - 1) // points)
    times: List[float] = []
    db: List[float] = []
    for i in range(0, n, bucket):
        seg = levels[i:i + bucket]
        center = (i + len(seg) / 2.0) * block_s
        times.append(center)
        db.append(sum(seg) / len(seg))
    return {"times": times, "db": db}


def _empty_report(sr: int, duration: float, frames: int, block_s: float,
                  threshold_db: Optional[float], noise: Optional[float],
                  active: Optional[float], envelope_points: int) -> Dict:
    adaptive = threshold_db is None
    return {
        "sr": sr, "frames": frames, "duration": duration, "block_s": block_s,
        "threshold": {
            "adaptive": adaptive,
            "threshold_db": round(threshold_db if threshold_db is not None else _FLOOR_DB, 2),
            "noise_floor_db": round(noise if noise is not None else _FLOOR_DB, 2),
            "active_db": round(active if active is not None else _FLOOR_DB, 2),
            "margin_db": DEFAULT_MARGIN_DB, "hysteresis_db": DEFAULT_HYSTERESIS_DB,
        },
        "params": {
            "threshold_db": threshold_db, "min_silence": DEFAULT_MIN_SILENCE,
            "min_sound": DEFAULT_MIN_SOUND, "margin_db": DEFAULT_MARGIN_DB,
            "hysteresis_db": DEFAULT_HYSTERESIS_DB, "block_s": block_s,
        },
        "regions": [],
        "stats": {"count": 0, "leading": 0, "trailing": 0, "internal": 0,
                  "total_silence": 0.0, "ratio": 0.0},
        "envelope": {"times": [], "db": []},
        "all_silence": frames == 0, "no_silence": True,
    }


# --------------------------------------------------------------------------- #
# Trim / compress rendering
# --------------------------------------------------------------------------- #

def _region_actions(report_regions: Sequence[Dict], trim_edges: bool,
                    middle: str, selection: Optional[Sequence[int]]) -> Dict[int, str]:
    """Decide remove / keep / leave for every detected region index."""
    selected = set(selection) if selection is not None else None
    actions: Dict[int, str] = {}
    for r in report_regions:
        idx = r["index"]
        if selected is not None and idx not in selected:
            actions[idx] = "leave"
            continue
        pos = r["position"]
        if pos in ("leading", "trailing", "all"):
            actions[idx] = "remove" if (selected is not None or trim_edges) else "leave"
        else:
            actions[idx] = middle if middle in ("keep", "remove", "leave") else "keep"
    return actions


def _build_keep_ranges(total: int, sr: int, regions: Sequence[Dict],
                       actions: Dict[int, str], edge_padding_s: float,
                       keep_seconds: float) -> Tuple[List[Tuple[int, int]], List[int], bool]:
    """Complement of the dropped intervals -> source frame ranges to render.

    Returns (ranges, applied_indices, all_silence_fallback).
    """
    edge_pad = max(0, int(round(edge_padding_s * sr)))
    keep_f = max(0, int(round(keep_seconds * sr)))
    drops: List[Tuple[int, int]] = []
    applied: List[int] = []

    for r in regions:
        idx = r["index"]
        act = actions.get(idx, "leave")
        if act == "leave":
            continue
        a = max(0, int(round(r["start"] * sr)))
        b = min(total, int(round(r["end"] * sr)))
        pos = r["position"]
        applied.append(idx)

        if act == "remove":
            if pos == "leading":
                drops.append((0, max(0, b - edge_pad)))
            elif pos == "trailing":
                drops.append((min(total, a + edge_pad), total))
            else:  # internal or "all"
                drops.append((a, b))
        elif act == "keep" and pos == "internal":
            length = b - a
            if keep_f >= length:
                # Nothing to do: the gap is already short enough.
                applied.pop()
                continue
            mid = (a + b) // 2
            k0 = max(a, mid - keep_f // 2)
            k1 = min(b, k0 + keep_f)
            if k0 > a:
                drops.append((a, k0))
            if b > k1:
                drops.append((k1, b))

    # Merge dropped intervals.
    drops.sort()
    merged: List[Tuple[int, int]] = []
    for a, b in drops:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    # Complement within [0, total).
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for a, b in merged:
        if a > cursor:
            ranges.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < total:
        ranges.append((cursor, total))

    if not ranges:
        # The whole file was removed — keep a short centered fragment so the
        # result stays a playable, non-empty WAV.
        fb = max(1, int(0.25 * sr))
        c0 = max(0, (total - fb) // 2)
        return [(c0, min(total, c0 + fb))], applied, True
    return ranges, applied, False


def render_kept_ranges(src_path: str, dst_path: str,
                       ranges: Sequence[Tuple[int, int]],
                       crossfade_s: float = DEFAULT_CROSSFADE_MS / 1000.0) -> int:
    """Stream ``src`` once and write only ``ranges`` (frame intervals).

    Consecutive ranges are joined with an equal-gain linear crossfade of
    ``crossfade_s`` seconds; touching ranges are merged beforehand and need
    none.  Bulk chunk slicing keeps the hot path in C-level list operations —
    only the few-millisecond joins are computed sample by sample.  Returns the
    number of output frames.
    """
    with audio_io.WavReader(src_path) as r:
        sr, ch, total = r.sr, r.channels, r.nframes
        xf = max(0, int(round(crossfade_s * sr)))

        clean: List[Tuple[int, int]] = []
        for s, e in ranges:
            s = max(0, int(s))
            e = min(total, int(e))
            if e <= s:
                continue
            if clean and s <= clean[-1][1]:
                clean[-1] = (clean[-1][0], max(clean[-1][1], e))
            else:
                clean.append((s, e))
        ranges = clean
        if not ranges:
            raise ValueError("no audio remains after trimming")

        # Tail frames held back from range i for its join with range i + 1.
        # A join is only scheduled when *both* ranges are longer than the
        # fade: otherwise (e.g. a short retained-silence island between two
        # programme segments) the whole short range is copied as-is and the
        # boundary is a clean, click-free abutment anyway.
        join_len: List[int] = [0] * len(ranges)
        if xf > 0:
            for i in range(len(ranges) - 1):
                gap = ranges[i + 1][0] - ranges[i][1]
                if gap > 0:
                    need = 2 * xf
                    if ranges[i][1] - ranges[i][0] >= need and \
                            ranges[i + 1][1] - ranges[i + 1][0] >= xf:
                        join_len[i] = xf

        with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
            # State for the range `ki` currently being processed:
            #   joining=False / need_tail=False — bulk-copy the range body;
            #   need_tail=True  — the body is done; buffer the final
            #                     `join_len[ki]` frames (may span chunks);
            #   joining=True    — skip the dropped gap and crossfade the
            #                     pending tail into range ki+1's head.
            ki = 0
            joining = False
            need_tail = False
            pending: List[List[float]] = [[] for _ in range(ch)]
            pos = 0
            n_ranges = len(ranges)

            for chunk in r.iter_chunks():
                n = len(chunk[0])
                c0, c1 = pos, pos + n
                seg_out: List[List[float]] = [[] for _ in range(ch)]
                cursor = c0

                while cursor < c1 and ki < n_ranges:
                    s, e = ranges[ki]

                    if joining:
                        if ki + 1 >= n_ranges:
                            ki = n_ranges
                            break
                        ts, te = ranges[ki + 1]
                        if cursor < ts:
                            cursor = min(c1, ts)
                            if cursor >= c1:
                                break
                        m = len(pending[0])
                        take = min(m, max(0, min(c1, te) - cursor))
                        if take <= 0:
                            break
                        base = cursor - c0
                        for c in range(ch):
                            row_p, row_c = pending[c], chunk[c]
                            seg_out[c].extend(
                                row_p[j] * (1.0 - (j + 1) / m)
                                + row_c[base + j] * ((j + 1) / m)
                                for j in range(take))
                        pending = [p[take:] for p in pending]
                        cursor += take
                        if pending[0]:
                            break  # fade continues in the next read chunk
                        joining = False
                        need_tail = False
                        ki += 1   # continue with the target range's body
                        continue

                    if need_tail:
                        # Buffer the held tail (starts at `cursor`).
                        hold = join_len[ki]
                        have = len(pending[0])
                        end_here = min(c1, s + (e - s))  # == min(c1, e)
                        end_here = min(c1, e)
                        take = min(hold - have, end_here - cursor)
                        if take > 0:
                            a, b = cursor - c0, cursor - c0 + take
                            for c in range(ch):
                                pending[c].extend(chunk[c][a:b])
                            cursor += take
                        if len(pending[0]) >= hold:
                            need_tail = False
                            joining = True
                            continue
                        break  # tail finishes in the next read chunk

                    if cursor < s:
                        cursor = min(c1, s)
                        if cursor >= c1:
                            break

                    hold = join_len[ki]
                    if not hold:
                        # No join after this range: copy verbatim to its end.
                        end_here = min(c1, e)
                        if end_here > cursor:
                            a, b = cursor - c0, end_here - c0
                            for c in range(ch):
                                seg_out[c].extend(chunk[c][a:b])
                            cursor = end_here
                        if cursor >= e:
                            ki += 1
                            need_tail = False
                        continue
                    body_end = e - hold
                    if c1 <= body_end:
                        # Rest of this chunk is all body (cursor == s here or
                        # already inside the body).
                        a = cursor - c0
                        for c in range(ch):
                            seg_out[c].extend(chunk[c][a:c1 - c0])
                        cursor = c1
                        break  # tail (and possibly more body) in later chunks
                    if cursor < body_end:
                        a, b = cursor - c0, body_end - c0
                        for c in range(ch):
                            seg_out[c].extend(chunk[c][a:b])
                        cursor = body_end
                    # cursor is now at the tail start (within this chunk).
                    take = min(hold, e - cursor)
                    if take > 0:
                        a, b = cursor - c0, cursor - c0 + take
                        for c in range(ch):
                            pending[c].extend(chunk[c][a:b])
                        cursor += take
                    if len(pending[0]) >= hold:
                        joining = True
                        continue
                    need_tail = True
                    break

                if seg_out[0]:
                    w.write_chunk(seg_out)
                pos = c1
                if ki >= n_ranges and not pending[0]:
                    break

            # Defensive flush (every pending tail is normally consumed by the
            # join with the following range).
            if pending[0]:
                w.write_chunk(pending)

    with audio_io.WavReader(dst_path) as r:
        return r.nframes


def trim_silence(src_path: str, dst_path: str,
                 threshold_db: Optional[float] = None,
                 min_silence: float = DEFAULT_MIN_SILENCE,
                 min_sound: float = DEFAULT_MIN_SOUND,
                 margin_db: float = DEFAULT_MARGIN_DB,
                 hysteresis_db: float = DEFAULT_HYSTERESIS_DB,
                 trim_edges: bool = True,
                 middle: str = "keep",
                 keep_seconds: float = DEFAULT_KEEP_SECONDS,
                 edge_padding: float = DEFAULT_EDGE_PADDING,
                 crossfade_ms: float = DEFAULT_CROSSFADE_MS,
                 selection: Optional[Sequence[int]] = None) -> Tuple[Dict, Dict]:
    """Detect, then render a trimmed/compressed copy.

    ``middle`` is one of ``"keep"`` (compress internal gaps to
    ``keep_seconds``), ``"remove"`` (delete them entirely) or ``"leave"``.
    ``selection`` optionally restricts all actions to a list of region indices
    chosen by the user.

    Returns ``(detection_report, result_summary)``.
    """
    report = detect_silence(
        src_path, threshold_db=threshold_db, min_silence=min_silence,
        min_sound=min_sound, margin_db=margin_db, hysteresis_db=hysteresis_db,
    )
    sr = report["sr"]
    total = report["frames"]

    actions = _region_actions(report["regions"], trim_edges, middle, selection)
    ranges, applied, all_fallback = _build_keep_ranges(
        total, sr, report["regions"], actions, edge_padding, keep_seconds)

    out_frames = render_kept_ranges(
        src_path, dst_path, ranges, crossfade_s=max(0.0, crossfade_ms) / 1000.0)

    removed = max(0.0, (total - out_frames) / sr if sr else 0.0)
    summary = {
        "regions_applied": applied,
        "removed_seconds": removed,
        "original_duration": report["duration"],
        "result_duration": out_frames / sr if sr else 0.0,
        "result_frames": out_frames,
        "all_silence": all_fallback,
        "mostly_silence": report["all_silence"],
        "trim_edges": bool(trim_edges),
        "middle": middle,
        "keep_seconds": keep_seconds,
    }
    return report, summary
